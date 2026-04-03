"""Gemini API client with rate limiting, retry, and structured JSON output."""

import json
import logging
import re
import time
from collections import deque
from typing import Any

import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiClient:
    """Thin wrapper around google.generativeai with rate-limit and retry logic.

    Parameters
    ----------
    api_key:
        Gemini API key.  Must be non-empty.
    model_name:
        Model identifier, e.g. ``"gemini-2.5-flash"``.
    rpm_limit:
        Maximum requests per minute (default 15).
    daily_limit:
        Maximum requests per calendar day (default 1500).
    temperature:
        Sampling temperature forwarded to the model (default 0.1).
    max_output_tokens:
        Token cap for model responses (default 2048).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        rpm_limit: int = 15,
        daily_limit: int = 1500,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key must not be empty.")

        self.model_name = model_name
        self.rpm_limit = rpm_limit
        self.daily_limit = daily_limit
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Rate-limit bookkeeping
        self._minute_window: deque[float] = deque()   # timestamps (last 60 s)
        self._daily_count: int = 0

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def requests_today(self) -> int:
        """Total requests made during this session (proxy for daily usage)."""
        return self._daily_count

    @property
    def requests_this_minute(self) -> int:
        """Requests made within the last 60 seconds."""
        self._purge_old_timestamps()
        return len(self._minute_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> str:
        """Send a prompt (optionally with images) and return the model's text.

        Parameters
        ----------
        prompt:
            The user-facing text prompt.
        system:
            Optional system instruction prepended to the content list.
        images:
            Optional list of image objects (e.g. PIL.Image instances) to
            include alongside the prompt.
        temperature:
            Override the instance-level temperature for this call.
        max_retries:
            Number of retry attempts with exponential back-off on failure.

        Returns
        -------
        str
            The model's text response.
        """
        self._wait_for_capacity()

        content: list[Any] = []
        if system:
            content.append(system)
        content.append(prompt)
        if images:
            content.extend(images)

        gen_config = genai.types.GenerationConfig(
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._model.generate_content(
                    content,
                    generation_config=gen_config,
                )
                self._record_request()
                return response.text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Gemini request failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini API failed after {max_retries} attempts."
        ) from last_exc

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Generate text and attempt to parse the result as JSON.

        Parsing strategy (in order):
        1. Direct ``json.loads`` on the full response.
        2. Extract content from a markdown code-fence (```json ... ```).
        3. Regex search for the first ``{...}`` block in the response.
        4. Fall back to ``{"raw_response": <text>}``.

        Returns
        -------
        dict
            Parsed JSON dict, or ``{"raw_response": text}`` on total failure.
        """
        text = self.generate_text(
            prompt, system=system, images=images, temperature=temperature
        )

        # Strategy 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: code fences  ```json ... ``` or ``` ... ```
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: first {...} block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 4: raw fallback
        logger.warning("generate_json: could not parse response as JSON.")
        return {"raw_response": text}

    def quota_remaining(self) -> dict:
        """Return a snapshot of remaining quota.

        Returns
        -------
        dict
            Keys: ``daily_remaining``, ``rpm_remaining``, ``daily_used``.
        """
        used_today = self._daily_count
        used_this_minute = self.requests_this_minute  # also purges stale ts
        return {
            "daily_used": used_today,
            "daily_remaining": max(0, self.daily_limit - used_today),
            "rpm_remaining": max(0, self.rpm_limit - used_this_minute),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _purge_old_timestamps(self) -> None:
        """Remove timestamps older than 60 seconds from the minute window."""
        cutoff = time.monotonic() - 60.0
        while self._minute_window and self._minute_window[0] < cutoff:
            self._minute_window.popleft()

    def _wait_for_capacity(self) -> None:
        """Block until both RPM and daily limits have capacity."""
        # Daily limit check
        if self._daily_count >= self.daily_limit:
            raise RuntimeError(
                f"Daily request limit ({self.daily_limit}) reached."
            )

        # RPM limit: wait until the oldest request in the window is > 60 s old
        while True:
            self._purge_old_timestamps()
            if len(self._minute_window) < self.rpm_limit:
                break
            sleep_for = 60.0 - (time.monotonic() - self._minute_window[0]) + 0.05
            if sleep_for > 0:
                logger.debug("RPM limit reached, sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)

    def _record_request(self) -> None:
        """Record that a request completed successfully."""
        self._daily_count += 1
        self._minute_window.append(time.monotonic())
