"""LLM factory for the analyst graph.

Central rule of this package: **``get_llm()`` returning ``None`` is a supported
mode, not an error.**  Every node that would call an LLM has a deterministic
heuristic fallback, so the graph runs end to end with no API key.  That keeps
the test suite offline and turns "LLM vs heuristic" into something we can
measure as a LangSmith experiment rather than assume.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE

logger = logging.getLogger(__name__)


# Provider defaults.  Gemini's free tier allows 20 requests/day per model and
# the graph spends ~8 calls per question, so a 25-example evaluation cannot run
# on it; Groq's free tier is far larger, which is why it is preferred when both
# are configured.
# gpt-oss-120b is the default because the graph needs `with_structured_output`
# for routing, grading and verification, and the 20b variant fails it with
# "Failed to parse tool call arguments as JSON".
_PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "groq": "openai/gpt-oss-120b",
    "google_genai": GEMINI_MODEL,
}

# Credential env var per provider, in autodetection order.
_PROVIDER_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("groq", ("GROQ_API_KEY",)),
    ("google_genai", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
)


def _api_key() -> str:
    """Resolve a Gemini key from the environment, then settings."""
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or GEMINI_API_KEY
        or ""
    ).strip()


def _provider_key(provider: str) -> str:
    """The configured credential for ``provider``, or ``""``."""
    if provider == "google_genai":
        return _api_key()
    for name, env_names in _PROVIDER_KEYS:
        if name == provider:
            for env_name in env_names:
                value = (os.getenv(env_name) or "").strip()
                if value:
                    return value
    return ""


def _base_url() -> str:
    """A self-hosted OpenAI-compatible endpoint, if one is configured."""
    return (os.getenv("GRAPH_LLM_BASE_URL") or "").strip()


def _extra_body() -> dict:
    """Provider-specific request fields, as JSON in ``GRAPH_LLM_EXTRA_BODY``.

    Reasoning models put their chain of thought in ``content``, which lands
    verbatim in synthesised answers and is then scored as if it were the
    answer.  Qwen3.5 on vLLM turns it off with
    ``{"chat_template_kwargs": {"enable_thinking": false}}``; other servers
    spell it differently, so this stays a pass-through rather than a flag.
    """
    raw = (os.getenv("GRAPH_LLM_EXTRA_BODY") or "").strip()
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except ValueError as exc:
        logger.warning("GRAPH_LLM_EXTRA_BODY is not valid JSON, ignoring: %s", exc)
        return {}
    if not isinstance(body, dict):
        logger.warning("GRAPH_LLM_EXTRA_BODY must be a JSON object, ignoring.")
        return {}
    return body


def _vertex_config() -> dict:
    """Vertex AI settings, when the environment asks for them.

    On Google Cloud (Cloud Run, GKE, a VM) Gemini is reached through Vertex AI
    with the runtime's own service account -- no API key exists or is needed.
    Enabled by ``GOOGLE_GENAI_USE_VERTEXAI=true`` plus ``GOOGLE_CLOUD_PROJECT``;
    ``GOOGLE_CLOUD_LOCATION`` defaults to us-central1.
    """
    flag = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()
    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    if flag not in ("1", "true", "yes") or not project:
        return {}
    return {
        "vertexai": True,
        "project": project,
        "location": (os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1").strip(),
    }


def resolve_provider() -> tuple[Optional[str], Optional[str]]:
    """``(provider, model)`` for the configured LLM, or ``(None, None)``.

    Pure resolution -- constructs nothing, so it is cheap and testable.
    ``GRAPH_LLM_PROVIDER`` / ``GRAPH_LLM_MODEL`` override autodetection.  The
    legacy ``LLM_PROVIDER`` setting is deliberately *not* consulted: it belongs
    to the v1 pipeline and already carries an unrelated value.
    """
    override = (os.getenv("GRAPH_LLM_PROVIDER") or "").strip().lower()
    model_override = (os.getenv("GRAPH_LLM_MODEL") or "").strip()

    if override:
        provider = override
    elif _base_url():
        # A self-hosted server has no daily budget, so prefer it over the
        # metered providers. Gemini allows 20 requests/day and Groq 200,000
        # tokens/day per model, against ~8 calls per question.
        provider = "openai"
    else:
        provider = next(
            (name for name, _ in _PROVIDER_KEYS if _provider_key(name)), ""
        )
        if not provider and _vertex_config():
            provider = "google_genai"

    if not provider:
        return None, None
    return provider, model_override or _PROVIDER_DEFAULT_MODEL.get(provider)


# Gemini 3 thinks at "high" unless told otherwise. That is right for writing
# the answer and wasted on yes/no judgements: grading one passage for
# relevance, picking specialists, checking groundedness. Those run five to
# fifteen times per question, and at "high" they made a demo question take
# 40-95 seconds. "low" is accepted by every Gemini 3 model.
_FAST_THINKING_DEFAULT = "low"
_THINKING_LEVELS = ("minimal", "low", "medium", "high")


def _supports_thinking_level(provider: Optional[str], model: Optional[str]) -> bool:
    """Only Gemini 3+ takes ``thinking_level``; 2.5 uses a token budget."""
    name = (model or "").lower().rsplit("/", 1)[-1]
    return provider == "google_genai" and name.startswith("gemini-") and not name.startswith(
        ("gemini-1", "gemini-2")
    )


def fast_thinking_level() -> Optional[str]:
    """Thinking level for short structured judgements, or ``None`` to leave it.

    ``GRAPH_LLM_THINKING_FAST`` overrides the default; ``off`` disables it.
    """
    provider, model = resolve_provider()
    if not _supports_thinking_level(provider, model):
        return None
    raw = (os.getenv("GRAPH_LLM_THINKING_FAST") or _FAST_THINKING_DEFAULT).strip().lower()
    if raw in ("", "off", "none", "default"):
        return None
    if raw not in _THINKING_LEVELS:
        logger.warning("GRAPH_LLM_THINKING_FAST=%r is not one of %s; ignoring.",
                       raw, ", ".join(_THINKING_LEVELS))
        return None
    return raw


@lru_cache(maxsize=8)
def get_llm(
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    thinking: Optional[str] = None,
) -> Optional[Any]:
    """Return a chat model, or ``None`` when no key is configured.

    Cached per (temperature, model, thinking) so nodes can call this freely.
    Call ``get_llm.cache_clear()`` after changing the environment.
    ``thinking`` is a Gemini 3 ``thinking_level``; other models ignore it.
    """
    provider, default_model = resolve_provider()
    if provider is None:
        return None

    base_url = _base_url()
    extra: dict = {}
    if base_url and provider == "openai":
        # The server names its own models, so guessing one would 404.
        if not (model or default_model):
            logger.warning("GRAPH_LLM_BASE_URL is set but GRAPH_LLM_MODEL is "
                           "not; falling back to heuristics.")
            return None
        # A local server needs no credential, but the SDK insists on one.
        extra = {"base_url": base_url,
                 "api_key": os.getenv("GRAPH_LLM_API_KEY") or "not-needed"}
        body = _extra_body()
        if body:
            extra["extra_body"] = body
    elif provider == "google_genai" and not _provider_key(provider) and _vertex_config():
        extra = _vertex_config()
    elif not _provider_key(provider):
        return None

    if thinking and _supports_thinking_level(provider, model or default_model):
        extra["thinking_level"] = thinking

    temp = GEMINI_TEMPERATURE if temperature is None else temperature
    try:
        from langchain.chat_models import init_chat_model

        chat = init_chat_model(
            model or default_model,
            model_provider=provider,
            temperature=temp,
            **extra,
        )
    except Exception as exc:  # unknown provider, missing extra, bad key shape
        logger.warning("LLM unavailable, falling back to heuristics: %s", exc)
        return None

    # Free tiers are request-capped, so pace calls rather than absorb 429s.
    rpm = (os.getenv("GRAPH_LLM_RPM") or "").strip()
    if rpm:
        try:
            from langchain_core.rate_limiters import InMemoryRateLimiter

            chat.rate_limiter = InMemoryRateLimiter(
                requests_per_second=float(rpm) / 60.0,
                check_every_n_seconds=0.5,
                max_bucket_size=1,
            )
        except Exception as exc:
            logger.warning("Rate limiter unavailable: %s", exc)
    return chat


def llm_available() -> bool:
    """True when a real LLM is wired up."""
    return get_llm() is not None


def llm_mode() -> str:
    """``"llm"`` or ``"heuristic"`` — stamped onto LangSmith run metadata."""
    return "llm" if llm_available() else "heuristic"


def get_structured_llm(
    schema: type[BaseModel],
    temperature: Optional[float] = None,
) -> Optional[Any]:
    """Return a model constrained to ``schema``, or ``None`` in heuristic mode.

    Structured calls are the graph's short judgements, so they run at
    :func:`fast_thinking_level`.
    """
    llm = get_llm(temperature, None, fast_thinking_level())
    if llm is None:
        return None
    try:
        return llm.with_structured_output(schema)
    except Exception as exc:
        logger.warning("Structured output unavailable for %s: %s", schema.__name__, exc)
        return None
