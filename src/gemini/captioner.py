"""Image captioner using Gemini for search-index descriptions."""

import time
import logging
from typing import List

from src.gemini.client import GeminiClient

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe this image in detail for a search index. "
    "Include: 1. What the image shows 2. Any text visible "
    "3. If chart: type, axes, trends 4. If diagram: components and relationships. "
    "Keep under 200 words."
)


class ImageCaptioner:
    """Generate descriptive captions for images using the Gemini API.

    Parameters
    ----------
    api_key:
        Gemini API key.
    rpm_limit:
        Maximum requests per minute passed to the underlying GeminiClient.
    """

    def __init__(self, api_key: str, rpm_limit: int = 8) -> None:
        self._client = GeminiClient(api_key=api_key, rpm_limit=rpm_limit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def caption(self, image) -> str:
        """Return a detailed caption for a single image.

        On any failure the fallback caption (dimensions + unavailability note)
        is returned instead of propagating the exception.

        Parameters
        ----------
        image:
            A PIL.Image instance (or any object accepted by GeminiClient).

        Returns
        -------
        str
            Caption text or fallback string.
        """
        try:
            return self._client.generate_text(_CAPTION_PROMPT, images=[image])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Caption generation failed: %s — using fallback.", exc)
            return self._fallback_caption(image)

    def caption_batch(self, images, delay_between: float = 0.5) -> List[str]:
        """Caption each image in *images* with a short sleep between requests.

        Parameters
        ----------
        images:
            Iterable of image objects.
        delay_between:
            Seconds to sleep between successive caption calls (default 0.5).

        Returns
        -------
        List[str]
            One caption string per input image, in the same order.
        """
        captions: List[str] = []
        for idx, image in enumerate(images):
            captions.append(self.caption(image))
            if idx < len(images) - 1:
                time.sleep(delay_between)
        return captions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback_caption(self, image) -> str:
        """Build a minimal caption when the API is unavailable.

        Returns
        -------
        str
            "Image (WxH, mode). Visual content — detailed caption unavailable."
        """
        try:
            width, height = image.size
            mode = image.mode
            return f"Image ({width}x{height}, {mode}). Visual content — detailed caption unavailable."
        except Exception:  # noqa: BLE001
            return "Image (unknown size, unknown mode). Visual content — detailed caption unavailable."
