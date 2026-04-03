"""Image document loader for the RAG ingestion pipeline."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LoaderResult:
    """Container returned by image loader functions."""

    text_documents: List[Document] = field(default_factory=list)
    visual_assets: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_doc_id() -> str:
    """Generate a short unique document identifier."""
    return uuid.uuid4().hex[:12]


def _ocr_image(image: Image.Image) -> str:
    """Attempt OCR on *image* using pytesseract.

    Returns the extracted text string, or an empty string on any failure
    (including pytesseract not being installed).
    """
    try:
        import pytesseract  # type: ignore

        text: str = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:  # noqa: BLE001 – covers ImportError and runtime errors
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_image(file_path: str, *, do_ocr: bool = True) -> LoaderResult:
    """Load an image file and return a :class:`LoaderResult`.

    Args:
        file_path: Absolute or relative path to the image file.
        do_ocr:    When *True*, attempt OCR via pytesseract (default).
                   If pytesseract is unavailable the step is silently skipped.

    Returns:
        A :class:`LoaderResult` containing one :class:`Document` and one
        visual-asset dict.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Open and normalise to RGB so downstream code always gets 3 channels.
    image: Image.Image = Image.open(str(path)).convert("RGB")

    filename = path.name
    file_type = path.suffix.lstrip(".").lower() or "unknown"
    image_size = image.size  # (width, height)
    doc_id = _generate_doc_id()

    # --- OCR -----------------------------------------------------------------
    ocr_text = ""
    if do_ocr:
        ocr_text = _ocr_image(image)

    has_ocr_text = bool(ocr_text)
    text = ocr_text if has_ocr_text else f"[Image: {filename}]"

    # --- Document ------------------------------------------------------------
    doc = Document(
        doc_id=doc_id,
        text=text,
        metadata={
            "modality": "image",
            "source": filename,
            "file_type": file_type,
            "image_size": image_size,
            "has_ocr_text": has_ocr_text,
        },
        source=str(path.resolve()),
    )

    # --- Visual asset --------------------------------------------------------
    visual_asset: Dict[str, Any] = {
        "image": image,
        "doc_id": doc_id,
        "modality": "image",
        "source": filename,
        "original_path": str(path.resolve()),
    }

    logger.info("Loaded image: %s (size=%s, ocr=%s)", filename, image_size, has_ocr_text)

    return LoaderResult(
        text_documents=[doc],
        visual_assets=[visual_asset],
    )
