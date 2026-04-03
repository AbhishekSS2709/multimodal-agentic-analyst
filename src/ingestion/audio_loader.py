"""Audio document loader for the RAG ingestion pipeline."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_doc_id() -> str:
    """Generate a short unique document identifier."""
    return uuid.uuid4().hex[:12]


def _transcribe_whisper(file_path: str) -> str:
    """Transcribe *file_path* using faster-whisper.

    Returns the transcript string, or an empty string on ImportError or any
    other failure.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(file_path)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except ImportError:
        logger.debug("faster_whisper is not installed; skipping Whisper transcription.")
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper transcription failed for %s: %s", file_path, exc)
        return ""


def _transcribe_gemini(file_path: str) -> str:
    """Transcribe *file_path* using the Gemini client.

    Returns the transcript string, or an empty string on any failure.
    """
    try:
        import google.generativeai as genai  # type: ignore

        model = genai.GenerativeModel("gemini-1.5-flash")
        audio_file = genai.upload_file(file_path)
        response = model.generate_content(
            ["Transcribe the audio.", audio_file]
        )
        return response.text.strip() if response.text else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini transcription failed for %s: %s", file_path, exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audio(
    file_path: str,
    *,
    provider: str = "whisper",
) -> List[Document]:
    """Load an audio file, transcribe it, and return a list of Documents.

    Args:
        file_path: Absolute or relative path to the audio file.
        provider:  Transcription backend – ``"whisper"`` (default) or
                   ``"gemini"``.

    Returns:
        A list containing a single :class:`Document` whose text is the
        transcript (or a placeholder when transcription is unavailable).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source_name = path.name
    file_type = path.suffix.lstrip(".").lower() or "unknown"

    # --- Transcribe ----------------------------------------------------------
    if provider == "gemini":
        transcript = _transcribe_gemini(str(path))
    else:
        transcript = _transcribe_whisper(str(path))

    if not transcript:
        transcript = f"[Audio: {source_name}]"
        logger.info("No transcript produced for %s; using placeholder.", source_name)
    else:
        logger.info("Transcribed audio: %s (%d chars)", source_name, len(transcript))

    doc = Document(
        doc_id=_generate_doc_id(),
        text=transcript,
        metadata={
            "modality": "audio",
            "source": source_name,
            "file_type": file_type,
            "provider": provider,
        },
        source=str(path.resolve()),
    )

    return [doc]
