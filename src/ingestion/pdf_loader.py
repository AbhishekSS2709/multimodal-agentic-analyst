"""PDF and text document loader for the RAG ingestion pipeline."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a single document chunk with text and metadata."""

    doc_id: str
    text: str
    metadata: dict
    source: str


def _generate_doc_id() -> str:
    """Generate a unique document identifier."""
    return uuid.uuid4().hex[:12]


def load_pdf(file_path: str, *, encoding: str = "utf-8") -> List[Document]:
    """Load a PDF file and return a list of Document objects, one per page.

    Attempts to parse the file with pypdf first. If the file is not a valid
    PDF (or pypdf is unavailable), falls back to reading it as plain text.

    Args:
        file_path: Absolute or relative path to the PDF file.
        encoding: Text encoding used for the plain-text fallback.

    Returns:
        A list of Document objects with page-level granularity.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source_name = path.name
    date_extracted = datetime.now(timezone.utc).isoformat()

    # --- Try pypdf first ---------------------------------------------------
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        documents: List[Document] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                logger.debug("Page %d of %s is empty, skipping.", page_num, source_name)
                continue

            doc = Document(
                doc_id=_generate_doc_id(),
                text=text,
                metadata={
                    "source": source_name,
                    "page_number": page_num,
                    "total_pages": len(reader.pages),
                    "file_type": "pdf",
                    "date_extracted": date_extracted,
                },
                source=str(path.resolve()),
            )
            documents.append(doc)

        if documents:
            logger.info(
                "Loaded %d page(s) from PDF: %s", len(documents), source_name
            )
            return documents

        # If pypdf returned zero pages, fall through to text fallback.
        logger.warning(
            "pypdf extracted no text from %s; falling back to plain-text read.",
            source_name,
        )
    except ImportError:
        logger.warning("pypdf is not installed; falling back to plain-text read.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pypdf failed on %s (%s); falling back to plain-text read.",
            source_name,
            exc,
        )

    # --- Plain-text fallback ------------------------------------------------
    return _load_as_text(path, encoding=encoding, date_extracted=date_extracted)


def _load_as_text(
    path: Path,
    *,
    encoding: str = "utf-8",
    date_extracted: str,
) -> List[Document]:
    """Read a file as plain text and wrap it in a single Document."""
    try:
        content = path.read_text(encoding=encoding, errors="replace").strip()
    except Exception as exc:
        logger.error("Unable to read %s as text: %s", path.name, exc)
        raise

    if not content:
        logger.warning("File %s is empty after reading as text.", path.name)
        return []

    doc = Document(
        doc_id=_generate_doc_id(),
        text=content,
        metadata={
            "source": path.name,
            "page_number": 1,
            "total_pages": 1,
            "file_type": "txt",
            "date_extracted": date_extracted,
            "fallback": True,
        },
        source=str(path.resolve()),
    )
    logger.info("Loaded %s as plain text (fallback).", path.name)
    return [doc]
