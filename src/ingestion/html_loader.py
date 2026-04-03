"""HTML document loader for the RAG ingestion pipeline."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)

_REMOVE_TAGS = {"script", "style", "nav", "footer", "header"}


def _generate_doc_id() -> str:
    return uuid.uuid4().hex[:12]


def load_html(file_path: str) -> List[Document]:
    """Load an HTML file and return a list of Document objects.

    Strips script, style, nav, footer, and header tags before extracting
    visible text.  Metadata includes source filename, file_type='html', and
    the page <title> if present.

    Args:
        file_path: Absolute or relative path to the HTML file.

    Returns:
        A list containing a single Document with the cleaned page text.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    # Extract title before stripping tags
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Remove unwanted tags
    for tag_name in _REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Get text and clean up whitespace
    raw_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines()]
    cleaned_text = "\n".join(line for line in lines if line)

    if not cleaned_text:
        logger.warning("No visible text extracted from %s", path.name)
        return []

    doc = Document(
        doc_id=_generate_doc_id(),
        text=cleaned_text,
        metadata={
            "source": path.name,
            "file_type": "html",
            "title": title,
        },
        source=str(path.resolve()),
    )
    logger.info("Loaded HTML file: %s", path.name)
    return [doc]
