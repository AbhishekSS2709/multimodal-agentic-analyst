"""Code file document loader for the RAG ingestion pipeline."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)

# Extension → language name mapping
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".lua": "lua",
    ".pl": "perl",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".clj": "clojure",
    ".dart": "dart",
    ".m": "objective-c",
    ".mm": "objective-cpp",
}


def _generate_doc_id() -> str:
    return uuid.uuid4().hex[:12]


def load_code(file_path: str) -> List[Document]:
    """Load a source-code file and return a list of Document objects.

    Language is detected from the file extension.  Metadata includes the
    source filename, file_type (extension without the leading dot), the
    detected language, modality='code', and the line count.

    Args:
        file_path: Absolute or relative path to the code file.

    Returns:
        A list containing a single Document with the raw source text.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    language = EXTENSION_LANGUAGE_MAP.get(ext, "unknown")
    # file_type without leading dot; fall back to "txt" for unknown extensions
    file_type = ext.lstrip(".") if ext else "txt"

    content = path.read_text(encoding="utf-8", errors="replace")
    line_count = len(content.splitlines())

    if not content.strip():
        logger.warning("Code file %s appears to be empty.", path.name)
        return []

    doc = Document(
        doc_id=_generate_doc_id(),
        text=content,
        metadata={
            "source": path.name,
            "file_type": file_type,
            "language": language,
            "modality": "code",
            "line_count": line_count,
        },
        source=str(path.resolve()),
    )
    logger.info("Loaded code file: %s (language=%s)", path.name, language)
    return [doc]
