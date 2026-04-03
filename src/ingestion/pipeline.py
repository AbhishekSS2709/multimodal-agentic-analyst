"""Unified ingestion pipeline that orchestrates all loaders and tagging.

Usage::

    from src.ingestion.pipeline import IngestPipeline

    pipeline = IngestPipeline()
    docs = pipeline.ingest("data/raw/orders.csv")          # single file
    docs = pipeline.ingest_directory("data/raw/")           # whole directory
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from .audio_loader import load_audio
from .code_loader import load_code
from .csv_loader import load_csv
from .docx_loader import load_docx
from .excel_loader import load_excel
from .html_loader import load_html
from .image_loader import load_image as _load_image_raw
from .json_yaml_loader import load_json_yaml
from .metadata_tagger import tag_documents
from .pdf_loader import Document, load_pdf
from .pptx_loader import load_pptx
from .txt_loader import load_txt
from .video_loader import load_video as _load_video_raw


def _load_image_wrapper(file_path: str, **kwargs) -> List[Document]:
    result = _load_image_raw(file_path, **kwargs)
    return result.text_documents


def _load_video_wrapper(file_path: str, **kwargs) -> List[Document]:
    result = _load_video_raw(file_path, **kwargs)
    return result.text_documents

logger = logging.getLogger(__name__)

# Mapping from file extension (lowercase, with dot) to the loader function.
_LOADER_MAP: Dict[str, callable] = {
    # Existing
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".txt": load_txt,
    ".log": load_txt,
    ".text": load_txt,
    ".md": load_txt,
    ".eml": load_txt,
    # Documents
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".xlsx": load_excel,
    ".xls": load_excel,
    # Web
    ".html": load_html,
    ".htm": load_html,
    # Code
    ".py": load_code,
    ".js": load_code,
    ".ts": load_code,
    ".java": load_code,
    ".cpp": load_code,
    ".c": load_code,
    ".go": load_code,
    ".rs": load_code,
    ".rb": load_code,
    ".php": load_code,
    ".sh": load_code,
    # Structured
    ".json": load_json_yaml,
    ".yaml": load_json_yaml,
    ".yml": load_json_yaml,
    # Audio
    ".mp3": load_audio,
    ".wav": load_audio,
    ".m4a": load_audio,
    # Images
    ".png": _load_image_wrapper,
    ".jpg": _load_image_wrapper,
    ".jpeg": _load_image_wrapper,
    ".webp": _load_image_wrapper,
    ".bmp": _load_image_wrapper,
    ".gif": _load_image_wrapper,
    # Video
    ".mp4": _load_video_wrapper,
    ".avi": _load_video_wrapper,
    ".mov": _load_video_wrapper,
}

# Extensions that are silently skipped when scanning a directory.
_SKIP_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".db", ".sqlite", ".sqlite3",
}


class IngestPipeline:
    """Orchestrates document loading, tagging, and normalisation.

    Args:
        auto_tag: If *True* (default), every loaded document is run through
            the metadata tagger automatically.
        csv_mode: Default CSV loading mode (``"row"`` or ``"column"``).
    """

    def __init__(
        self,
        *,
        auto_tag: bool = True,
        csv_mode: str = "row",
    ) -> None:
        self.auto_tag = auto_tag
        self.csv_mode = csv_mode
        self._stats: Dict[str, int] = {
            "files_processed": 0,
            "files_skipped": 0,
            "documents_created": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        """Return a copy of the processing statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Zero out all counters."""
        for key in self._stats:
            self._stats[key] = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str, **loader_kwargs) -> List[Document]:
        """Ingest a single file and return its Document objects.

        The appropriate loader is selected based on file extension. Extra
        keyword arguments are forwarded to the loader (e.g. ``mode`` for CSV).

        Args:
            file_path: Path to the file.
            **loader_kwargs: Additional arguments passed to the loader.

        Returns:
            A list of Document objects with consistent schema.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error("File not found: %s", file_path)
            self._stats["errors"] += 1
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        loader = _LOADER_MAP.get(ext)

        if loader is None:
            logger.warning(
                "No loader registered for extension %r — skipping %s", ext, path.name
            )
            self._stats["files_skipped"] += 1
            return []

        try:
            # Inject csv_mode default for CSV files if not explicitly given.
            if ext == ".csv" and "mode" not in loader_kwargs:
                loader_kwargs["mode"] = self.csv_mode

            documents = loader(str(path), **loader_kwargs)
        except Exception as exc:
            logger.error("Error loading %s: %s", path.name, exc, exc_info=True)
            self._stats["errors"] += 1
            raise

        # Ensure consistent metadata on every document.
        for doc in documents:
            doc.metadata.setdefault("source", path.name)
            doc.metadata.setdefault("file_type", ext.lstrip("."))

        if self.auto_tag:
            tag_documents(documents)

        self._stats["files_processed"] += 1
        self._stats["documents_created"] += len(documents)

        logger.info(
            "Ingested %s -> %d document(s)", path.name, len(documents)
        )
        return documents

    def ingest_directory(
        self,
        dir_path: str,
        *,
        recursive: bool = True,
        extensions: Optional[Set[str]] = None,
    ) -> List[Document]:
        """Batch-ingest all supported files in a directory.

        Args:
            dir_path: Path to the directory.
            recursive: Walk subdirectories when *True* (default).
            extensions: If provided, only process files with these extensions
                (including the dot, e.g. ``{".pdf", ".csv"}``). When *None*,
                all registered extensions are processed.

        Returns:
            A flat list of all Document objects created.
        """
        root = Path(dir_path)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        allowed = extensions or set(_LOADER_MAP.keys())
        all_documents: List[Document] = []

        file_iter = root.rglob("*") if recursive else root.glob("*")

        for file_path in sorted(file_iter):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()

            if ext in _SKIP_EXTENSIONS:
                continue

            if ext not in allowed:
                logger.debug("Skipping unsupported file: %s", file_path.name)
                self._stats["files_skipped"] += 1
                continue

            try:
                docs = self.ingest(str(file_path))
                all_documents.extend(docs)
            except Exception as exc:
                # Log and continue — don't let one bad file abort the batch.
                logger.error(
                    "Failed to ingest %s: %s", file_path.name, exc
                )
                # Stats already updated inside self.ingest().

        logger.info(
            "Directory ingestion complete: %d file(s) processed, "
            "%d document(s) created, %d error(s).",
            self._stats["files_processed"],
            self._stats["documents_created"],
            self._stats["errors"],
        )
        return all_documents
