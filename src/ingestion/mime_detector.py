"""MIME-type detection gateway for the multimodal ingestion pipeline.

Determines the type of an incoming file and routes it to the correct loader.
Content-based detection via ``python-magic`` is used when available; the
module gracefully degrades to extension-only detection when libmagic is not
installed on the host system.

Typical usage::

    from src.ingestion.mime_detector import detect_file_type, get_loader_for_mime

    info = detect_file_type("/uploads/report.pdf")
    # {'mime_type': 'application/pdf', 'extension': '.pdf',
    #  'category': 'document', 'loader_name': 'pdf_loader', 'blocked': False}
"""

from __future__ import annotations

import mimetypes
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Optional python-magic import (content-based MIME sniffing)
# ---------------------------------------------------------------------------
try:
    import magic as _magic  # type: ignore[import]
    _MAGIC_AVAILABLE = True
except ImportError:
    _MAGIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Extension → (category, loader_name)
# ---------------------------------------------------------------------------
_EXTENSION_MAP: dict[str, tuple[str, str]] = {
    # ---- text ----
    ".txt":  ("text",       "txt_loader"),
    ".md":   ("text",       "txt_loader"),
    ".rst":  ("text",       "txt_loader"),
    ".log":  ("text",       "txt_loader"),
    # ---- document ----
    ".pdf":  ("document",   "pdf_loader"),
    ".docx": ("document",   "docx_loader"),
    ".doc":  ("document",   "docx_loader"),
    ".pptx": ("document",   "pptx_loader"),
    ".ppt":  ("document",   "pptx_loader"),
    # ---- table ----
    ".csv":  ("table",      "csv_loader"),
    ".tsv":  ("table",      "csv_loader"),
    ".xlsx": ("table",      "excel_loader"),
    ".xls":  ("table",      "excel_loader"),
    ".ods":  ("table",      "excel_loader"),
    # ---- image ----
    ".png":  ("image",      "image_loader"),
    ".jpg":  ("image",      "image_loader"),
    ".jpeg": ("image",      "image_loader"),
    ".gif":  ("image",      "image_loader"),
    ".bmp":  ("image",      "image_loader"),
    ".tiff": ("image",      "image_loader"),
    ".tif":  ("image",      "image_loader"),
    ".webp": ("image",      "image_loader"),
    ".svg":  ("image",      "image_loader"),
    # ---- video ----
    ".mp4":  ("video",      "video_loader"),
    ".avi":  ("video",      "video_loader"),
    ".mov":  ("video",      "video_loader"),
    ".mkv":  ("video",      "video_loader"),
    ".wmv":  ("video",      "video_loader"),
    ".flv":  ("video",      "video_loader"),
    ".webm": ("video",      "video_loader"),
    # ---- audio ----
    ".mp3":  ("audio",      "audio_loader"),
    ".wav":  ("audio",      "audio_loader"),
    ".flac": ("audio",      "audio_loader"),
    ".ogg":  ("audio",      "audio_loader"),
    ".m4a":  ("audio",      "audio_loader"),
    ".aac":  ("audio",      "audio_loader"),
    # ---- web ----
    ".html": ("web",        "html_loader"),
    ".htm":  ("web",        "html_loader"),
    ".xhtml":("web",        "html_loader"),
    # ---- code ----
    ".py":   ("code",       "code_loader"),
    ".js":   ("code",       "code_loader"),
    ".ts":   ("code",       "code_loader"),
    ".java": ("code",       "code_loader"),
    ".cpp":  ("code",       "code_loader"),
    ".c":    ("code",       "code_loader"),
    ".cs":   ("code",       "code_loader"),
    ".go":   ("code",       "code_loader"),
    ".rb":   ("code",       "code_loader"),
    ".php":  ("code",       "code_loader"),
    ".sh":   ("code",       "code_loader"),
    ".sql":  ("code",       "code_loader"),
    # ---- structured ----
    ".json": ("structured", "json_yaml_loader"),
    ".yaml": ("structured", "json_yaml_loader"),
    ".yml":  ("structured", "json_yaml_loader"),
    ".toml": ("structured", "json_yaml_loader"),
    ".xml":  ("structured", "json_yaml_loader"),
    ".ini":  ("structured", "json_yaml_loader"),
    ".cfg":  ("structured", "json_yaml_loader"),
}

# ---------------------------------------------------------------------------
# MIME type → (category, loader_name)
# ---------------------------------------------------------------------------
_MIME_MAP: dict[str, tuple[str, str]] = {
    # ---- text ----
    "text/plain":                                           ("text",       "txt_loader"),
    "text/markdown":                                        ("text",       "txt_loader"),
    "text/x-rst":                                          ("text",       "txt_loader"),
    # ---- document ----
    "application/pdf":                                      ("document",   "pdf_loader"),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                                            ("document",   "docx_loader"),
    "application/msword":                                   ("document",   "docx_loader"),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                                                            ("document",   "pptx_loader"),
    "application/vnd.ms-powerpoint":                        ("document",   "pptx_loader"),
    # ---- table ----
    "text/csv":                                             ("table",      "csv_loader"),
    "text/tab-separated-values":                            ("table",      "csv_loader"),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                                                            ("table",      "excel_loader"),
    "application/vnd.ms-excel":                             ("table",      "excel_loader"),
    "application/vnd.oasis.opendocument.spreadsheet":       ("table",      "excel_loader"),
    # ---- image ----
    "image/png":                                            ("image",      "image_loader"),
    "image/jpeg":                                           ("image",      "image_loader"),
    "image/gif":                                            ("image",      "image_loader"),
    "image/bmp":                                            ("image",      "image_loader"),
    "image/tiff":                                           ("image",      "image_loader"),
    "image/webp":                                           ("image",      "image_loader"),
    "image/svg+xml":                                        ("image",      "image_loader"),
    # ---- video ----
    "video/mp4":                                            ("video",      "video_loader"),
    "video/x-msvideo":                                      ("video",      "video_loader"),
    "video/quicktime":                                      ("video",      "video_loader"),
    "video/x-matroska":                                     ("video",      "video_loader"),
    "video/webm":                                           ("video",      "video_loader"),
    # ---- audio ----
    "audio/mpeg":                                           ("audio",      "audio_loader"),
    "audio/wav":                                            ("audio",      "audio_loader"),
    "audio/x-wav":                                         ("audio",      "audio_loader"),
    "audio/flac":                                           ("audio",      "audio_loader"),
    "audio/ogg":                                            ("audio",      "audio_loader"),
    "audio/mp4":                                            ("audio",      "audio_loader"),
    "audio/aac":                                            ("audio",      "audio_loader"),
    # ---- web ----
    "text/html":                                            ("web",        "html_loader"),
    "application/xhtml+xml":                                ("web",        "html_loader"),
    # ---- code ----
    "text/x-python":                                        ("code",       "code_loader"),
    "application/javascript":                               ("code",       "code_loader"),
    "text/javascript":                                      ("code",       "code_loader"),
    "text/x-java-source":                                   ("code",       "code_loader"),
    "text/x-csrc":                                          ("code",       "code_loader"),
    "text/x-c++src":                                        ("code",       "code_loader"),
    "text/x-shellscript":                                   ("code",       "code_loader"),
    "application/x-sh":                                     ("code",       "code_loader"),
    "application/x-sql":                                    ("code",       "code_loader"),
    # ---- structured ----
    "application/json":                                     ("structured", "json_yaml_loader"),
    "application/x-yaml":                                   ("structured", "json_yaml_loader"),
    "text/yaml":                                            ("structured", "json_yaml_loader"),
    "application/xml":                                      ("structured", "json_yaml_loader"),
    "text/xml":                                             ("structured", "json_yaml_loader"),
    "application/toml":                                     ("structured", "json_yaml_loader"),
}

# ---------------------------------------------------------------------------
# Blocked extensions — executables / system binaries
# ---------------------------------------------------------------------------
_BLOCKED_EXTENSIONS: set[str] = {
    ".exe", ".dll", ".so", ".bat", ".cmd", ".com", ".msi", ".scr",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_file_type(file_path: str) -> dict:
    """Inspect *file_path* and return a classification dict.

    The returned dict always contains:

    * ``mime_type``   – MIME type string (from libmagic or stdlib guess)
    * ``extension``   – lower-cased file extension including the leading dot
    * ``category``    – broad category string (text, document, image, …)
    * ``loader_name`` – name of the loader to use, or ``None`` if unknown
    * ``blocked``     – ``True`` when the file must not be ingested

    Parameters
    ----------
    file_path:
        Absolute or relative path to the file on disk.
    """
    ext = os.path.splitext(file_path)[1].lower()
    blocked = ext in _BLOCKED_EXTENSIONS

    # --- MIME detection ---
    mime_type: Optional[str] = None

    if _MAGIC_AVAILABLE:
        try:
            mime_type = _magic.from_file(file_path, mime=True)
        except Exception:
            pass  # fall through to stdlib

    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(file_path)

    # --- Category / loader resolution ---
    category: Optional[str] = None
    loader_name: Optional[str] = None

    # When python-magic performed content-based detection the MIME type is
    # trustworthy, so consult _MIME_MAP first.  When we fell back to stdlib
    # mimetypes (registry-based, platform-dependent), the file extension is
    # the more reliable signal, so prefer _EXTENSION_MAP in that case.

    if _MAGIC_AVAILABLE and mime_type and mime_type in _MIME_MAP:
        # 1a. Content-sniffed MIME → MIME map
        category, loader_name = _MIME_MAP[mime_type]
    elif ext in _EXTENSION_MAP:
        # 1b. Extension map (reliable, platform-independent)
        category, loader_name = _EXTENSION_MAP[ext]
    elif mime_type and mime_type in _MIME_MAP:
        # 1c. Stdlib MIME guess → MIME map (last resort)
        category, loader_name = _MIME_MAP[mime_type]

    return {
        "mime_type":   mime_type,
        "extension":   ext,
        "category":    category,
        "loader_name": loader_name,
        "blocked":     blocked,
    }


def get_loader_for_mime(mime_type: str, extension: str) -> Optional[str]:
    """Return the loader name for the given *mime_type* and *extension*.

    Lookup order:
    1. ``_MIME_MAP`` keyed on *mime_type*.
    2. ``_EXTENSION_MAP`` keyed on the lower-cased *extension*.

    Returns ``None`` when neither lookup succeeds.

    Parameters
    ----------
    mime_type:
        MIME type string, e.g. ``"application/pdf"``.
    extension:
        File extension including the leading dot, e.g. ``".pdf"``.
        Case is ignored.
    """
    ext = extension.lower()

    # 1. MIME map
    if mime_type in _MIME_MAP:
        return _MIME_MAP[mime_type][1]

    # 2. Extension map
    if ext in _EXTENSION_MAP:
        return _EXTENSION_MAP[ext][1]

    return None
