"""Text file loader with format-aware section splitting for the RAG pipeline.

Supports plain text, structured log files, and email (mbox-style) files.
Each detected section becomes its own Document with rich metadata.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pdf_loader import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches common log timestamps, e.g.:
#   2024-01-12 08:00:00  |  2024-01-12T08:00:00  |  [2024-01-12 08:00]
_LOG_LINE_RE = re.compile(
    r"^[\[{(]?"
    r"(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[T ]?\d{2}:\d{2}(?::\d{2})?)"
    r"[\]}).]?\s*"
    r"(?:[|\-]\s*)?"
    r"(?:(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|NOTICE)\s*[|\-:]?\s*)?"
    r"(?:\[?(?P<entry_id>[A-Z]{2,5}-\d{3,})\]?\s*[|\-:]?\s*)?"
    r"(?P<message>.*)",
    re.IGNORECASE,
)

# Email separator: a line starting with "From " or a common mbox boundary.
_EMAIL_BOUNDARY_RE = re.compile(r"^From\s+\S+|^-{3,}\s*$", re.MULTILINE)

# Email header fields.
_EMAIL_HEADER_RE = re.compile(
    r"^(?P<key>From|To|Date|Subject|Cc|Bcc):\s*(?P<value>.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def _generate_doc_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(content: str) -> str:
    """Heuristically detect whether the content is a log, email, or plain text.

    Returns one of: ``"log"``, ``"email"``, ``"plain"``.
    """
    lines = content.splitlines()[:30]  # Sample the first 30 lines.

    log_hits = sum(1 for line in lines if _LOG_LINE_RE.match(line.strip()))
    if log_hits >= 3 or (lines and log_hits / max(len(lines), 1) > 0.4):
        return "log"

    email_headers = {"from", "to", "subject", "date"}
    header_keys = {
        m.group("key").lower()
        for m in _EMAIL_HEADER_RE.finditer("\n".join(lines))
    }
    if len(header_keys & email_headers) >= 2:
        return "email"

    return "plain"


# ---------------------------------------------------------------------------
# Section splitters
# ---------------------------------------------------------------------------

def _split_plain(content: str) -> List[str]:
    """Split plain text on double-newline paragraph boundaries."""
    sections = re.split(r"\n{2,}", content)
    return [s.strip() for s in sections if s.strip()]


def _split_log(content: str) -> List[Dict]:
    """Split a log file into individual entries with parsed fields.

    Returns a list of dicts with keys: ``text``, ``timestamp``, ``level``,
    ``entry_id``, ``message``.
    """
    entries: List[Dict] = []
    current_entry: Optional[Dict] = None

    for line in content.splitlines():
        m = _LOG_LINE_RE.match(line.strip())
        if m:
            if current_entry is not None:
                entries.append(current_entry)
            current_entry = {
                "text": line,
                "timestamp": m.group("timestamp"),
                "level": (m.group("level") or "").upper() or None,
                "entry_id": m.group("entry_id"),
                "message": m.group("message").strip(),
            }
        elif current_entry is not None:
            # Continuation line for the current entry.
            current_entry["text"] += "\n" + line
            current_entry["message"] += " " + line.strip()
        else:
            # Leading text before any timestamped line.
            if line.strip():
                current_entry = {
                    "text": line,
                    "timestamp": None,
                    "level": None,
                    "entry_id": None,
                    "message": line.strip(),
                }

    if current_entry is not None:
        entries.append(current_entry)

    return entries


def _split_email(content: str) -> List[Dict]:
    """Split an mbox-style or header-delimited email file into messages.

    Returns a list of dicts with keys: ``text``, ``from``, ``to``, ``date``,
    ``subject``.
    """
    # Try splitting on "From " lines (mbox format).
    parts = re.split(r"(?=^From\s+\S+)", content, flags=re.MULTILINE)
    if len(parts) < 2:
        # Fallback: split on dashed separators.
        parts = re.split(r"^-{3,}\s*$", content, flags=re.MULTILINE)

    emails: List[Dict] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        headers: Dict[str, str] = {}
        for m in _EMAIL_HEADER_RE.finditer(part):
            headers[m.group("key").lower()] = m.group("value").strip()

        emails.append(
            {
                "text": part,
                "from": headers.get("from"),
                "to": headers.get("to"),
                "date": headers.get("date"),
                "subject": headers.get("subject"),
            }
        )

    return emails


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_txt(file_path: str, *, encoding: str = "utf-8") -> List[Document]:
    """Load a text file and split it into section-level Document objects.

    The loader auto-detects the file format (log, email, or plain text) and
    applies the appropriate splitting strategy.

    Args:
        file_path: Path to the text file.
        encoding: File encoding.

    Returns:
        A list of Document objects.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source_name = path.name
    date_extracted = datetime.now(timezone.utc).isoformat()

    try:
        content = path.read_text(encoding=encoding, errors="replace")
    except Exception as exc:
        logger.error("Failed to read %s: %s", source_name, exc)
        raise

    if not content.strip():
        logger.warning("File %s is empty.", source_name)
        return []

    detected_format = _detect_format(content)
    logger.info("Detected format for %s: %s", source_name, detected_format)

    if detected_format == "log":
        return _build_log_documents(content, source_name, path, date_extracted)
    elif detected_format == "email":
        return _build_email_documents(content, source_name, path, date_extracted)
    else:
        return _build_plain_documents(content, source_name, path, date_extracted)


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def _build_plain_documents(
    content: str, source_name: str, path: Path, date_extracted: str
) -> List[Document]:
    sections = _split_plain(content)
    documents: List[Document] = []

    for idx, section in enumerate(sections):
        doc = Document(
            doc_id=_generate_doc_id(),
            text=section,
            metadata={
                "source": source_name,
                "section_index": idx,
                "total_sections": len(sections),
                "file_type": "txt",
                "detected_format": "plain",
                "date_extracted": date_extracted,
            },
            source=str(path.resolve()),
        )
        documents.append(doc)

    logger.info("Loaded %d section(s) from plain text: %s", len(documents), source_name)
    return documents


def _build_log_documents(
    content: str, source_name: str, path: Path, date_extracted: str
) -> List[Document]:
    entries = _split_log(content)
    documents: List[Document] = []

    for idx, entry in enumerate(entries):
        metadata = {
            "source": source_name,
            "section_index": idx,
            "total_sections": len(entries),
            "file_type": "txt",
            "detected_format": "log",
            "date_extracted": date_extracted,
        }
        if entry["timestamp"]:
            metadata["timestamp"] = entry["timestamp"]
        if entry["level"]:
            metadata["log_level"] = entry["level"]
        if entry["entry_id"]:
            metadata["entry_id"] = entry["entry_id"]

        doc = Document(
            doc_id=_generate_doc_id(),
            text=entry["text"],
            metadata=metadata,
            source=str(path.resolve()),
        )
        documents.append(doc)

    logger.info("Loaded %d log entry(ies) from: %s", len(documents), source_name)
    return documents


def _build_email_documents(
    content: str, source_name: str, path: Path, date_extracted: str
) -> List[Document]:
    emails = _split_email(content)
    documents: List[Document] = []

    for idx, email in enumerate(emails):
        metadata = {
            "source": source_name,
            "section_index": idx,
            "total_sections": len(emails),
            "file_type": "txt",
            "detected_format": "email",
            "date_extracted": date_extracted,
        }
        if email["from"]:
            metadata["email_from"] = email["from"]
        if email["to"]:
            metadata["email_to"] = email["to"]
        if email["date"]:
            metadata["email_date"] = email["date"]
        if email["subject"]:
            metadata["email_subject"] = email["subject"]

        doc = Document(
            doc_id=_generate_doc_id(),
            text=email["text"],
            metadata=metadata,
            source=str(path.resolve()),
        )
        documents.append(doc)

    logger.info("Loaded %d email(s) from: %s", len(documents), source_name)
    return documents
