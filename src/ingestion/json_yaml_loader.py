"""JSON and YAML document loader for the RAG ingestion pipeline."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore[import]
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False
    logger.warning("PyYAML is not installed; .yaml/.yml files will be loaded as raw text.")


def _generate_doc_id() -> str:
    return uuid.uuid4().hex[:12]


def _flatten(obj: Any, prefix: str = "") -> List[str]:
    """Recursively flatten a nested object to 'path: value' lines.

    Args:
        obj:    The object to flatten (dict, list, or scalar).
        prefix: Dot-separated key prefix accumulated during recursion.

    Returns:
        A list of strings in the form ``"some.key: value"``.
    """
    lines: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten(value, prefix=child_prefix))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            child_prefix = f"{prefix}[{index}]"
            lines.extend(_flatten(item, prefix=child_prefix))
    else:
        # Scalar — emit a single "path: value" line
        lines.append(f"{prefix}: {obj}")
    return lines


def load_json_yaml(file_path: str) -> List[Document]:
    """Load a JSON or YAML file and return a list of Document objects.

    The nested structure is flattened to human-readable ``path: value`` lines
    so that downstream text embedders can index every key/value pair.

    Args:
        file_path: Absolute or relative path to the .json, .yaml, or .yml file.

    Returns:
        A list containing a single Document with the flattened text.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    file_type = ext.lstrip(".")  # "json", "yaml", or "yml"
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    # --- Parse the file -------------------------------------------------------
    if ext == ".json":
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON file %s: %s", path.name, exc)
            raise
        lines = _flatten(data)

    elif ext in {".yaml", ".yml"}:
        if _YAML_AVAILABLE:
            try:
                data = yaml.safe_load(raw_text)
            except yaml.YAMLError as exc:
                logger.error("Failed to parse YAML file %s: %s", path.name, exc)
                raise
            lines = _flatten(data) if data is not None else []
        else:
            # Fallback: treat as plain text
            lines = raw_text.splitlines()

    else:
        # Unknown extension — best-effort raw text
        logger.warning("Unknown extension '%s'; loading %s as raw text.", ext, path.name)
        lines = raw_text.splitlines()

    text = "\n".join(lines)
    if not text.strip():
        logger.warning("No content extracted from %s", path.name)
        return []

    doc = Document(
        doc_id=_generate_doc_id(),
        text=text,
        metadata={
            "source": path.name,
            "file_type": file_type,
            "modality": "structured",
        },
        source=str(path.resolve()),
    )
    logger.info("Loaded structured file: %s (file_type=%s)", path.name, file_type)
    return [doc]
