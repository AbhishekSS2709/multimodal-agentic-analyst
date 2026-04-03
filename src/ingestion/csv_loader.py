"""CSV document loader for the RAG ingestion pipeline."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from .pdf_loader import Document

logger = logging.getLogger(__name__)


def _generate_doc_id() -> str:
    return uuid.uuid4().hex[:12]


def load_csv(
    file_path: str,
    *,
    mode: Literal["row", "column"] = "row",
    encoding: str = "utf-8",
    text_template: Optional[str] = None,
) -> List[Document]:
    """Load a CSV file and convert its contents to Document objects.

    Args:
        file_path: Path to the CSV file.
        mode: ``"row"`` creates one Document per row (default);
              ``"column"`` creates one Document per column.
        encoding: File encoding.
        text_template: An optional f-string style template for row mode.
            Column names wrapped in curly braces will be substituted with
            the row value, e.g. ``"Order {order_id}: {description}"``.
            When *None*, every field is rendered as ``column: value``.

    Returns:
        A list of Document objects.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for CSV loading. Install it with: pip install pandas"
        ) from exc

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    date_extracted = datetime.now(timezone.utc).isoformat()
    source_name = path.name

    df = pd.read_csv(str(path), encoding=encoding)
    # Replace NaN with empty string for clean text representations.
    df = df.fillna("")

    column_names = list(df.columns)

    if mode == "row":
        return _load_rows(df, source_name, path, column_names, date_extracted, text_template)
    elif mode == "column":
        return _load_columns(df, source_name, path, column_names, date_extracted)
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'row' or 'column'.")


def _row_to_text(row: "pd.Series", columns: List[str], template: Optional[str]) -> str:
    """Convert a single DataFrame row to a text string."""
    if template is not None:
        try:
            return template.format(**row.to_dict())
        except KeyError:
            # Fall through to default representation if template has bad keys.
            pass

    parts: List[str] = []
    for col in columns:
        value = row[col]
        # Skip empty values to keep text concise.
        if value == "":
            continue
        parts.append(f"{col}: {value}")
    return "\n".join(parts)


def _load_rows(
    df: "pd.DataFrame",
    source_name: str,
    path: Path,
    column_names: List[str],
    date_extracted: str,
    text_template: Optional[str],
) -> List[Document]:
    """Create one Document per row."""
    documents: List[Document] = []

    for idx, row in df.iterrows():
        text = _row_to_text(row, column_names, text_template)
        if not text.strip():
            logger.debug("Row %d in %s produced empty text, skipping.", idx, source_name)
            continue

        doc = Document(
            doc_id=_generate_doc_id(),
            text=text,
            metadata={
                "source": source_name,
                "row_index": int(idx),
                "total_rows": len(df),
                "column_names": column_names,
                "file_type": "csv",
                "mode": "row",
                "date_extracted": date_extracted,
            },
            source=str(path.resolve()),
        )
        documents.append(doc)

    logger.info(
        "Loaded %d row-level document(s) from CSV: %s", len(documents), source_name
    )
    return documents


def _load_columns(
    df: "pd.DataFrame",
    source_name: str,
    path: Path,
    column_names: List[str],
    date_extracted: str,
) -> List[Document]:
    """Create one Document per column, containing all values for that column."""
    documents: List[Document] = []

    for col in column_names:
        values = df[col].astype(str).tolist()
        text = f"Column: {col}\n" + "\n".join(
            f"  [{i}] {v}" for i, v in enumerate(values) if v.strip()
        )

        doc = Document(
            doc_id=_generate_doc_id(),
            text=text,
            metadata={
                "source": source_name,
                "column_name": col,
                "total_columns": len(column_names),
                "row_count": len(df),
                "file_type": "csv",
                "mode": "column",
                "date_extracted": date_extracted,
            },
            source=str(path.resolve()),
        )
        documents.append(doc)

    logger.info(
        "Loaded %d column-level document(s) from CSV: %s", len(documents), source_name
    )
    return documents
