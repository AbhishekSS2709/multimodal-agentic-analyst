"""Automatic metadata tagging for ingested documents.

Enriches Document objects with semantic tags: document type, content category,
extracted dates, detected entities, and priority level.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from .pdf_loader import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword / pattern banks for classification
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "log": [
        "timestamp", "log level", "ERROR", "WARNING", "INFO", "DEBUG",
        "CRITICAL", "traceback", "stack trace", "exception",
    ],
    "email": [
        "From:", "To:", "Subject:", "Date:", "Dear ", "Sincerely",
        "Best regards", "Sent from", "Reply-To",
    ],
    "contract": [
        "agreement", "herein", "whereas", "party", "obligations",
        "termination", "indemnif", "liability", "governing law",
        "executed", "witness", "clause", "amendment",
    ],
    "order": [
        "order id", "order number", "quantity", "unit price", "total",
        "shipping", "invoice", "SKU", "purchase order", "line item",
        "dispatch", "delivery", "shipment", "tracking",
    ],
    "report": [
        "summary", "analysis", "findings", "recommendation", "conclusion",
        "executive summary", "overview", "quarterly", "annual", "metrics",
        "KPI", "performance",
    ],
}

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "logistics": [
        "shipment", "dispatch", "warehouse", "delivery", "freight",
        "carrier", "route", "tracking", "pallet", "dock",
        "transit", "ETA", "load", "unload",
    ],
    "finance": [
        "invoice", "payment", "revenue", "expense", "budget",
        "profit", "margin", "cost", "balance", "ledger",
        "accounts", "fiscal", "tax",
    ],
    "hr": [
        "employee", "onboarding", "payroll", "benefits", "leave",
        "performance review", "hiring", "termination", "PTO",
    ],
    "legal": [
        "contract", "agreement", "clause", "liability", "compliance",
        "regulation", "statute", "litigation", "indemnif",
    ],
    "operations": [
        "maintenance", "production", "schedule", "inventory", "quality",
        "downtime", "capacity", "throughput",
    ],
    "customer_service": [
        "complaint", "ticket", "resolution", "SLA", "escalat",
        "customer", "support", "feedback", "satisfaction",
    ],
}

# Priority signals.
_HIGH_PRIORITY_PATTERNS: List[str] = [
    r"\bURGENT\b", r"\bCRITICAL\b", r"\bASAP\b", r"\bIMMEDIATE\b",
    r"\bESCALAT", r"\bSEVERE\b", r"\bBLOCK(?:ED|ING|ER)\b",
    r"\bOUTAGE\b", r"\bDOWNTIME\b", r"\bERROR\b",
]

_LOW_PRIORITY_PATTERNS: List[str] = [
    r"\bFYI\b", r"\binformational\b", r"\broutine\b", r"\bminor\b",
    r"\bno.action.required\b",
]

# Date extraction pattern (ISO-like and common US/EU formats).
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})"    # 2024-01-12
    r"|(\d{1,2}[-/]\d{1,2}[-/]\d{4})"      # 01/12/2024
    r"|(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})"
    r"|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)

# Entity extraction: capitalised multi-word names and common identifiers.
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"  # Title Case names
    r"|\b(?:[A-Z][a-z]+-[A-Z0-9]+)\b"           # Codes like "Chicago-W2"
    r"|\b(?:[A-Z]{2,}[-_]\d{2,})\b"             # IDs like "PO-4521"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag_document(doc: Document) -> Document:
    """Enrich a Document's metadata with auto-detected tags.

    Modifies the document *in place* and also returns it for convenience.
    Added metadata keys:

    - ``type``: detected document type (log, email, contract, order, report, general).
    - ``category``: content category (logistics, finance, hr, ...).
    - ``priority``: ``high``, ``medium``, or ``low``.
    - ``dates``: list of date strings found in the text.
    - ``entities``: list of extracted entity names.

    Args:
        doc: The Document to tag.

    Returns:
        The same Document instance with enriched metadata.
    """
    text = doc.text
    text_lower = text.lower()

    doc.metadata["type"] = _detect_type(text, text_lower, doc.metadata)
    doc.metadata["category"] = _detect_category(text_lower)
    doc.metadata["priority"] = _detect_priority(text)
    doc.metadata["dates"] = _extract_dates(text)
    doc.metadata["entities"] = _extract_entities(text)

    return doc


def tag_documents(docs: List[Document]) -> List[Document]:
    """Tag a batch of documents. Convenience wrapper around :func:`tag_document`."""
    for doc in docs:
        tag_document(doc)
    return docs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_keywords(text_lower: str, keywords: List[str]) -> int:
    """Count how many keywords from the list appear in the lowered text."""
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _detect_type(text: str, text_lower: str, metadata: dict) -> str:
    """Determine the document type."""
    # If the loader already identified the format, prefer that.
    detected_format = metadata.get("detected_format")
    if detected_format in ("log", "email"):
        return detected_format

    best_type = "general"
    best_score = 0

    for doc_type, keywords in _TYPE_KEYWORDS.items():
        score = _score_keywords(text_lower, keywords)
        if score > best_score:
            best_score = score
            best_type = doc_type

    # Require a minimum confidence of 2 keyword matches.
    return best_type if best_score >= 2 else "general"


def _detect_category(text_lower: str) -> str:
    """Determine the content category."""
    best_cat = "general"
    best_score = 0

    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = _score_keywords(text_lower, keywords)
        if score > best_score:
            best_score = score
            best_cat = category

    return best_cat if best_score >= 2 else "general"


def _detect_priority(text: str) -> str:
    """Assign a priority level based on signal words."""
    high_hits = sum(
        1 for p in _HIGH_PRIORITY_PATTERNS if re.search(p, text, re.IGNORECASE)
    )
    low_hits = sum(
        1 for p in _LOW_PRIORITY_PATTERNS if re.search(p, text, re.IGNORECASE)
    )

    if high_hits >= 2:
        return "high"
    elif high_hits == 1 and low_hits == 0:
        return "medium"
    elif low_hits >= 1 and high_hits == 0:
        return "low"
    return "medium"


def _extract_dates(text: str) -> List[str]:
    """Pull all date-like strings from the text, deduplicated."""
    matches: List[str] = []
    for m in _DATE_RE.finditer(text):
        # The match object may have several groups; take the first non-None.
        date_str = next((g for g in m.groups() if g is not None), m.group(0))
        date_str = date_str.strip()
        if date_str and date_str not in matches:
            matches.append(date_str)
    return matches


def _extract_entities(text: str) -> List[str]:
    """Extract plausible named entities (capitalised phrases, coded IDs)."""
    # Deduplicate while preserving order.
    seen: Set[str] = set()
    entities: List[str] = []

    # Common stopword phrases to ignore.
    stopwords = {
        "The", "This", "That", "These", "Those", "Monday", "Tuesday",
        "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    }

    for m in _ENTITY_RE.finditer(text):
        entity = m.group(0).strip()
        if entity in stopwords or entity in seen or len(entity) < 3:
            continue
        seen.add(entity)
        entities.append(entity)

    return entities
