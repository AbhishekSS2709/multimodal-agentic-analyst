"""Multi-hop retrieval engine for the RAG pipeline.

Implements iterative retrieval that follows chains of reasoning across
multiple hops, extracting entities and generating follow-up queries to
gather comprehensive context for complex questions.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import TOP_K

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol for the base retriever dependency (duck typing)
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseRetriever(Protocol):
    """Minimal interface that any base retriever must satisfy."""

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Return a list of dicts, each containing at least 'text' and optionally
        'metadata', 'score', 'doc_id', etc."""
        ...


# ---------------------------------------------------------------------------
# Entity extraction (regex + keyword, no ML)
# ---------------------------------------------------------------------------

# Pre-compiled patterns for common enterprise entities
_SUPPLIER_PATTERN = re.compile(
    r"(?:supplier|vendor|manufacturer|provider)\s+([A-Z][A-Za-z0-9&\s\-]{1,40}?)(?:\s+(?:delayed|shipped|supplied|failed|caused|reported|delivered|received|was|has|had|is))",
    re.IGNORECASE,
)
_SUPPLIER_NAME_PATTERN = re.compile(
    r"(?:from|by|with|to)\s+(?:supplier|vendor)\s+([A-Z][A-Za-z0-9&\s\-]{1,40}?)(?:\.|,|\s+(?:and|or|which|who|that))",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
    r"|\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_ORDER_ID_PATTERN = re.compile(
    r"\b((?:ORD|PO|SO|INV|REQ)[-#]?\d{3,10})\b", re.IGNORECASE
)
_MACHINE_ID_PATTERN = re.compile(
    r"\b((?:MCH|MACH|EQP|LINE|UNIT)[-#]?\d{2,10})\b", re.IGNORECASE
)
_LOCATION_PATTERN = re.compile(
    r"(?:in|at|from|to|near|located\s+(?:in|at))\s+([A-Z][A-Za-z\s]{2,30}?)(?:\.|,|\s+(?:and|or|which|where|facility|plant|warehouse|office))",
    re.IGNORECASE,
)
_QUOTED_ENTITY_PATTERN = re.compile(r'"([^"]{2,60})"')

# Key concept words that signal topics worth following up on
_CONCEPT_KEYWORDS = [
    "delay", "failure", "shortage", "defect", "quality", "compliance",
    "shipment", "delivery", "production", "downtime", "maintenance",
    "cancellation", "escalation", "resolution", "inspection", "recall",
    "cost", "overdue", "penalty", "backorder", "outage",
]


@dataclass
class ExtractedEntities:
    """Container for entities extracted from text."""

    suppliers: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    order_ids: List[str] = field(default_factory=list)
    machine_ids: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    quoted_entities: List[str] = field(default_factory=list)

    @property
    def all_entities(self) -> List[str]:
        """Return a flat list of every extracted entity."""
        return (
            self.suppliers
            + self.dates
            + self.order_ids
            + self.machine_ids
            + self.locations
            + self.concepts
            + self.quoted_entities
        )

    @property
    def is_empty(self) -> bool:
        return len(self.all_entities) == 0


def extract_entities(text: str) -> ExtractedEntities:
    """Extract structured entities from *text* using regex and keyword matching.

    Parameters
    ----------
    text : str
        Raw text to scan for entities.

    Returns
    -------
    ExtractedEntities
        Container with categorised entity lists.
    """
    entities = ExtractedEntities()

    if not text or not text.strip():
        return entities

    # Suppliers
    for match in _SUPPLIER_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name and len(name) > 1:
            entities.suppliers.append(name)
    for match in _SUPPLIER_NAME_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name and len(name) > 1 and name not in entities.suppliers:
            entities.suppliers.append(name)

    # Dates
    for match in _DATE_PATTERN.finditer(text):
        date_str = match.group(0).strip()
        if date_str:
            entities.dates.append(date_str)

    # Order IDs
    for match in _ORDER_ID_PATTERN.finditer(text):
        entities.order_ids.append(match.group(1).upper())

    # Machine IDs
    for match in _MACHINE_ID_PATTERN.finditer(text):
        entities.machine_ids.append(match.group(1).upper())

    # Locations
    for match in _LOCATION_PATTERN.finditer(text):
        loc = match.group(1).strip()
        if loc and len(loc) > 2:
            entities.locations.append(loc)

    # Quoted entities
    for match in _QUOTED_ENTITY_PATTERN.finditer(text):
        entities.quoted_entities.append(match.group(1))

    # Concept keywords
    text_lower = text.lower()
    for keyword in _CONCEPT_KEYWORDS:
        if keyword in text_lower:
            entities.concepts.append(keyword)

    # Deduplicate within each category
    entities.suppliers = list(dict.fromkeys(entities.suppliers))
    entities.dates = list(dict.fromkeys(entities.dates))
    entities.order_ids = list(dict.fromkeys(entities.order_ids))
    entities.machine_ids = list(dict.fromkeys(entities.machine_ids))
    entities.locations = list(dict.fromkeys(entities.locations))
    entities.concepts = list(dict.fromkeys(entities.concepts))
    entities.quoted_entities = list(dict.fromkeys(entities.quoted_entities))

    return entities


# ---------------------------------------------------------------------------
# Follow-up query generation
# ---------------------------------------------------------------------------

def _generate_follow_up_queries(
    original_query: str,
    entities: ExtractedEntities,
    previous_queries: List[str],
) -> List[str]:
    """Generate follow-up queries based on extracted entities and information gaps.

    The strategy is to create targeted queries that pursue the most promising
    entities or concepts found in the already-retrieved context.

    Parameters
    ----------
    original_query : str
        The user's original question.
    entities : ExtractedEntities
        Entities extracted from retrieved chunks so far.
    previous_queries : list[str]
        Queries already issued (to avoid repetition).

    Returns
    -------
    list[str]
        Up to 3 follow-up queries.
    """
    follow_ups: List[str] = []
    prev_lower = {q.lower() for q in previous_queries}

    def _add(q: str) -> None:
        if q.lower() not in prev_lower and q not in follow_ups:
            follow_ups.append(q)

    # Pursue supplier-specific queries
    for supplier in entities.suppliers[:2]:
        _add(f"{supplier} delay cause root analysis")
        _add(f"{supplier} shipment delivery performance")

    # Pursue order-specific queries
    for order_id in entities.order_ids[:2]:
        _add(f"{order_id} status details timeline")

    # Pursue machine-specific queries
    for machine_id in entities.machine_ids[:2]:
        _add(f"{machine_id} failure maintenance history")

    # Pursue location-based queries
    for location in entities.locations[:1]:
        _add(f"{location} operations supply chain issues")

    # Pursue concept-based follow-ups by combining with key entities
    if entities.concepts and entities.suppliers:
        for concept in entities.concepts[:2]:
            for supplier in entities.suppliers[:1]:
                _add(f"{supplier} {concept} details impact")

    # If we still have no follow-ups, try broadening with concept keywords
    if not follow_ups and entities.concepts:
        for concept in entities.concepts[:3]:
            _add(f"{concept} details root cause {original_query.split()[0]}")

    return follow_ups[:3]


# ---------------------------------------------------------------------------
# Hop detail record
# ---------------------------------------------------------------------------

@dataclass
class HopDetail:
    """Record of a single retrieval hop."""

    hop_number: int
    query: str
    chunks_retrieved: List[Dict[str, Any]]
    entities_extracted: ExtractedEntities
    confidence: float
    num_new_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON-friendly output."""
        return {
            "hop_number": self.hop_number,
            "query": self.query,
            "num_chunks_retrieved": len(self.chunks_retrieved),
            "num_new_chunks": self.num_new_chunks,
            "entities": {
                "suppliers": self.entities_extracted.suppliers,
                "dates": self.entities_extracted.dates,
                "order_ids": self.entities_extracted.order_ids,
                "machine_ids": self.entities_extracted.machine_ids,
                "locations": self.entities_extracted.locations,
                "concepts": self.entities_extracted.concepts,
            },
            "confidence": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# MultiHopRetriever
# ---------------------------------------------------------------------------

class MultiHopRetriever:
    """Iterative multi-hop retriever that follows entity chains.

    Given a base retriever (any object with a ``retrieve(query, top_k)``
    method), this class performs up to *max_hops* rounds of retrieval,
    extracting entities from each round's results and generating targeted
    follow-up queries to fill information gaps.

    Parameters
    ----------
    base_retriever : BaseRetriever
        Any object satisfying the :class:`BaseRetriever` protocol.
    top_k : int
        Number of chunks to retrieve per hop.
    """

    def __init__(
        self,
        base_retriever: Any,
        top_k: int = TOP_K,
    ) -> None:
        self._base_retriever = base_retriever
        self._top_k = top_k
        logger.info(
            "MultiHopRetriever initialised (top_k=%d).",
            top_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        max_hops: int = 3,
    ) -> Dict[str, Any]:
        """Run multi-hop retrieval for *query*.

        Parameters
        ----------
        query : str
            The user's natural-language question.
        max_hops : int
            Maximum number of retrieval hops (1-based). The first hop uses
            the original query; subsequent hops use generated follow-up
            queries.

        Returns
        -------
        dict
            ``all_chunks``       — deduplicated list of all retrieved chunk dicts.
            ``hop_details``      — list of :class:`HopDetail` dicts (one per hop).
            ``reasoning_chain``  — human-readable list of reasoning steps.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to multi-hop retriever.")
            return {
                "all_chunks": [],
                "hop_details": [],
                "reasoning_chain": ["No query provided."],
            }

        max_hops = max(1, min(max_hops, 10))  # clamp to sensible range

        all_chunks: List[Dict[str, Any]] = []
        seen_texts: set[str] = set()
        hop_details: List[HopDetail] = []
        reasoning_chain: List[str] = []
        all_entities = ExtractedEntities()
        previous_queries: List[str] = [query]

        # ---- Hop 1: initial retrieval on the original query ----
        reasoning_chain.append(f"Hop 1: Retrieving context for original query: '{query}'")
        hop1 = self._execute_hop(
            hop_number=1,
            query=query,
            all_chunks=all_chunks,
            seen_texts=seen_texts,
            all_entities=all_entities,
        )
        hop_details.append(hop1)
        reasoning_chain.append(
            f"Hop 1 result: retrieved {hop1.num_new_chunks} new chunk(s), "
            f"found entities: {hop1.entities_extracted.all_entities[:5]}"
        )

        # ---- Subsequent hops ----
        for hop_num in range(2, max_hops + 1):
            # Generate follow-up queries from accumulated entities
            follow_ups = _generate_follow_up_queries(
                original_query=query,
                entities=all_entities,
                previous_queries=previous_queries,
            )

            if not follow_ups:
                reasoning_chain.append(
                    f"Hop {hop_num}: No follow-up queries generated — "
                    "stopping early (entities exhausted)."
                )
                break

            follow_up_query = follow_ups[0]
            previous_queries.append(follow_up_query)

            reasoning_chain.append(
                f"Hop {hop_num}: Following up with query: '{follow_up_query}'"
            )

            hop_result = self._execute_hop(
                hop_number=hop_num,
                query=follow_up_query,
                all_chunks=all_chunks,
                seen_texts=seen_texts,
                all_entities=all_entities,
            )
            hop_details.append(hop_result)

            reasoning_chain.append(
                f"Hop {hop_num} result: retrieved {hop_result.num_new_chunks} new chunk(s), "
                f"confidence={hop_result.confidence:.2f}"
            )

            # Early termination if no new information is surfacing
            if hop_result.num_new_chunks == 0:
                reasoning_chain.append(
                    f"Hop {hop_num}: No new chunks found — stopping early."
                )
                break

        reasoning_chain.append(
            f"Multi-hop retrieval complete: {len(all_chunks)} total chunk(s) "
            f"across {len(hop_details)} hop(s)."
        )

        return {
            "all_chunks": all_chunks,
            "hop_details": [h.to_dict() for h in hop_details],
            "reasoning_chain": reasoning_chain,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_hop(
        self,
        hop_number: int,
        query: str,
        all_chunks: List[Dict[str, Any]],
        seen_texts: set,
        all_entities: ExtractedEntities,
    ) -> HopDetail:
        """Execute a single retrieval hop and update accumulators in place.

        Parameters
        ----------
        hop_number : int
            1-based hop index.
        query : str
            The query to send to the base retriever.
        all_chunks : list
            Accumulator list — new unique chunks are appended.
        seen_texts : set
            Set of chunk text hashes already seen (for deduplication).
        all_entities : ExtractedEntities
            Accumulator for entities across all hops.

        Returns
        -------
        HopDetail
            Record of this hop's results.
        """
        try:
            raw_results = self._base_retriever.retrieve(query, top_k=self._top_k)
        except Exception:
            logger.exception("Base retriever failed on hop %d for query '%s'.", hop_number, query)
            raw_results = []

        # Normalise results: ensure each result is a dict with at least 'text'
        chunks = self._normalise_results(raw_results)

        # Deduplicate against previously seen chunks
        new_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            text_key = chunk.get("text", "").strip()
            if text_key and text_key not in seen_texts:
                seen_texts.add(text_key)
                new_chunks.append(chunk)
                all_chunks.append(chunk)

        # Extract entities from new chunks
        combined_text = " ".join(c.get("text", "") for c in new_chunks)
        hop_entities = extract_entities(combined_text)

        # Merge into global entity accumulators
        self._merge_entities(all_entities, hop_entities)

        # Compute a simple confidence score for this hop
        confidence = self._compute_hop_confidence(
            query=query,
            chunks=new_chunks,
        )

        return HopDetail(
            hop_number=hop_number,
            query=query,
            chunks_retrieved=chunks,
            entities_extracted=hop_entities,
            confidence=confidence,
            num_new_chunks=len(new_chunks),
        )

    @staticmethod
    def _normalise_results(raw_results: Any) -> List[Dict[str, Any]]:
        """Coerce raw retriever output into a list of chunk dicts."""
        if not raw_results:
            return []

        normalised: List[Dict[str, Any]] = []
        for item in raw_results:
            if isinstance(item, dict):
                normalised.append(item)
            elif isinstance(item, str):
                normalised.append({"text": item})
            elif hasattr(item, "text"):
                # Handle Chunk-like objects
                d: Dict[str, Any] = {"text": item.text}
                if hasattr(item, "metadata"):
                    d["metadata"] = item.metadata
                if hasattr(item, "doc_id"):
                    d["doc_id"] = item.doc_id
                if hasattr(item, "chunk_id"):
                    d["chunk_id"] = item.chunk_id
                if hasattr(item, "score"):
                    d["score"] = item.score
                normalised.append(d)
            else:
                logger.warning("Unrecognised retriever result type: %s", type(item))
        return normalised

    @staticmethod
    def _merge_entities(target: ExtractedEntities, source: ExtractedEntities) -> None:
        """Merge *source* entities into *target*, deduplicating."""
        for attr in (
            "suppliers", "dates", "order_ids", "machine_ids",
            "locations", "concepts", "quoted_entities",
        ):
            existing = set(getattr(target, attr))
            for val in getattr(source, attr):
                if val not in existing:
                    getattr(target, attr).append(val)
                    existing.add(val)

    @staticmethod
    def _compute_hop_confidence(
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> float:
        """Compute a 0-1 confidence score for a hop's results.

        Factors:
        - Number of chunks retrieved (more = higher, up to a point)
        - Average relevance score if available
        - Query-term overlap with chunk text
        """
        if not chunks:
            return 0.0

        # Factor 1: chunk count (diminishing returns)
        count_score = min(len(chunks) / 5.0, 1.0)

        # Factor 2: average retriever score (if available)
        scores = [c.get("score", 0.0) for c in chunks if "score" in c]
        avg_score = sum(scores) / len(scores) if scores else 0.5

        # Factor 3: query-term coverage in retrieved text
        query_terms = set(query.lower().split())
        if query_terms:
            combined_text_lower = " ".join(c.get("text", "") for c in chunks).lower()
            covered = sum(1 for t in query_terms if t in combined_text_lower)
            coverage = covered / len(query_terms)
        else:
            coverage = 0.0

        confidence = 0.3 * count_score + 0.3 * avg_score + 0.4 * coverage
        return min(max(confidence, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    # Minimal stub retriever for demonstration
    class _DemoRetriever:
        """Fake retriever that returns canned results for demo purposes."""

        _CORPUS = [
            {
                "text": "Supplier Acme Corp delayed shipment ORD-12345 by 3 weeks due to part shortage at their Dallas facility.",
                "doc_id": "doc_001",
                "score": 0.92,
                "metadata": {"source": "incident_report_q3.pdf"},
            },
            {
                "text": "Machine MCH-4401 failed on 2025-06-15 because replacement parts from Acme Corp were not delivered on time.",
                "doc_id": "doc_002",
                "score": 0.87,
                "metadata": {"source": "maintenance_log.csv"},
            },
            {
                "text": "Order ORD-12345 was escalated to VP of Operations after production line downtime exceeded 48 hours.",
                "doc_id": "doc_003",
                "score": 0.81,
                "metadata": {"source": "escalation_tracker.pdf"},
            },
            {
                "text": "Acme Corp has a history of delivery delays — 5 incidents in the last quarter, primarily affecting the Dallas plant.",
                "doc_id": "doc_004",
                "score": 0.78,
                "metadata": {"source": "supplier_scorecard.csv"},
            },
            {
                "text": "Vendor BetaTech supplied backup parts on 2025-07-01, resolving the MCH-4401 downtime after 10 days.",
                "doc_id": "doc_005",
                "score": 0.72,
                "metadata": {"source": "resolution_notes.pdf"},
            },
        ]

        def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            """Return results that overlap with query terms."""
            query_lower = query.lower()
            scored = []
            for doc in self._CORPUS:
                text_lower = doc["text"].lower()
                overlap = sum(1 for w in query_lower.split() if w in text_lower)
                if overlap > 0:
                    scored.append((doc, overlap))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in scored[:top_k]]

    retriever = MultiHopRetriever(base_retriever=_DemoRetriever(), top_k=5)
    result = retriever.retrieve("Which supplier caused most delays and why?", max_hops=3)

    print("\n=== Multi-Hop Retrieval Demo ===\n")
    print("Reasoning chain:")
    for step in result["reasoning_chain"]:
        print(f"  - {step}")
    print(f"\nTotal chunks retrieved: {len(result['all_chunks'])}")
    print(f"Hops executed: {len(result['hop_details'])}")
    for hop in result["hop_details"]:
        print(f"\n  Hop {hop['hop_number']}: query='{hop['query']}'")
        print(f"    chunks={hop['num_chunks_retrieved']}, new={hop['num_new_chunks']}, confidence={hop['confidence']}")
        print(f"    entities={hop['entities']}")
