"""Source attribution and confidence scoring for RAG answers.

Provides :class:`SourceAttributor` which analyses a generated answer against
the retrieved context to produce:
- A list of source documents with relevance scores and excerpts
- An overall confidence score
- A faithfulness indicator (how well the answer stays within the evidence)
- A coverage metric (how well the query's key terms are addressed)
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import TOP_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop words for keyword extraction (lightweight, no external dependencies)
# ---------------------------------------------------------------------------
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "were", "are", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "not", "no", "nor", "so", "if", "then",
    "than", "too", "very", "just", "about", "above", "after", "again",
    "all", "also", "am", "any", "as", "because", "before", "between",
    "both", "each", "few", "more", "most", "other", "own", "same",
    "some", "such", "through", "up", "down", "out", "off", "over",
    "under", "until", "while", "into", "during", "only",
}

# Sentence boundary regex (simple, same style as the chunker module)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_key_terms(text: str) -> List[str]:
    """Extract meaningful terms from *text*, filtering stop words.

    Returns
    -------
    list[str]
        Lower-cased terms that are not stop words, preserving order.
    """
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in parts if s and s.strip()]


def _compute_text_overlap(text_a: str, text_b: str) -> float:
    """Return the fraction of words in *text_a* that also appear in *text_b*.

    A simple unigram overlap metric, 0-1.
    """
    words_a = set(_extract_key_terms(text_a))
    words_b = set(_extract_key_terms(text_b))
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def _best_excerpt(chunk_text: str, query: str, max_len: int = 200) -> str:
    """Return the most relevant excerpt from *chunk_text* for *query*.

    Selects the sentence with the highest word overlap with the query.
    If the chunk is shorter than *max_len*, returns the full text.
    """
    if len(chunk_text) <= max_len:
        return chunk_text.strip()

    sentences = _split_sentences(chunk_text)
    if not sentences:
        return chunk_text[:max_len].strip()

    query_terms = set(_extract_key_terms(query))
    if not query_terms:
        return sentences[0][:max_len].strip()

    best_sent = sentences[0]
    best_score = -1.0

    for sent in sentences:
        sent_terms = set(_extract_key_terms(sent))
        overlap = len(sent_terms & query_terms)
        if overlap > best_score:
            best_score = overlap
            best_sent = sent

    if len(best_sent) > max_len:
        return best_sent[:max_len].strip() + "..."
    return best_sent.strip()


# ---------------------------------------------------------------------------
# Source dataclass
# ---------------------------------------------------------------------------

@dataclass
class SourceRecord:
    """A single attributed source."""

    doc_id: str
    source: str
    excerpt: str
    relevance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "excerpt": self.excerpt,
            "relevance": round(self.relevance, 3),
        }


# ---------------------------------------------------------------------------
# SourceAttributor
# ---------------------------------------------------------------------------

class SourceAttributor:
    """Attribute answers to their source documents and score confidence.

    Analyses how well a generated answer is grounded in retrieved context
    and produces transparency metrics.

    Parameters
    ----------
    min_relevance : float
        Minimum relevance score for a chunk to be listed as a source
        (default 0.1).
    """

    def __init__(self, min_relevance: float = 0.1) -> None:
        self._min_relevance = min_relevance
        logger.info("SourceAttributor initialised (min_relevance=%.2f).", min_relevance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        answer: str,
        query: str,
        retrieved_chunks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Attribute *answer* to *retrieved_chunks* and compute metrics.

        Parameters
        ----------
        answer : str
            The generated answer text.
        query : str
            The original user query.
        retrieved_chunks : sequence of dict
            Each dict must have at least ``"text"``; optional keys include
            ``"doc_id"``, ``"source"`` (or nested in ``"metadata"``),
            ``"score"``.

        Returns
        -------
        dict
            ``sources``       — list of source dicts.
            ``confidence``    — overall 0-1 confidence score.
            ``faithfulness``  — 0-1 fraction of answer sentences supported.
            ``coverage``      — 0-1 fraction of query key terms addressed.
            ``warning``       — None or a human-readable warning string.
        """
        if not answer or not answer.strip():
            return self._empty_result("Empty answer provided.")

        if not retrieved_chunks:
            return self._empty_result("No retrieved chunks available.")

        # 1. Score relevance of each chunk to the answer
        sources = self._score_sources(answer, query, retrieved_chunks)

        # 2. Compute faithfulness
        faithfulness = self._compute_faithfulness(answer, retrieved_chunks)

        # 3. Compute query-term coverage
        coverage = self._compute_coverage(query, retrieved_chunks)

        # 4. Compute overall confidence
        confidence = self._compute_confidence(
            sources=sources,
            faithfulness=faithfulness,
            coverage=coverage,
        )

        # 5. Generate warnings
        warning = self._generate_warning(
            confidence=confidence,
            faithfulness=faithfulness,
            coverage=coverage,
            num_sources=len(sources),
        )

        return {
            "sources": [s.to_dict() for s in sources],
            "confidence": round(confidence, 2),
            "faithfulness": round(faithfulness, 2),
            "coverage": round(coverage, 2),
            "warning": warning,
        }

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_sources(
        self,
        answer: str,
        query: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[SourceRecord]:
        """Score each chunk's relevance to the answer and return qualifying sources."""
        sources: List[SourceRecord] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue

            # Relevance = weighted combination of answer overlap and retriever score
            answer_overlap = _compute_text_overlap(answer, text)
            retriever_score = float(chunk.get("score", 0.0))
            # Normalise retriever score to 0-1 range (BM25 scores can be >1)
            retriever_norm = min(retriever_score, 1.0) if retriever_score <= 1.0 else min(retriever_score / 10.0, 1.0)

            relevance = 0.6 * answer_overlap + 0.4 * retriever_norm

            if relevance < self._min_relevance:
                continue

            # Extract doc_id and source
            doc_id = chunk.get("doc_id", chunk.get("chunk_id", "unknown"))
            metadata = chunk.get("metadata", {})
            source = chunk.get("source", metadata.get("source", "unknown"))

            excerpt = _best_excerpt(text, query)

            sources.append(SourceRecord(
                doc_id=str(doc_id),
                source=str(source),
                excerpt=excerpt,
                relevance=relevance,
            ))

        # Sort by relevance descending
        sources.sort(key=lambda s: s.relevance, reverse=True)
        return sources

    def _compute_faithfulness(
        self,
        answer: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> float:
        """Compute what fraction of answer sentences have supporting evidence.

        For each sentence in the answer, check if there is a chunk whose
        text has meaningful word overlap with the sentence. A sentence is
        "supported" if at least 30 % of its key terms appear in any single
        chunk.

        Returns
        -------
        float
            0-1 faithfulness score.
        """
        answer_sentences = _split_sentences(answer)
        if not answer_sentences:
            return 0.0

        chunk_texts = [c.get("text", "") for c in chunks if c.get("text")]
        if not chunk_texts:
            return 0.0

        supported_count = 0
        for sentence in answer_sentences:
            sentence_terms = set(_extract_key_terms(sentence))
            if not sentence_terms:
                # Trivial sentence (e.g. "Yes.") — count as supported
                supported_count += 1
                continue

            # Check each chunk for support
            is_supported = False
            for chunk_text in chunk_texts:
                chunk_terms = set(_extract_key_terms(chunk_text))
                overlap = len(sentence_terms & chunk_terms)
                # A sentence is supported if >= 30% of its terms appear in a chunk
                if overlap / len(sentence_terms) >= 0.3:
                    is_supported = True
                    break

            if is_supported:
                supported_count += 1

        return supported_count / len(answer_sentences)

    def _compute_coverage(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> float:
        """Compute what fraction of the query's key terms are covered by chunks.

        Returns
        -------
        float
            0-1 coverage score.
        """
        query_terms = set(_extract_key_terms(query))
        if not query_terms:
            return 1.0  # vacuously true

        all_chunk_terms: Set[str] = set()
        for chunk in chunks:
            text = chunk.get("text", "")
            if text:
                all_chunk_terms.update(_extract_key_terms(text))

        covered = len(query_terms & all_chunk_terms)
        return covered / len(query_terms)

    def _compute_confidence(
        self,
        sources: List[SourceRecord],
        faithfulness: float,
        coverage: float,
    ) -> float:
        """Aggregate individual metrics into an overall confidence score.

        Factors and weights:
        - Number of supporting sources (20 %)
        - Average relevance of sources (25 %)
        - Faithfulness (30 %)
        - Coverage (25 %)

        Returns
        -------
        float
            0-1 overall confidence.
        """
        # Source count factor (diminishing returns: 5 sources = max)
        source_count_score = min(len(sources) / 5.0, 1.0) if sources else 0.0

        # Average relevance
        avg_relevance = (
            sum(s.relevance for s in sources) / len(sources)
            if sources else 0.0
        )

        confidence = (
            0.20 * source_count_score
            + 0.25 * avg_relevance
            + 0.30 * faithfulness
            + 0.25 * coverage
        )
        return min(max(confidence, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Warning generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_warning(
        confidence: float,
        faithfulness: float,
        coverage: float,
        num_sources: int,
    ) -> Optional[str]:
        """Generate a human-readable warning when metrics are concerning.

        Returns
        -------
        str | None
            Warning string, or None if all metrics look healthy.
        """
        warnings: List[str] = []

        if confidence < 0.3:
            warnings.append("Low confidence - limited source material")
        elif confidence < 0.5:
            warnings.append("Moderate confidence - some sources may be tangential")

        if faithfulness < 0.4:
            warnings.append("Low faithfulness - answer may contain unsupported claims")

        if coverage < 0.4:
            warnings.append("Low coverage - key query terms not well addressed by sources")

        if num_sources == 0:
            warnings.append("No relevant sources identified")
        elif num_sources == 1:
            warnings.append("Only one source found - consider verifying independently")

        return "; ".join(warnings) if warnings else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(reason: str) -> Dict[str, Any]:
        """Return a result dict for degenerate cases."""
        return {
            "sources": [],
            "confidence": 0.0,
            "faithfulness": 0.0,
            "coverage": 0.0,
            "warning": reason,
        }


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    attributor = SourceAttributor()

    sample_query = "Which supplier caused most delays and why?"
    sample_answer = (
        "Acme Corp caused the most delays, with 5 incidents in the last quarter. "
        "The primary reason was part shortages at their Dallas facility, which "
        "led to delayed shipments including ORD-12345. Machine MCH-4401 failed "
        "because replacement parts were not delivered on time."
    )
    sample_chunks = [
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

    result = attributor.attribute(sample_answer, sample_query, sample_chunks)

    print("\n=== Source Attribution Demo ===\n")
    print(f"Confidence:  {result['confidence']}")
    print(f"Faithfulness: {result['faithfulness']}")
    print(f"Coverage:    {result['coverage']}")
    print(f"Warning:     {result['warning']}")
    print(f"\nSources ({len(result['sources'])}):")
    for src in result["sources"]:
        print(f"  [{src['relevance']:.2f}] {src['doc_id']} ({src['source']})")
        print(f"         \"{src['excerpt'][:80]}...\"")
