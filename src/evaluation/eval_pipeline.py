"""Comprehensive RAG evaluation pipeline.

Computes retrieval precision, retrieval recall, answer relevance,
faithfulness, and latency for individual queries and full batches.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import EMBEDDING_MODEL, EVALUATION_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight NLP helpers (no heavy deps beyond numpy)
# ---------------------------------------------------------------------------

_STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "when", "where", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it", "its",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop-words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _extract_entities(text: str) -> set[str]:
    """Heuristic entity extraction — capitalised multi-word spans + numbers."""
    entities: set[str] = set()
    # Capitalised word sequences (crude NER)
    for match in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text):
        entities.add(match.group().lower())
    # Numbers / percentages
    for match in re.finditer(r"\d+(?:\.\d+)?%?", text):
        entities.add(match.group())
    return entities


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity over non-stop-word tokens."""
    set_a = set(_tokenize(text_a))
    set_b = set(_tokenize(text_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = float(np.dot(vec_a, vec_b))
    norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if norm < 1e-12:
        return 0.0
    return dot / norm


# ---------------------------------------------------------------------------
# Claim extraction for faithfulness
# ---------------------------------------------------------------------------

def _extract_claims(answer: str) -> list[str]:
    """Split an answer into individual sentence-level claims.

    Each non-trivial sentence is treated as a separate claim that must be
    supportable by the retrieved context.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    return claims


def _claim_supported(claim: str, context: str) -> float:
    """Score how well *claim* is supported by *context* (0.0 – 1.0).

    Uses keyword overlap as a lightweight proxy.  A more sophisticated
    implementation could call an LLM-based NLI model.
    """
    return min(_keyword_overlap(claim, context) * 3.0, 1.0)


# ---------------------------------------------------------------------------
# RAGEvaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """End-to-end evaluator for the RAG pipeline.

    Parameters
    ----------
    embedding_engine : object | None
        An ``EmbeddingEngine`` instance.  When provided, answer-relevance
        is computed via cosine similarity of embeddings.  When *None*,
        a keyword-based fallback is used.
    """

    def __init__(self, embedding_engine: Any = None) -> None:
        self._embedding_engine = embedding_engine
        logger.info("RAGEvaluator initialised (embedding_engine=%s).",
                     type(embedding_engine).__name__ if embedding_engine else "None")

    # ------------------------------------------------------------------
    # Retrieval precision
    # ------------------------------------------------------------------

    @staticmethod
    def _retrieval_precision(
        retrieved_chunks: Sequence[str],
        ground_truth: str,
        threshold: float = 0.05,
    ) -> float:
        """Fraction of retrieved chunks that are relevant to *ground_truth*.

        Relevance is determined by keyword overlap exceeding *threshold*.
        """
        if not retrieved_chunks:
            return 0.0
        relevant = sum(
            1 for chunk in retrieved_chunks
            if _keyword_overlap(chunk, ground_truth) >= threshold
        )
        return relevant / len(retrieved_chunks)

    # ------------------------------------------------------------------
    # Retrieval recall
    # ------------------------------------------------------------------

    @staticmethod
    def _retrieval_recall(
        retrieved_chunks: Sequence[str],
        ground_truth: str,
    ) -> float:
        """Estimated recall via keyword/entity overlap with ground truth.

        Measures what fraction of ground-truth entities/keywords appear
        in at least one retrieved chunk.
        """
        gt_tokens = set(_tokenize(ground_truth))
        gt_entities = _extract_entities(ground_truth)
        target_items = gt_tokens | gt_entities
        if not target_items:
            return 0.0

        found: set[str] = set()
        for chunk in retrieved_chunks:
            chunk_tokens = set(_tokenize(chunk))
            chunk_entities = _extract_entities(chunk)
            found |= (chunk_tokens | chunk_entities) & target_items

        return len(found) / len(target_items)

    # ------------------------------------------------------------------
    # Answer relevance
    # ------------------------------------------------------------------

    def _answer_relevance(self, query: str, answer: str) -> float:
        """Score how well *answer* addresses *query* (0.0 – 1.0).

        Uses cosine similarity of embeddings when an embedding engine is
        available; falls back to keyword overlap otherwise.
        """
        if self._embedding_engine is not None:
            try:
                q_vec = self._embedding_engine.embed_query(query)
                a_vec = self._embedding_engine.embed_query(answer)
                return float(np.clip(_cosine_similarity(q_vec, a_vec), 0.0, 1.0))
            except Exception as exc:
                logger.warning("Embedding-based relevance failed (%s); "
                               "falling back to keyword overlap.", exc)
        # Fallback
        return min(_keyword_overlap(query, answer) * 2.5, 1.0)

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------

    @staticmethod
    def _faithfulness(answer: str, retrieved_chunks: Sequence[str]) -> float:
        """Score whether the *answer* stays faithful to retrieved context.

        Extracts sentence-level claims from the answer and checks each
        against the concatenated retrieved context.
        """
        if not answer.strip() or not retrieved_chunks:
            return 0.0

        context = " ".join(retrieved_chunks)
        claims = _extract_claims(answer)
        if not claims:
            return 1.0  # trivially faithful if no substantive claims

        scores = [_claim_supported(claim, context) for claim in claims]
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        query: str,
        answer: str,
        retrieved_chunks: Sequence[str],
        ground_truth: Optional[str] = None,
        *,
        latency_ms: Optional[float] = None,
        expected_answer_contains: Optional[list[str]] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single RAG interaction.

        Parameters
        ----------
        query : str
            The user query.
        answer : str
            The generated answer.
        retrieved_chunks : Sequence[str]
            Texts of retrieved chunks.
        ground_truth : str | None
            Reference answer / relevant passage for precision/recall.
        latency_ms : float | None
            Pre-measured latency; if *None* the metric is omitted.
        expected_answer_contains : list[str] | None
            Keywords the answer is expected to contain (for test-case
            validation).
        category : str | None
            Query category label (e.g. ``"factual"``, ``"sql"``).

        Returns
        -------
        dict
            Metric name -> value mapping.
        """
        t_start = time.perf_counter()

        # Use ground_truth if provided; otherwise use the answer itself as
        # a weak reference for retrieval metrics.
        gt = ground_truth if ground_truth else answer

        metrics: Dict[str, Any] = {
            "query": query,
            "category": category,
            "retrieval_precision": self._retrieval_precision(retrieved_chunks, gt),
            "retrieval_recall": self._retrieval_recall(retrieved_chunks, gt),
            "answer_relevance": self._answer_relevance(query, answer),
            "faithfulness": self._faithfulness(answer, retrieved_chunks),
        }

        # Expected-keyword hit rate
        if expected_answer_contains:
            answer_lower = answer.lower()
            hits = sum(1 for kw in expected_answer_contains if kw.lower() in answer_lower)
            metrics["keyword_hit_rate"] = hits / len(expected_answer_contains)
        else:
            metrics["keyword_hit_rate"] = None

        # Latency
        eval_time_ms = (time.perf_counter() - t_start) * 1000.0
        metrics["latency_ms"] = latency_ms
        metrics["eval_overhead_ms"] = round(eval_time_ms, 2)

        return metrics

    def evaluate_batch(
        self,
        test_cases: list[Dict[str, Any]],
        *,
        run_pipeline_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Evaluate a batch of test cases.

        Parameters
        ----------
        test_cases : list[dict]
            Each dict must contain ``"query"`` and optionally
            ``"expected_answer_contains"``, ``"ground_truth"``,
            ``"category"``.  If *run_pipeline_fn* is *None*, the dict
            must also include ``"answer"`` and ``"retrieved_chunks"``.
        run_pipeline_fn : callable | None
            ``(query: str) -> dict`` returning at least ``"answer"``
            and ``"retrieved_chunks"``.  When provided, queries are
            executed through the pipeline automatically.

        Returns
        -------
        dict
            ``{"per_case": [...], "aggregated": {...}, "per_category": {...}}``
        """
        per_case: list[Dict[str, Any]] = []
        category_buckets: Dict[str, list[Dict[str, Any]]] = defaultdict(list)

        for idx, tc in enumerate(test_cases):
            query = tc["query"]
            category = tc.get("category")
            expected = tc.get("expected_answer_contains")
            ground_truth = tc.get("ground_truth")

            # Execute pipeline if a runner is provided
            if run_pipeline_fn is not None:
                t0 = time.perf_counter()
                try:
                    result = run_pipeline_fn(query)
                except Exception as exc:
                    logger.error("Pipeline failed on query %d (%s): %s",
                                 idx, query[:60], exc)
                    per_case.append({
                        "query": query,
                        "category": category,
                        "error": str(exc),
                    })
                    continue
                latency_ms = (time.perf_counter() - t0) * 1000.0
                answer = result.get("answer", "")
                retrieved_chunks = result.get("retrieved_chunks", [])
            else:
                answer = tc.get("answer", "")
                retrieved_chunks = tc.get("retrieved_chunks", [])
                latency_ms = tc.get("latency_ms")

            result_metrics = self.evaluate_single(
                query=query,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                ground_truth=ground_truth,
                latency_ms=latency_ms,
                expected_answer_contains=expected,
                category=category,
            )
            per_case.append(result_metrics)
            if category:
                category_buckets[category].append(result_metrics)

        # Aggregate
        aggregated = self._aggregate_metrics(per_case)
        per_category = {
            cat: self._aggregate_metrics(cases)
            for cat, cases in category_buckets.items()
        }

        return {
            "per_case": per_case,
            "aggregated": aggregated,
            "per_category": per_category,
        }

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Save a JSON evaluation report.

        This is a convenience wrapper around
        :class:`~src.evaluation.report_generator.ReportGenerator`.

        Parameters
        ----------
        results : dict
            Output of :meth:`evaluate_batch`.
        output_path : str | Path | None
            Destination directory.  Defaults to ``EVALUATION_DIR``.

        Returns
        -------
        Path
            Path to the saved JSON report.
        """
        from src.evaluation.report_generator import ReportGenerator

        out_dir = Path(output_path) if output_path else EVALUATION_DIR
        generator = ReportGenerator(output_dir=out_dir)
        return generator.generate_json_report(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(cases: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute mean / min / max for each numeric metric."""
        metric_keys = [
            "retrieval_precision",
            "retrieval_recall",
            "answer_relevance",
            "faithfulness",
            "keyword_hit_rate",
            "latency_ms",
        ]
        agg: Dict[str, Any] = {"num_cases": len(cases)}

        for key in metric_keys:
            values = [
                c[key] for c in cases
                if key in c and c[key] is not None and not isinstance(c.get("error"), str)
            ]
            if values:
                arr = np.array(values, dtype=np.float64)
                agg[key] = {
                    "mean": round(float(np.mean(arr)), 4),
                    "median": round(float(np.median(arr)), 4),
                    "min": round(float(np.min(arr)), 4),
                    "max": round(float(np.max(arr)), 4),
                    "std": round(float(np.std(arr)), 4),
                }
            else:
                agg[key] = None

        return agg
