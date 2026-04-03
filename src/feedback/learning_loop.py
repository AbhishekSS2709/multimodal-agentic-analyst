"""Feedback learning loop — analyse patterns, suggest improvements, adjust weights.

Implements the closed loop: collect feedback -> analyse -> adjust
retrieval parameters -> evaluate -> repeat.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import HYBRID_BM25_WEIGHT, HYBRID_VECTOR_WEIGHT

from src.feedback.feedback_store import FeedbackStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeedbackLearner
# ---------------------------------------------------------------------------

class FeedbackLearner:
    """Analyse user feedback and derive actionable improvements.

    Parameters
    ----------
    feedback_store : FeedbackStore
        The store from which raw feedback and retrieval logs are read.
    low_rating_threshold : int
        Ratings at or below this value are considered "poor" (default 3).
    """

    def __init__(
        self,
        feedback_store: FeedbackStore,
        low_rating_threshold: int = 3,
    ) -> None:
        self._store = feedback_store
        self._low_threshold = low_rating_threshold
        logger.info("FeedbackLearner initialised (low_threshold=%d).",
                     low_rating_threshold)

    # ------------------------------------------------------------------
    # Analyse feedback
    # ------------------------------------------------------------------

    def analyze_feedback(self) -> Dict[str, Any]:
        """Mine patterns from accumulated feedback.

        Returns
        -------
        dict
            Structured insights including:

            - ``overall_stats`` — high-level numbers from FeedbackStore
            - ``low_rated_categories`` — categories with the worst ratings
            - ``common_failure_keywords`` — frequent words in low-rated comments
            - ``chunk_relevance_rates`` — per-chunk relevance fractions
            - ``top_problematic_chunks`` — chunks most frequently judged irrelevant
            - ``rating_trend`` — rating averages bucketed by date
        """
        stats = self._store.get_feedback_stats()
        low_queries = self._store.get_low_rated_queries(self._low_threshold)

        # Category analysis
        category_stats = stats.get("category_stats", {})
        low_rated_categories = sorted(
            [
                {"category": cat, **vals}
                for cat, vals in category_stats.items()
                if vals["avg_rating"] <= self._low_threshold
            ],
            key=lambda x: x["avg_rating"],
        )

        # Common failure keywords from comments
        failure_keywords = self._extract_failure_keywords(low_queries)

        # Chunk relevance analysis
        chunk_relevance = self._chunk_relevance_analysis()

        # Rating trend over time
        rating_trend = self._rating_trend()

        insights: Dict[str, Any] = {
            "overall_stats": stats,
            "low_rated_categories": low_rated_categories,
            "common_failure_keywords": failure_keywords,
            "chunk_relevance_rates": chunk_relevance.get("per_chunk", {}),
            "top_problematic_chunks": chunk_relevance.get("worst_chunks", []),
            "rating_trend": rating_trend,
            "total_low_rated_queries": len(low_queries),
        }
        logger.info("Feedback analysis complete: %d low-rated queries, "
                     "%d under-performing categories.",
                     len(low_queries), len(low_rated_categories))
        return insights

    # ------------------------------------------------------------------
    # Suggest improvements
    # ------------------------------------------------------------------

    def suggest_improvements(self) -> List[Dict[str, str]]:
        """Generate actionable improvement suggestions based on feedback.

        Returns
        -------
        list[dict]
            Each dict has ``area``, ``issue``, and ``suggestion``.
        """
        analysis = self.analyze_feedback()
        suggestions: List[Dict[str, str]] = []

        # 1. Low-rated categories
        for cat_info in analysis.get("low_rated_categories", []):
            cat = cat_info["category"]
            avg = cat_info["avg_rating"]
            suggestions.append({
                "area": "query_category",
                "issue": (f"Category '{cat}' has a low average rating of "
                          f"{avg:.2f}."),
                "suggestion": (f"Review and improve the retrieval/generation "
                               f"strategy for '{cat}' queries. Consider adding "
                               f"specialised prompts or additional data sources."),
            })

        # 2. Problematic chunks
        for chunk_info in analysis.get("top_problematic_chunks", [])[:5]:
            chunk_id = chunk_info["chunk_id"]
            rel_rate = chunk_info["relevance_rate"]
            suggestions.append({
                "area": "retrieval_quality",
                "issue": (f"Chunk '{chunk_id}' has a relevance rate of "
                          f"{rel_rate:.1%} — frequently retrieved but rarely "
                          f"useful."),
                "suggestion": (f"Consider re-chunking, re-embedding, or "
                               f"demoting chunk '{chunk_id}' in the index. "
                               f"It may contain noisy or off-topic content."),
            })

        # 3. Keyword signals from failure comments
        failure_kw = analysis.get("common_failure_keywords", [])
        if failure_kw:
            top_kw = [kw for kw, _ in failure_kw[:5]]
            suggestions.append({
                "area": "answer_quality",
                "issue": (f"Common themes in negative feedback: "
                          f"{', '.join(top_kw)}."),
                "suggestion": ("Investigate whether these themes point to "
                               "missing data, incorrect retrieval, or "
                               "generation hallucinations. Add guardrails "
                               "or expand the knowledge base accordingly."),
            })

        # 4. Overall rating health
        overall = analysis.get("overall_stats", {})
        avg = overall.get("avg_rating")
        if avg is not None and avg < 3.5:
            suggestions.append({
                "area": "overall_pipeline",
                "issue": (f"Overall average rating is {avg:.2f}, below the "
                          f"3.5 target."),
                "suggestion": ("Run a full evaluation batch to identify "
                               "system-wide bottlenecks. Consider adjusting "
                               "hybrid retrieval weights or upgrading the LLM."),
            })

        # 5. Weight tuning hint
        suggestions.append({
            "area": "retrieval_weights",
            "issue": ("Hybrid retrieval weights may not be optimal for the "
                      "current query distribution."),
            "suggestion": ("Use `adjust_retrieval_weights()` to compute "
                           "data-driven weights from retrieval relevance logs."),
        })

        logger.info("Generated %d improvement suggestions.", len(suggestions))
        return suggestions

    # ------------------------------------------------------------------
    # Adjust retrieval weights
    # ------------------------------------------------------------------

    def adjust_retrieval_weights(
        self,
        feedback_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Compute updated hybrid-retrieval weights from feedback data.

        The idea: if chunks found via BM25 tend to be marked relevant more
        often than those found via vector search (or vice versa), shift
        the weight toward the more effective source.

        Parameters
        ----------
        feedback_data : list[dict] | None
            Custom dataset.  When *None*, data is pulled from the store's
            retrieval logs.

        Returns
        -------
        dict
            ``{"vector_weight": float, "bm25_weight": float,
               "confidence": float, "sample_size": int}``
        """
        if feedback_data is None:
            feedback_data = self._get_retrieval_relevance_data()

        if not feedback_data:
            logger.warning("No retrieval-relevance data available; "
                           "returning current weights.")
            return {
                "vector_weight": HYBRID_VECTOR_WEIGHT,
                "bm25_weight": HYBRID_BM25_WEIGHT,
                "confidence": 0.0,
                "sample_size": 0,
            }

        # Separate scores by relevance judgment
        relevant_scores: list[float] = []
        irrelevant_scores: list[float] = []
        for item in feedback_data:
            if item.get("was_relevant"):
                relevant_scores.append(item.get("score", 0.0))
            else:
                irrelevant_scores.append(item.get("score", 0.0))

        # Compute a relevance-weighted adjustment factor.
        # Chunks with higher scores that are relevant validate the current
        # balance; chunks with high scores that are irrelevant suggest we
        # should shift.
        total = len(feedback_data)
        if total < 10:
            logger.warning("Only %d samples — low confidence.", total)

        relevance_rate = len(relevant_scores) / total if total > 0 else 0.5

        # Heuristic: if relevance rate is high, the current weighting works.
        # If it's low, increase BM25 weight (keyword precision) slightly.
        # This is a simple linear adjustment; production systems would use
        # Bayesian optimisation or grid search.
        current_vec = HYBRID_VECTOR_WEIGHT
        current_bm25 = HYBRID_BM25_WEIGHT

        # Adjustment magnitude scales with how far relevance is from 0.7 target
        target_relevance = 0.7
        delta = (relevance_rate - target_relevance) * 0.2  # damped adjustment

        new_vec = np.clip(current_vec + delta, 0.1, 0.9)
        new_bm25 = 1.0 - new_vec

        confidence = min(total / 100.0, 1.0)  # saturates at 100 samples

        result = {
            "vector_weight": round(float(new_vec), 4),
            "bm25_weight": round(float(new_bm25), 4),
            "confidence": round(float(confidence), 4),
            "sample_size": total,
            "relevance_rate": round(float(relevance_rate), 4),
            "previous_vector_weight": HYBRID_VECTOR_WEIGHT,
            "previous_bm25_weight": HYBRID_BM25_WEIGHT,
        }
        logger.info("Adjusted weights: vector=%.4f, bm25=%.4f "
                     "(confidence=%.2f, n=%d).",
                     result["vector_weight"], result["bm25_weight"],
                     confidence, total)
        return result

    # ------------------------------------------------------------------
    # Full learning loop
    # ------------------------------------------------------------------

    def run_learning_loop(
        self,
        evaluator: Optional[Any] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute one full iteration of the feedback learning loop.

        Steps:
        1. Analyse accumulated feedback.
        2. Suggest improvements.
        3. Compute adjusted retrieval weights.
        4. Optionally re-evaluate with the new weights.

        Parameters
        ----------
        evaluator : RAGEvaluator | None
            If provided, a follow-up evaluation is run.
        test_cases : list[dict] | None
            Test cases to re-evaluate (used only when *evaluator* is given).

        Returns
        -------
        dict
            ``{"analysis": ..., "suggestions": ..., "new_weights": ...,
               "evaluation": ...}``
        """
        logger.info("Starting learning loop iteration.")

        analysis = self.analyze_feedback()
        suggestions = self.suggest_improvements()
        new_weights = self.adjust_retrieval_weights()

        evaluation_result: Optional[Dict[str, Any]] = None
        if evaluator is not None and test_cases is not None:
            logger.info("Re-evaluating with updated weights.")
            evaluation_result = evaluator.evaluate_batch(test_cases)

        result = {
            "analysis": analysis,
            "suggestions": suggestions,
            "new_weights": new_weights,
            "evaluation": evaluation_result,
        }
        logger.info("Learning loop iteration complete.")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_failure_keywords(
        low_queries: List[Dict[str, Any]],
    ) -> List[Tuple[str, int]]:
        """Extract frequently occurring words from low-rated comments."""
        import re

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "it", "this",
            "that", "not", "but", "and", "or", "to", "of", "in", "for",
            "on", "with", "as", "at", "by", "from", "be", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "i", "my", "me", "we", "you", "your", "they",
        }
        counter: Counter[str] = Counter()
        for q in low_queries:
            comment = q.get("comment") or ""
            words = re.findall(r"[a-z]+", comment.lower())
            counter.update(w for w in words if w not in stop_words and len(w) > 2)
        return counter.most_common(20)

    def _chunk_relevance_analysis(self) -> Dict[str, Any]:
        """Compute per-chunk relevance rates from retrieval logs."""
        import sqlite3

        per_chunk: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"relevant": 0, "irrelevant": 0, "total": 0}
        )

        try:
            with self._store._connection() as conn:
                rows = conn.execute(
                    "SELECT chunk_id, was_relevant FROM retrieval_logs "
                    "WHERE was_relevant IS NOT NULL"
                ).fetchall()
        except Exception as exc:
            logger.warning("Could not read retrieval logs: %s", exc)
            return {"per_chunk": {}, "worst_chunks": []}

        for r in rows:
            chunk = r["chunk_id"]
            per_chunk[chunk]["total"] += 1
            if r["was_relevant"] == 1:
                per_chunk[chunk]["relevant"] += 1
            else:
                per_chunk[chunk]["irrelevant"] += 1

        # Compute relevance rates
        chunk_rates: Dict[str, float] = {}
        for cid, counts in per_chunk.items():
            if counts["total"] > 0:
                chunk_rates[cid] = counts["relevant"] / counts["total"]

        # Worst chunks (most frequently irrelevant, minimum 2 appearances)
        worst = sorted(
            [
                {"chunk_id": cid, "relevance_rate": rate, **per_chunk[cid]}
                for cid, rate in chunk_rates.items()
                if per_chunk[cid]["total"] >= 2
            ],
            key=lambda x: x["relevance_rate"],
        )

        return {"per_chunk": chunk_rates, "worst_chunks": worst[:10]}

    def _rating_trend(self) -> List[Dict[str, Any]]:
        """Compute daily average ratings from feedback history."""
        try:
            with self._store._connection() as conn:
                rows = conn.execute(
                    "SELECT DATE(timestamp) AS day, "
                    "       COUNT(*) AS cnt, "
                    "       AVG(user_rating) AS avg_r "
                    "FROM feedback "
                    "GROUP BY DATE(timestamp) "
                    "ORDER BY day"
                ).fetchall()
        except Exception as exc:
            logger.warning("Could not compute rating trend: %s", exc)
            return []

        return [
            {
                "date": r["day"],
                "count": r["cnt"],
                "avg_rating": round(r["avg_r"], 2),
            }
            for r in rows
        ]

    def _get_retrieval_relevance_data(self) -> List[Dict[str, Any]]:
        """Pull retrieval log entries that have relevance judgments."""
        try:
            with self._store._connection() as conn:
                rows = conn.execute(
                    "SELECT query, chunk_id, rank, score, was_relevant "
                    "FROM retrieval_logs WHERE was_relevant IS NOT NULL"
                ).fetchall()
        except Exception as exc:
            logger.warning("Could not read retrieval relevance data: %s", exc)
            return []

        return [
            {
                "query": r["query"],
                "chunk_id": r["chunk_id"],
                "rank": r["rank"],
                "score": r["score"],
                "was_relevant": bool(r["was_relevant"]),
            }
            for r in rows
        ]
