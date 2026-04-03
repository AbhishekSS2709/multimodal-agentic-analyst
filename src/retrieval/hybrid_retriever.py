"""Hybrid retrieval combining dense vector search with BM25 keyword search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    TOP_K,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocols so callers are not forced to import concrete classes
# ---------------------------------------------------------------------------

class VectorRetriever(Protocol):
    """Minimal interface that any vector retriever must satisfy."""

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return ``(chunk_index, similarity_score)`` pairs."""
        ...


class BM25Searcher(Protocol):
    """Minimal interface that any BM25 searcher must satisfy."""

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return ``(chunk_index, bm25_score)`` pairs."""
        ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """A single result from hybrid retrieval."""

    chunk_index: int
    final_score: float
    vector_score: float
    bm25_score: float
    chunk: Any = None  # populated when a chunk store is provided

    def as_tuple(self) -> Tuple[Any, float, float, float]:
        """Return ``(chunk, final_score, vector_score, bm25_score)``."""
        return (self.chunk, self.final_score, self.vector_score, self.bm25_score)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _min_max_normalize(
    scores: Dict[int, float],
) -> Dict[int, float]:
    """Apply min-max normalization to *scores*, mapping them into [0, 1].

    If all scores are identical the function returns 1.0 for every key,
    avoiding a division-by-zero.
    """
    if not scores:
        return {}

    min_s = min(scores.values())
    max_s = max(scores.values())
    span = max_s - min_s

    if span == 0.0:
        return {k: 1.0 for k in scores}

    return {k: (v - min_s) / span for k, v in scores.items()}


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
) -> Dict[int, float]:
    """Reciprocal Rank Fusion (RRF) across multiple ranked result lists.

    Each document's fused score is the sum of ``1 / (k + rank)`` across
    every list in which it appears.  The constant *k* (default 60) is the
    standard smoothing parameter from the original paper by Cormack,
    Clarke & Buettcher (2009).

    Parameters
    ----------
    ranked_lists : list[list[tuple[int, float]]]
        Each inner list contains ``(chunk_index, score)`` pairs sorted
        by descending score.
    k : int
        Smoothing constant.

    Returns
    -------
    dict[int, float]
        Mapping of ``chunk_index -> rrf_score``.
    """
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_idx, _score) in enumerate(ranked, start=1):
            fused[chunk_idx] = fused.get(chunk_idx, 0.0) + 1.0 / (k + rank)
    return fused


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Merge dense vector and sparse BM25 results with configurable weights.

    Parameters
    ----------
    vector_retriever
        Any object that exposes ``search(query, top_k) -> list[(idx, score)]``.
    bm25_searcher
        Any object that exposes ``search(query, top_k) -> list[(idx, score)]``.
    vector_weight : float
        Weight applied to the normalised vector score (default from settings).
    bm25_weight : float
        Weight applied to the normalised BM25 score (default from settings).
    chunks : list | None
        Optional ordered list of chunk objects.  When provided, results
        will include the chunk object itself (via :pyattr:`HybridResult.chunk`).
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_searcher: BM25Searcher,
        vector_weight: float = HYBRID_VECTOR_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        chunks: Optional[List[Any]] = None,
    ) -> None:
        self._vector = vector_retriever
        self._bm25 = bm25_searcher
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self._chunks = chunks

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        *,
        strategy: str = "weighted",
    ) -> List[Tuple[Any, float, float, float]]:
        """Run hybrid retrieval and return the top results.

        Parameters
        ----------
        query : str
            Natural language query.
        top_k : int
            Number of results to return.
        strategy : str
            Merging strategy — ``"weighted"`` for normalised weighted sum
            (default) or ``"rrf"`` for reciprocal rank fusion.

        Returns
        -------
        list[tuple[chunk, final_score, vector_score, bm25_score]]
            Sorted by descending *final_score*.  *chunk* is the chunk
            object when a chunk store was provided, otherwise the
            integer chunk index.
        """
        fetch_k = top_k * 2

        # 1. Gather candidates from both sources
        vector_results = self._vector.search(query, top_k=fetch_k)
        bm25_results = self._bm25.search(query, top_k=fetch_k)

        logger.debug(
            "Vector returned %d results, BM25 returned %d results.",
            len(vector_results),
            len(bm25_results),
        )

        if strategy == "rrf":
            return self._merge_rrf(vector_results, bm25_results, top_k)
        return self._merge_weighted(vector_results, bm25_results, top_k)

    # ------------------------------------------------------------------
    # Merging strategies
    # ------------------------------------------------------------------

    def _merge_weighted(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        top_k: int,
    ) -> List[Tuple[Any, float, float, float]]:
        """Weighted linear combination after min-max normalization."""

        # Build raw score dicts
        vector_raw: Dict[int, float] = {idx: score for idx, score in vector_results}
        bm25_raw: Dict[int, float] = {idx: score for idx, score in bm25_results}

        # Normalize
        vector_norm = _min_max_normalize(vector_raw)
        bm25_norm = _min_max_normalize(bm25_raw)

        # Collect all candidate indices
        all_indices = set(vector_norm.keys()) | set(bm25_norm.keys())

        merged: List[HybridResult] = []
        for idx in all_indices:
            v_score = vector_norm.get(idx, 0.0)
            b_score = bm25_norm.get(idx, 0.0)
            final = self.vector_weight * v_score + self.bm25_weight * b_score
            chunk = self._resolve_chunk(idx)
            merged.append(
                HybridResult(
                    chunk_index=idx,
                    final_score=final,
                    vector_score=v_score,
                    bm25_score=b_score,
                    chunk=chunk,
                )
            )

        merged.sort(key=lambda r: r.final_score, reverse=True)
        return [r.as_tuple() for r in merged[:top_k]]

    def _merge_rrf(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        top_k: int,
    ) -> List[Tuple[Any, float, float, float]]:
        """Reciprocal Rank Fusion merging strategy."""

        fused_scores = _reciprocal_rank_fusion([vector_results, bm25_results])

        # We still want to report per-source scores for transparency.
        vector_raw: Dict[int, float] = {idx: score for idx, score in vector_results}
        bm25_raw: Dict[int, float] = {idx: score for idx, score in bm25_results}
        vector_norm = _min_max_normalize(vector_raw)
        bm25_norm = _min_max_normalize(bm25_raw)

        merged: List[HybridResult] = []
        for idx, rrf_score in fused_scores.items():
            chunk = self._resolve_chunk(idx)
            merged.append(
                HybridResult(
                    chunk_index=idx,
                    final_score=rrf_score,
                    vector_score=vector_norm.get(idx, 0.0),
                    bm25_score=bm25_norm.get(idx, 0.0),
                    chunk=chunk,
                )
            )

        merged.sort(key=lambda r: r.final_score, reverse=True)
        return [r.as_tuple() for r in merged[:top_k]]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_chunk(self, index: int) -> Any:
        """Return the chunk object for *index*, or the index itself."""
        if self._chunks is not None and 0 <= index < len(self._chunks):
            return self._chunks[index]
        return index
