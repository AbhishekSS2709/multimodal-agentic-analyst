"""Multimodal retriever with score normalization and result fusion."""

import logging
from typing import Callable, Dict, List, Optional

from src.retrieval.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


def normalize_scores(results: List[dict]) -> List[dict]:
    """Min-max normalize the ``score`` key of each result to [0, 1].

    The normalised value is stored under the new ``normalized_score`` key.
    The original ``score`` key is left unchanged.

    Edge cases
    ----------
    - Empty list  -> returns ``[]``.
    - Single item -> ``normalized_score`` is set to ``1.0``.
    - All scores equal (min == max) -> all ``normalized_score`` values are ``1.0``.
    """
    if not results:
        return []

    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)
    span = max_s - min_s

    out = []
    for r in results:
        copy = dict(r)
        if span == 0:
            copy["normalized_score"] = 1.0
        else:
            copy["normalized_score"] = (r["score"] - min_s) / span
        out.append(copy)
    return out


class MultimodalRetriever:
    """Fuse text and visual search results weighted by query modality.

    Parameters
    ----------
    text_search_fn:
        Callable ``(query: str, top_k: int) -> List[dict]``.
        Each result dict must contain at least ``doc_id``, ``page_or_slide``,
        and ``score``.
    visual_search_fn:
        Optional callable with the same signature.  When ``None`` only text
        retrieval is performed.
    query_analyzer:
        Optional :class:`~src.retrieval.query_analyzer.QueryAnalyzer` instance.
        A default instance is created if not provided.
    """

    def __init__(
        self,
        text_search_fn: Callable,
        visual_search_fn: Optional[Callable] = None,
        query_analyzer: Optional[object] = None,
    ) -> None:
        self.text_search_fn = text_search_fn
        self.visual_search_fn = visual_search_fn
        self.query_analyzer = query_analyzer if query_analyzer is not None else QueryAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        text_weight: Optional[float] = None,
        visual_weight: Optional[float] = None,
    ) -> List[dict]:
        """Retrieve and fuse results for *query*.

        Parameters
        ----------
        query:
            User query string.
        top_k:
            Maximum number of results to return.
        text_weight:
            Override the analyzer's text weight (0–1).
        visual_weight:
            Override the analyzer's visual weight (0–1).

        Returns
        -------
        List[dict]
            Up to *top_k* result dicts, sorted by ``final_score`` descending.
            Each dict contains a ``final_score`` key (the weighted, normalised score).
        """
        # 1. Determine modality weights
        analysis = self.query_analyzer.analyze(query)
        tw = text_weight if text_weight is not None else analysis["text_weight"]
        vw = visual_weight if visual_weight is not None else analysis["visual_weight"]

        logger.debug(
            "MultimodalRetriever: query=%r text_weight=%.2f visual_weight=%.2f",
            query, tw, vw,
        )

        fused: List[dict] = []

        # 2. Text search
        raw_text = self.text_search_fn(query, top_k=top_k * 2)
        for r in normalize_scores(raw_text):
            r["final_score"] = r["normalized_score"] * tw
            fused.append(r)

        # 3. Visual search (only when a function is registered and weight is meaningful)
        if self.visual_search_fn is not None and vw > 0.05:
            raw_visual = self.visual_search_fn(query, top_k=top_k * 2)
            for r in normalize_scores(raw_visual):
                r["final_score"] = r["normalized_score"] * vw
                fused.append(r)

        # 4. Deduplicate by (doc_id, page_or_slide) — prefer "image" type entries
        deduped = self._deduplicate(fused)

        # 5. Sort descending by final_score and return top_k
        deduped.sort(key=lambda r: r["final_score"], reverse=True)
        return deduped[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(results: List[dict]) -> List[dict]:
        """Deduplicate *results* by ``(doc_id, page_or_slide)``.

        When multiple entries share the same key, the ``image`` type wins.
        Among equal-type entries the one with the higher ``final_score`` wins.
        """
        best: Dict[tuple, dict] = {}
        for r in results:
            key = (r.get("doc_id"), r.get("page_or_slide"))
            if key not in best:
                best[key] = r
            else:
                incumbent = best[key]
                # "image" beats everything else
                r_is_image = r.get("type") == "image"
                inc_is_image = incumbent.get("type") == "image"
                if r_is_image and not inc_is_image:
                    best[key] = r
                elif not r_is_image and inc_is_image:
                    pass  # keep incumbent
                else:
                    # same type priority — keep higher score
                    if r["final_score"] > incumbent["final_score"]:
                        best[key] = r
        return list(best.values())
