"""Reranking modules for post-retrieval result refinement."""

from __future__ import annotations

import logging
import re
import string
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pre-compile punctuation regex (shared with bm25_search tokenizer logic)
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _tokenize_simple(text: str) -> set[str]:
    """Return a set of lowercase tokens with punctuation stripped."""
    cleaned = _PUNCT_RE.sub("", text.lower())
    return {tok for tok in cleaned.split() if tok}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseReranker(ABC):
    """Interface all rerankers must implement."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[Tuple[Any, float]],
        top_k: int = 5,
    ) -> List[Tuple[Any, float]]:
        """Rerank *results* for *query* and return the top *top_k*.

        Parameters
        ----------
        query : str
            The user query.
        results : list[tuple[Any, float]]
            Each element is ``(chunk, score)``.  *chunk* can be any object
            that has a ``text`` attribute **or** is itself a string.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[tuple[Any, float]]
            Reranked ``(chunk, new_score)`` pairs sorted by descending
            score.
        """
        ...


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseReranker):
    """Rerank results using a cross-encoder model from sentence-transformers.

    The cross-encoder scores each ``(query, document)`` pair jointly,
    producing a relevance score that is typically more accurate than
    the bi-encoder similarity used during first-stage retrieval.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for a cross-encoder.  Defaults to
        ``cross-encoder/ms-marco-MiniLM-L-6-v2`` which offers a good
        trade-off between speed and accuracy.
    device : str | None
        Torch device string (``"cpu"``, ``"cuda"``, …).  ``None`` lets
        the library choose automatically.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._device = device

    # Lazy-load so importing the module is cheap and tests that don't
    # exercise cross-encoding never need the model weights.
    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self._device)
                logger.info("Loaded cross-encoder model: %s", self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        results: List[Tuple[Any, float]],
        top_k: int = 5,
    ) -> List[Tuple[Any, float]]:
        """Score every (query, chunk_text) pair with the cross-encoder."""
        if not results:
            return []

        model = self._ensure_model()
        texts = [_extract_text(chunk) for chunk, _score in results]

        # sentence-transformers CrossEncoder.predict expects a list of
        # [query, passage] pairs.
        pairs = [[query, t] for t in texts]
        ce_scores = model.predict(pairs)

        scored = [
            (chunk, float(ce_score))
            for (chunk, _orig_score), ce_score in zip(results, ce_scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.debug("Cross-encoder reranked %d results.", len(scored))
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Simple (lightweight) reranker
# ---------------------------------------------------------------------------

class SimpleReranker(BaseReranker):
    """Lightweight reranker using keyword overlap and positional signals.

    This avoids heavy model inference and is suitable for latency-critical
    paths or environments without GPU resources.

    Scoring formula
    ---------------
    ``rerank_score = overlap_weight * overlap_ratio + position_weight * position_bonus``

    * **overlap_ratio** — fraction of query tokens found in the chunk.
    * **position_bonus** — linear decay from 1.0 (first result) to 0.0
      (last result), rewarding the original retrieval ordering.

    Parameters
    ----------
    overlap_weight : float
        Weight for the keyword overlap signal.
    position_weight : float
        Weight for the original-position signal.
    """

    def __init__(
        self,
        overlap_weight: float = 0.7,
        position_weight: float = 0.3,
    ) -> None:
        self.overlap_weight = overlap_weight
        self.position_weight = position_weight

    def rerank(
        self,
        query: str,
        results: List[Tuple[Any, float]],
        top_k: int = 5,
    ) -> List[Tuple[Any, float]]:
        """Rerank using keyword overlap + positional bonus."""
        if not results:
            return []

        query_tokens = _tokenize_simple(query)
        if not query_tokens:
            # Nothing to compare — fall back to original ordering
            return results[:top_k]

        n = len(results)
        scored: List[Tuple[Any, float]] = []

        for position, (chunk, _orig_score) in enumerate(results):
            chunk_tokens = _tokenize_simple(_extract_text(chunk))
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            position_bonus = 1.0 - (position / max(n, 1))

            score = (
                self.overlap_weight * overlap
                + self.position_weight * position_bonus
            )
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        logger.debug("Simple reranker rescored %d results.", len(scored))
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(chunk: Any) -> str:
    """Get the text payload from a chunk (object or raw string)."""
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "text"):
        return chunk.text
    # Last resort: stringify
    return str(chunk)
