"""Retrieval engine — finds the most relevant chunks for a query.

Combines :class:`EmbeddingEngine` (query embedding) with :class:`VectorStore`
(FAISS search) to produce ranked results with scores and metadata.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import TOP_K

from src.chunking.chunker import Chunk
from src.embedding.embed_chunks import EmbeddingEngine
from src.embedding.store_vector_db import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Retrieve the most relevant chunks for a natural-language query.

    Parameters
    ----------
    embedding_engine : EmbeddingEngine | None
        Pre-initialised engine; one will be created if not supplied.
    vector_store : VectorStore | None
        Pre-initialised store; one will be created if not supplied.
    default_top_k : int
        Default number of results (overridable per call).
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine | None = None,
        vector_store: VectorStore | None = None,
        default_top_k: int = TOP_K,
    ) -> None:
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.vector_store = vector_store or VectorStore()
        self.default_top_k = default_top_k
        logger.info("Retriever initialised (default top_k=%d).", default_top_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> List[Tuple[Chunk, float]]:
        """Retrieve the top-K most relevant chunks for *query*.

        Parameters
        ----------
        query : str
            Natural-language question or search query.
        top_k : int | None
            Number of results; falls back to ``self.default_top_k``.

        Returns
        -------
        list[tuple[Chunk, float]]
            Each entry is ``(Chunk, similarity_score)`` sorted by descending
            score.
        """
        if not query or not query.strip():
            logger.warning("Empty query — returning no results.")
            return []

        k = top_k if top_k is not None else self.default_top_k

        # Embed the query
        query_embedding: np.ndarray = self.embedding_engine.embed_query(query)

        # Search the vector store
        raw_results: List[Dict[str, Any]] = self.vector_store.search(
            query_embedding, top_k=k
        )

        # Reconstruct Chunk objects from stored metadata
        results: List[Tuple[Chunk, float]] = []
        for hit in raw_results:
            score = hit.pop("score", 0.0)
            chunk = Chunk(
                chunk_id=hit.get("chunk_id", ""),
                text=hit.get("text", ""),
                metadata={
                    k: v
                    for k, v in hit.items()
                    if k not in ("chunk_id", "text", "doc_id", "token_count")
                },
                doc_id=hit.get("doc_id", ""),
                token_count=hit.get("token_count", 0),
            )
            results.append((chunk, score))

        logger.info(
            "Retrieved %d chunks for query: '%s' (top score=%.4f)",
            len(results),
            query[:80],
            results[0][1] if results else 0.0,
        )
        return results
