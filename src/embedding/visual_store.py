"""FAISS visual vector store with JSON metadata mapping.

Stores normalised CLIP embeddings in a FAISS ``IndexFlatIP`` (inner-product)
index and maintains an auxiliary JSON file that maps integer index positions
to image/frame metadata for retrieval.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import CLIP_EMBEDDING_DIMENSION, VISUAL_FAISS_INDEX_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths  (parallel to the text VectorStore convention)
# ---------------------------------------------------------------------------
_INDEX_FILE = Path(str(VISUAL_FAISS_INDEX_PATH) + ".index")
_META_FILE = Path(str(VISUAL_FAISS_INDEX_PATH) + "_meta.json")


# ---------------------------------------------------------------------------
# VisualVectorStore
# ---------------------------------------------------------------------------

class VisualVectorStore:
    """Thin wrapper around a FAISS flat inner-product index for CLIP vectors.

    Parameters
    ----------
    dimension : int
        Embedding dimension (default: ``CLIP_EMBEDDING_DIMENSION`` from settings).
    index_path : Path | str | None
        Where to persist the FAISS index on disk.  Defaults to
        ``VISUAL_FAISS_INDEX_PATH.index``.
    meta_path : Path | str | None
        Where to persist the metadata JSON.  Defaults to
        ``VISUAL_FAISS_INDEX_PATH_meta.json``.
    """

    def __init__(
        self,
        dimension: int = CLIP_EMBEDDING_DIMENSION,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
    ) -> None:
        self.dimension = dimension
        self._index_path = Path(index_path) if index_path else _INDEX_FILE
        self._meta_path = Path(meta_path) if meta_path else _META_FILE

        self._index = None  # lazily initialised
        self._metadata: Dict[int, Dict[str, Any]] = {}  # position -> metadata
        self._next_id: int = 0

        # Try to load an existing index from disk
        self._load()

        logger.info(
            "VisualVectorStore ready (dim=%d, vectors=%d)",
            self.dimension,
            self._next_id,
        )

    # ------------------------------------------------------------------
    # FAISS index management
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the FAISS index if it doesn't exist yet."""
        if self._index is not None:
            return
        try:
            import faiss

            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info(
                "Created new FAISS IndexFlatIP for visuals (dim=%d).", self.dimension
            )
        except ImportError:
            logger.error(
                "faiss-cpu is not installed. Install it with: pip install faiss-cpu"
            )
            raise

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load index and metadata from disk if files exist."""
        if self._index_path.exists() and self._meta_path.exists():
            try:
                import faiss

                self._index = faiss.read_index(str(self._index_path))
                with open(self._meta_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                # JSON keys are strings; convert back to int
                self._metadata = {int(k): v for k, v in raw.items()}
                self._next_id = self._index.ntotal
                logger.info(
                    "Loaded existing visual FAISS index (%d vectors) from %s",
                    self._next_id,
                    self._index_path,
                )
            except Exception as exc:
                logger.warning("Could not load existing visual index: %s", exc)
                self._index = None
                self._metadata = {}
                self._next_id = 0

    def save(self) -> None:
        """Persist the FAISS index and metadata JSON to disk."""
        if self._index is None:
            logger.warning("No visual index to save.")
            return
        try:
            import faiss

            # Ensure parent directories exist
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))
            with open(self._meta_path, "w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, indent=2, default=str)
            logger.info(
                "Saved visual FAISS index (%d vectors) to %s",
                self._index.ntotal,
                self._index_path,
            )
        except Exception as exc:
            logger.error("Failed to save visual vector store: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_embeddings(
        self,
        vectors: List[np.ndarray],
        metadata_list: List[dict],
    ) -> None:
        """Add *vectors* and their *metadata_list* to the index.

        Parameters
        ----------
        vectors : list[np.ndarray]
            Normalised CLIP embedding vectors.
        metadata_list : list[dict]
            Corresponding metadata dicts (one per vector).

        Raises
        ------
        ValueError
            If the lengths of *vectors* and *metadata_list* differ, or if the
            vector dimension does not match the store dimension.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata_list ({len(metadata_list)}) "
                "must have the same length."
            )
        if not vectors:
            logger.warning("store_embeddings called with empty input.")
            return

        self._ensure_index()

        matrix = np.vstack(vectors).astype(np.float32)
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {matrix.shape[1]}."
            )

        start_id = self._next_id
        self._index.add(matrix)

        for i, meta in enumerate(metadata_list):
            self._metadata[start_id + i] = dict(meta)

        self._next_id = self._index.ntotal
        logger.info(
            "Stored %d visual embeddings (total vectors: %d).",
            len(vectors),
            self._next_id,
        )

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the index for the *top_k* nearest neighbours.

        Parameters
        ----------
        query_vector : np.ndarray
            Normalised CLIP query embedding of shape ``(dimension,)``.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict contains ``score`` (float) plus all stored metadata keys.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Search called on an empty visual index.")
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for missing results
            meta = self._metadata.get(int(idx), {})
            results.append({"score": float(score), **meta})

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors currently in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal
