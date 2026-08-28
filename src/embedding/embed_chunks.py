"""Embedding engine — produces normalised vector embeddings for chunks and queries.

Uses sentence-transformers with the model specified in ``config.settings``.
Includes an in-memory + optional on-disk cache to avoid redundant computation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
)

from src.chunking.chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache path
# ---------------------------------------------------------------------------
_CACHE_PATH = VECTOR_DB_DIR / "embedding_cache.npz"


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Compute, normalise, and cache embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default from settings).
    dimension : int
        Expected embedding dimension (default from settings).
    batch_size : int
        Number of texts embedded per forward pass.
    device : str | None
        Torch device string; ``None`` for auto-detect.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIMENSION,
        batch_size: int = 32,
        device: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._model = None
        self._device = device
        # When set, embedding happens on a server and nothing is loaded here.
        self.base_url = (
            EMBEDDING_BASE_URL if base_url is None else base_url
        ).strip().rstrip("/")
        self.timeout = timeout

        # In-memory cache: content hash -> embedding vector
        self._cache: Dict[str, np.ndarray] = {}

        # Try loading persistent cache
        self._load_cache()

        logger.info(
            "EmbeddingEngine created (model=%s, dim=%d, batch=%d)",
            model_name,
            dimension,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the sentence-transformer model on first use.

        A no-op when embedding remotely -- that is the whole point of the
        remote path, since the weights and the torch runtime are what cost
        memory locally.
        """
        if self.base_url or self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
            logger.info("Loaded sentence-transformer model: %s", self.model_name)
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self.model_name, exc)
            raise

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        """SHA-256 hex digest of the text (used as cache key)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Load the on-disk embedding cache if it exists."""
        if _CACHE_PATH.exists():
            try:
                data = np.load(str(_CACHE_PATH), allow_pickle=True)
                keys = data["keys"].tolist()
                vectors = data["vectors"]
                for key, vec in zip(keys, vectors):
                    self._cache[key] = vec
                logger.info("Loaded %d cached embeddings from disk.", len(self._cache))
            except Exception as exc:
                logger.warning("Could not load embedding cache: %s", exc)

    def save_cache(self) -> None:
        """Persist the in-memory cache to disk."""
        if not self._cache:
            return
        try:
            keys = list(self._cache.keys())
            vectors = np.array(list(self._cache.values()), dtype=np.float32)
            np.savez(str(_CACHE_PATH), keys=np.array(keys), vectors=vectors)
            logger.info("Saved %d embeddings to cache on disk.", len(self._cache))
        except Exception as exc:
            logger.warning("Could not save embedding cache: %s", exc)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise a batch of row vectors in-place (for cosine / IP search)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        return vectors / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self, chunks: List[Chunk]
    ) -> List[Tuple[Chunk, np.ndarray]]:
        """Embed a list of :class:`Chunk` objects.

        Returns
        -------
        list[tuple[Chunk, np.ndarray]]
            Each entry is ``(chunk, normalised_embedding)``.
        """
        self._ensure_model()

        results: List[Tuple[Chunk, np.ndarray]] = []
        to_embed_texts: List[str] = []
        to_embed_indices: List[int] = []

        for idx, chunk in enumerate(chunks):
            h = self._content_hash(chunk.text)
            if h in self._cache:
                results.append((chunk, self._cache[h]))
            else:
                to_embed_texts.append(chunk.text)
                to_embed_indices.append(idx)
                results.append((chunk, None))  # placeholder

        if to_embed_texts:
            logger.info(
                "Embedding %d new chunks (%d from cache).",
                len(to_embed_texts),
                len(chunks) - len(to_embed_texts),
            )
            raw = self._batch_encode(to_embed_texts)
            normalised = self._normalise(raw)

            for i, vec_idx in enumerate(to_embed_indices):
                vec = normalised[i]
                chunk = chunks[vec_idx]
                h = self._content_hash(chunk.text)
                self._cache[h] = vec
                results[vec_idx] = (chunk, vec)
        else:
            logger.info("All %d chunks served from cache.", len(chunks))

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns
        -------
        np.ndarray
            Normalised embedding vector of shape ``(dimension,)``.
        """
        self._ensure_model()

        h = self._content_hash(query)
        if h in self._cache:
            return self._cache[h]

        raw = self._batch_encode([query])
        normalised = self._normalise(raw)[0]
        self._cache[h] = normalised
        return normalised

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _remote_encode(self, texts: List[str]) -> np.ndarray:
        """Encode *texts* via an OpenAI-compatible ``/v1/embeddings`` endpoint."""
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = json.dumps(
                {"model": self.model_name, "input": batch}
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
            # The spec does not promise response order, and a reordered batch
            # would silently attach every vector to the wrong chunk.
            rows = sorted(body["data"], key=lambda row: row.get("index", 0))
            all_embeddings.append(
                np.asarray([row["embedding"] for row in rows], dtype=np.float32)
            )
        matrix = np.vstack(all_embeddings).astype(np.float32)
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"{self.base_url} returned {matrix.shape[1]}-dim vectors but "
                f"EMBEDDING_DIMENSION is {self.dimension}; the index would be "
                f"unusable."
            )
        return matrix

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode *texts* in batches and return the raw numpy matrix."""
        if self.base_url:
            return self._remote_encode(texts)
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            emb = self._model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # we normalise ourselves
            )
            all_embeddings.append(emb)
        return np.vstack(all_embeddings).astype(np.float32)
