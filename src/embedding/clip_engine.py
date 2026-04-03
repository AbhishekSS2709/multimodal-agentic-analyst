"""CLIP visual embedding engine.

Produces L2-normalised 512-dimensional embeddings for images and text queries
using OpenAI's CLIP model (``openai/clip-vit-base-patch32`` by default).

Includes an in-memory + optional on-disk cache keyed by SHA-256 hashes so
repeated calls never re-invoke the model.
"""

from __future__ import annotations

import hashlib
import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project-root bootstrap (mirrors embed_chunks.py pattern)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import CLIP_EMBEDDING_DIMENSION, CLIP_MODEL, VECTOR_DB_DIR

# Imported at module scope so tests can patch them via
# ``unittest.mock.patch("src.embedding.clip_engine.CLIPModel")``.
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disk-cache path (module-level so tests can monkeypatch it)
# ---------------------------------------------------------------------------
_CLIP_CACHE_PATH: Path = VECTOR_DB_DIR / "clip_cache.npz"


# ---------------------------------------------------------------------------
# CLIPEngine
# ---------------------------------------------------------------------------

class CLIPEngine:
    """Embed images and text into the shared CLIP latent space.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for CLIP (default from settings).
    dimension:
        Expected embedding dimension (default 512 for ViT-B/32).
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        dimension: int = CLIP_EMBEDDING_DIMENSION,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension

        # Lazy-load: model and processor are created on first use
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

        # In-memory cache: hash -> normalised (dimension,) np.ndarray
        self._cache: Dict[str, np.ndarray] = {}

        # Load on-disk cache if available
        self._load_cache()

        logger.info(
            "CLIPEngine created (model=%s, dim=%d)",
            model_name,
            dimension,
        )

    # ------------------------------------------------------------------
    # Lazy model initialisation
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the CLIP model and processor on first use."""
        if self._model is not None:
            return
        try:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()
            logger.info("Loaded CLIP model: %s", self.model_name)
        except Exception as exc:
            logger.error("Failed to load CLIP model '%s': %s", self.model_name, exc)
            raise

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Load the on-disk CLIP cache if it exists."""
        if _CLIP_CACHE_PATH.exists():
            try:
                data = np.load(str(_CLIP_CACHE_PATH), allow_pickle=True)
                keys: List[str] = data["keys"].tolist()
                vectors: np.ndarray = data["vectors"]
                for key, vec in zip(keys, vectors):
                    self._cache[key] = vec
                logger.info(
                    "Loaded %d cached CLIP embeddings from disk.", len(self._cache)
                )
            except Exception as exc:
                logger.warning("Could not load CLIP cache: %s", exc)

    def save_cache(self) -> None:
        """Persist the in-memory cache to disk as an npz archive."""
        if not self._cache:
            return
        try:
            keys = list(self._cache.keys())
            vectors = np.array(list(self._cache.values()), dtype=np.float32)
            np.savez(str(_CLIP_CACHE_PATH), keys=np.array(keys), vectors=vectors)
            logger.info("Saved %d CLIP embeddings to cache.", len(self._cache))
        except Exception as exc:
            logger.warning("Could not save CLIP cache: %s", exc)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_hash(image) -> str:
        """SHA-256 hex digest of the image's PNG bytes."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    @staticmethod
    def _text_hash(text: str) -> str:
        """SHA-256 hex digest of a UTF-8 encoded text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a 1-D vector (returns a new array)."""
        norm = np.linalg.norm(vec)
        norm = max(norm, 1e-12)  # guard against zero vectors
        return (vec / norm).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API — images
    # ------------------------------------------------------------------

    def embed_image(self, image) -> np.ndarray:
        """Embed a single PIL image into the CLIP visual space.

        Parameters
        ----------
        image:
            A ``PIL.Image.Image`` object.

        Returns
        -------
        np.ndarray
            L2-normalised embedding of shape ``(dimension,)``.
        """
        h = self._image_hash(image)
        if h in self._cache:
            return self._cache[h]

        self._ensure_model()

        inputs = self._processor(images=image, return_tensors="pt")
        features = self._model.get_image_features(**inputs)
        vec = features.detach().cpu().numpy()[0].astype(np.float32)
        vec = self._normalise(vec)

        self._cache[h] = vec
        return vec

    def embed_images(self, images: List) -> List[np.ndarray]:
        """Batch-embed a list of PIL images.

        Cache is checked per image so duplicate images are only embedded once.

        Parameters
        ----------
        images:
            List of ``PIL.Image.Image`` objects.

        Returns
        -------
        list[np.ndarray]
            One L2-normalised ``(dimension,)`` vector per input image,
            in the same order as *images*.
        """
        results: List[np.ndarray] = []
        for img in images:
            results.append(self.embed_image(img))
        return results

    # ------------------------------------------------------------------
    # Public API — text
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query into the same CLIP latent space as images.

        Parameters
        ----------
        text:
            A natural-language string (e.g. ``"a photo of a dog"``).

        Returns
        -------
        np.ndarray
            L2-normalised embedding of shape ``(dimension,)``.
        """
        h = self._text_hash(text)
        if h in self._cache:
            return self._cache[h]

        self._ensure_model()

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        features = self._model.get_text_features(**inputs)
        vec = features.detach().cpu().numpy()[0].astype(np.float32)
        vec = self._normalise(vec)

        self._cache[h] = vec
        return vec
