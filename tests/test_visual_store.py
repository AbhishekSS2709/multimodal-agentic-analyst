"""Tests for VisualVectorStore.

Follows TDD: tests are written before implementation.
Uses tmp_path so no real disk paths are polluted.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_unit_vector(dim: int = 512, seed: int | None = None) -> np.ndarray:
    """Return a normalised float32 vector of the given dimension."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVisualVectorStore:

    def test_store_creates_empty(self, tmp_path):
        """A freshly created store with no persisted files has 0 vectors."""
        from src.embedding.visual_store import VisualVectorStore

        index_path = tmp_path / "vis.index"
        meta_path = tmp_path / "vis_meta.json"

        store = VisualVectorStore(
            dimension=512,
            index_path=index_path,
            meta_path=meta_path,
        )

        assert store.total_vectors == 0

    def test_store_and_search(self, tmp_path):
        """Storing 3 vectors and searching with the first returns it as the top hit."""
        from src.embedding.visual_store import VisualVectorStore

        index_path = tmp_path / "vis.index"
        meta_path = tmp_path / "vis_meta.json"

        store = VisualVectorStore(
            dimension=512,
            index_path=index_path,
            meta_path=meta_path,
        )

        vectors = [_random_unit_vector(512, seed=i) for i in range(3)]
        metadata_list = [
            {"source": "img_0.png", "page": 1},
            {"source": "img_1.png", "page": 2},
            {"source": "img_2.png", "page": 3},
        ]

        store.store_embeddings(vectors, metadata_list)
        assert store.total_vectors == 3

        results = store.search(vectors[0], top_k=3)

        assert len(results) > 0
        # The best match for vectors[0] must be vectors[0] itself
        top = results[0]
        assert top["source"] == "img_0.png"
        assert "score" in top
        # Inner-product of a unit vector with itself == 1.0
        assert top["score"] == pytest.approx(1.0, abs=1e-5)

    def test_save_and_load(self, tmp_path):
        """Data persists across save/load cycles."""
        from src.embedding.visual_store import VisualVectorStore

        index_path = tmp_path / "vis.index"
        meta_path = tmp_path / "vis_meta.json"

        store = VisualVectorStore(dimension=512, index_path=index_path, meta_path=meta_path)

        vectors = [_random_unit_vector(512, seed=42 + i) for i in range(2)]
        metadata_list = [
            {"source": "doc_A.pdf", "frame": 0},
            {"source": "doc_B.pdf", "frame": 1},
        ]
        store.store_embeddings(vectors, metadata_list)
        store.save()

        # Create a brand-new instance pointing at the same files
        store2 = VisualVectorStore(dimension=512, index_path=index_path, meta_path=meta_path)

        assert store2.total_vectors == 2

        results = store2.search(vectors[0], top_k=1)
        assert len(results) == 1
        assert results[0]["source"] == "doc_A.pdf"

    def test_search_empty_store(self, tmp_path):
        """Searching an empty store returns an empty list (no crash)."""
        from src.embedding.visual_store import VisualVectorStore

        store = VisualVectorStore(
            dimension=512,
            index_path=tmp_path / "vis.index",
            meta_path=tmp_path / "vis_meta.json",
        )

        query = _random_unit_vector(512, seed=99)
        results = store.search(query, top_k=5)

        assert results == []
