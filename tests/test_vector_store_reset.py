"""A full rebuild must replace the index, not grow it.

``setup()`` calls ``store_embeddings`` on a VectorStore that has already loaded
the previous index from disk, so re-running it stored every chunk a second time:
490 chunks became 980 vectors, then 1470. Duplicates crowd the top-k and push
distinct evidence out of the results -- answer_correctness measurably dropped
(0.604 -> 0.510) on a doubled index.
"""

import unittest

import numpy as np


def _vec(seed, dim):
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class _Chunk:
    def __init__(self, i):
        self.chunk_id = f"c{i}"
        self.doc_id = "d1"
        self.text = f"chunk {i}"
        self.token_count = 2
        self.metadata = {"source": "s.txt"}


class TestVectorStoreReset(unittest.TestCase):

    def _store(self, tmp):
        from src.embedding.store_vector_db import VectorStore
        return VectorStore(dimension=8,
                           index_path=tmp / "idx.index",
                           meta_path=tmp / "idx_meta.json")

    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reset_empties_the_index(self):
        store = self._store(self.tmp)
        store.store_embeddings([_Chunk(0), _Chunk(1)], [_vec(0, 8), _vec(1, 8)])
        self.assertEqual(store.total_vectors, 2)
        store.reset()
        self.assertEqual(store.total_vectors, 0)

    def test_rebuild_after_reset_does_not_double(self):
        store = self._store(self.tmp)
        chunks = [_Chunk(i) for i in range(3)]
        vecs = [_vec(i, 8) for i in range(3)]
        store.store_embeddings(chunks, vecs)
        store.reset()
        store.store_embeddings(chunks, vecs)
        self.assertEqual(store.total_vectors, 3, "rebuild duplicated the corpus")

    def test_without_reset_it_still_appends(self):
        """Appending stays available -- incremental upload depends on it."""
        store = self._store(self.tmp)
        store.store_embeddings([_Chunk(0)], [_vec(0, 8)])
        store.store_embeddings([_Chunk(1)], [_vec(1, 8)])
        self.assertEqual(store.total_vectors, 2)

    def test_search_after_reset_returns_nothing(self):
        store = self._store(self.tmp)
        store.store_embeddings([_Chunk(0)], [_vec(0, 8)])
        store.reset()
        self.assertEqual(store.search(_vec(0, 8), top_k=3), [])


class TestVisualStoreReset(unittest.TestCase):
    """The visual index is rebuilt by the same setup pass."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reset_empties_the_visual_index(self):
        from src.embedding.visual_store import VisualVectorStore
        store = VisualVectorStore(dimension=8,
                                  index_path=self.tmp / "v.index",
                                  meta_path=self.tmp / "v_meta.json")
        store.store_embeddings([_vec(0, 8)], [{"source": "a.png"}])
        self.assertEqual(store.total_vectors, 1)
        store.reset()
        self.assertEqual(store.total_vectors, 0)


if __name__ == "__main__":
    unittest.main()
