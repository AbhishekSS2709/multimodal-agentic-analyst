"""Embedding can happen on a server instead of in local memory.

The embedding model is the largest *unavoidable* local cost in this pipeline.
Every LLM call site degrades to a heuristic when no model is configured, but
retrieval cannot: something has to turn text into vectors. On an 8 GB machine
`bge-large` plus the torch runtime is ~1.5 GB, which is the difference between
the graph running comfortably and the OS paging model weights to disk.

`EMBEDDING_BASE_URL` points at any OpenAI-compatible `/v1/embeddings` endpoint.
These tests make no network call.
"""

import json
import unittest
from unittest import mock

import numpy as np

BASE = "http://10.24.6.107:8002/v1"
MODEL = "BAAI/bge-large-en-v1.5"
DIM = 8


def _response(vectors, reverse=False):
    """Fake an OpenAI embeddings payload."""
    rows = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if reverse:
        rows = list(reversed(rows))
    body = json.dumps({"object": "list", "data": rows}).encode()
    ctx = mock.MagicMock()
    ctx.__enter__.return_value.read.return_value = body
    return ctx


def _engine(**kw):
    from src.embedding.embed_chunks import EmbeddingEngine
    kw.setdefault("model_name", MODEL)
    kw.setdefault("dimension", DIM)
    kw.setdefault("base_url", BASE)
    with mock.patch.object(EmbeddingEngine, "_load_cache", lambda self: None):
        return EmbeddingEngine(**kw)


class TestRemoteEmbedding(unittest.TestCase):

    def test_no_local_model_is_loaded(self):
        """The entire point: weights and torch never enter this process."""
        engine = _engine()
        engine._ensure_model()
        self.assertIsNone(engine._model)

    def test_vectors_come_back_as_a_float32_matrix(self):
        engine = _engine()
        vectors = [[float(i)] * DIM for i in range(3)]
        with mock.patch("urllib.request.urlopen", return_value=_response(vectors)):
            out = engine._batch_encode(["a", "b", "c"])
        self.assertEqual(out.shape, (3, DIM))
        self.assertEqual(out.dtype, np.float32)

    def test_out_of_order_responses_are_realigned(self):
        """The spec does not promise order; a shuffle would mislabel every chunk."""
        engine = _engine()
        vectors = [[float(i)] * DIM for i in range(3)]
        with mock.patch("urllib.request.urlopen",
                        return_value=_response(vectors, reverse=True)):
            out = engine._batch_encode(["a", "b", "c"])
        np.testing.assert_allclose(out[0], vectors[0])
        np.testing.assert_allclose(out[2], vectors[2])

    def test_a_dimension_mismatch_is_refused_loudly(self):
        """A silently wrong dimension produces an index that cannot be searched."""
        engine = _engine(dimension=1024)
        with mock.patch("urllib.request.urlopen",
                        return_value=_response([[1.0] * DIM])):
            with self.assertRaises(ValueError) as ctx:
                engine._batch_encode(["a"])
        self.assertIn("1024", str(ctx.exception))

    def test_it_posts_the_model_and_the_batch(self):
        engine = _engine()
        with mock.patch("urllib.request.urlopen",
                        return_value=_response([[1.0] * DIM, [2.0] * DIM])) as urlopen:
            engine._batch_encode(["first", "second"])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, f"{BASE}/embeddings")
        sent = json.loads(request.data)
        self.assertEqual(sent["model"], MODEL)
        self.assertEqual(sent["input"], ["first", "second"])

    def test_batching_is_respected(self):
        engine = _engine(batch_size=2)
        with mock.patch("urllib.request.urlopen",
                        side_effect=[_response([[1.0] * DIM, [2.0] * DIM]),
                                     _response([[3.0] * DIM])]) as urlopen:
            out = engine._batch_encode(["a", "b", "c"])
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(out.shape, (3, DIM))

    def test_a_trailing_slash_does_not_double_up(self):
        self.assertEqual(_engine(base_url=BASE + "/").base_url, BASE)


class TestLocalPathUnchanged(unittest.TestCase):

    def test_without_a_base_url_the_local_model_is_still_used(self):
        engine = _engine(base_url="")
        self.assertEqual(engine.base_url, "")
        fake = mock.MagicMock()
        fake.encode.return_value = np.ones((2, DIM), dtype=np.float32)
        engine._model = fake
        out = engine._batch_encode(["a", "b"])
        self.assertTrue(fake.encode.called)
        self.assertEqual(out.shape, (2, DIM))


if __name__ == "__main__":
    unittest.main()
