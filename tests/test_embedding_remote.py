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


class TestContextOverflowBackoff(unittest.TestCase):
    """A character budget cannot be exact without the server's tokenizer.

    bge accepts 512 tokens. Prose runs ~5 chars/token, but the pipe-delimited
    records in this corpus run ~2.5 -- 1716 characters came to 677 tokens. The
    server rejects the *whole request* with HTTP 400 rather than truncating, so
    one long chunk would take out the 31 batched alongside it and fail the
    entire index build. Halve and retry rather than fail on a bad guess.
    """

    def _overflow(self):
        import urllib.error
        return urllib.error.HTTPError(
            "u", 400, "Bad Request", {},
            __import__("io").BytesIO(json.dumps({"error": {
                "code": 400, "type": "exceed_context_size_error",
                "message": "input (677 tokens) is larger than the max context size (512 tokens)"
            }}).encode()))

    def test_it_retries_with_a_halved_limit(self):
        engine = _engine(max_chars=1200)
        with mock.patch("urllib.request.urlopen",
                        side_effect=[self._overflow(),
                                     _response([[1.0] * DIM])]) as urlopen:
            out = engine._batch_encode(["x" * 5000])
        self.assertEqual(urlopen.call_count, 2)
        second = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertEqual(len(second["input"][0]), 600)
        self.assertEqual(out.shape, (1, DIM))

    def test_it_gives_up_rather_than_looping_forever(self):
        engine = _engine(max_chars=1200)
        with mock.patch("urllib.request.urlopen",
                        side_effect=[self._overflow() for _ in range(6)]):
            with self.assertRaises(RuntimeError):
                engine._batch_encode(["x" * 5000])

    def test_a_non_context_error_is_not_retried(self):
        """Only overflow is recoverable by truncating; 500s are not."""
        import io
        import urllib.error
        boom = urllib.error.HTTPError("u", 500, "err", {}, io.BytesIO(b"upstream died"))
        engine = _engine()
        with mock.patch("urllib.request.urlopen", side_effect=[boom]) as urlopen:
            with self.assertRaises(RuntimeError) as ctx:
                engine._batch_encode(["a"])
        self.assertEqual(urlopen.call_count, 1)
        self.assertIn("500", str(ctx.exception))

    def test_the_default_limit_is_conservative_for_dense_text(self):
        from config.settings import EMBEDDING_MAX_CHARS
        self.assertLessEqual(EMBEDDING_MAX_CHARS, 1200)
