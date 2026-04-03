"""Tests for CLIPEngine — TDD: written before the implementation.

Mock CLIPModel and CLIPProcessor so no actual GPU/model download is needed.
Run with: python -m pytest tests/test_clip_engine.py -v
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width: int = 32, height: int = 32, colour=(128, 64, 32)):
    """Return a small solid-colour PIL Image without hitting the model."""
    from PIL import Image
    return Image.new("RGB", (width, height), color=colour)


def _make_random_tensor(shape=(1, 512)):
    """Return a torch-like object with .detach().cpu().numpy() returning a random array."""
    import numpy as np

    arr = np.random.randn(*shape).astype(np.float32)

    # Simulate torch.Tensor interface used in CLIPEngine
    # Chain: features.detach().cpu().numpy()[0]
    mock_tensor = MagicMock()
    mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = arr
    return mock_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_clip_classes():
    """Patch CLIPModel and CLIPProcessor at the module level before import.

    The fixture yields (CLIPEngine, model_instance, processor_instance) where
    model_instance is the concrete MagicMock that _ensure_model stores as
    self._model (returned by CLIPModel.from_pretrained).
    """
    # Create concrete model/processor instances first so we can configure them
    # and refer to them in assertions.
    mock_model_instance = MagicMock(name="clip_model_instance")
    mock_processor_instance = MagicMock(name="clip_processor_instance")

    # get_image_features / get_text_features return (1, 512) mock tensors
    mock_model_instance.get_image_features.side_effect = (
        lambda **kw: _make_random_tensor((1, 512))
    )
    mock_model_instance.get_text_features.side_effect = (
        lambda **kw: _make_random_tensor((1, 512))
    )

    # Build class-level mocks whose from_pretrained() returns our instances
    mock_model_cls = MagicMock(name="CLIPModel")
    mock_model_cls.from_pretrained.return_value = mock_model_instance

    mock_processor_cls = MagicMock(name="CLIPProcessor")
    mock_processor_cls.from_pretrained.return_value = mock_processor_instance

    # Ensure the module is imported first (so the attribute exists to patch),
    # then patch it in place.  Do NOT re-import inside the context — that would
    # re-execute the top-level `from transformers import ...` and overwrite the
    # patches.
    import src.embedding.clip_engine  # noqa: F401  (ensure module is loaded)
    from src.embedding.clip_engine import CLIPEngine

    with patch("src.embedding.clip_engine.CLIPModel", mock_model_cls), \
         patch("src.embedding.clip_engine.CLIPProcessor", mock_processor_cls):
        yield CLIPEngine, mock_model_instance, mock_processor_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCLIPEngineInit:
    """CLIPEngine initialises correctly."""

    def test_engine_initializes(self, mock_clip_classes, tmp_path):
        CLIPEngine, _, _ = mock_clip_classes
        engine = CLIPEngine()
        assert engine.dimension == 512, "Default dimension must be 512"
        assert engine._cache == {}, "Cache should start empty"

    def test_engine_custom_dimension(self, mock_clip_classes, tmp_path):
        CLIPEngine, _, _ = mock_clip_classes
        engine = CLIPEngine(dimension=768)
        assert engine.dimension == 768


class TestEmbedImage:
    """embed_image returns a normalised (512,) vector."""

    def test_embed_image_returns_correct_shape(self, mock_clip_classes):
        CLIPEngine, mock_model, mock_processor = mock_clip_classes
        engine = CLIPEngine()
        img = _make_pil_image()

        vec = engine.embed_image(img)

        assert isinstance(vec, np.ndarray), "Result must be np.ndarray"
        assert vec.shape == (512,), f"Expected shape (512,) got {vec.shape}"

    def test_embed_image_is_l2_normalised(self, mock_clip_classes):
        CLIPEngine, _, _ = mock_clip_classes
        engine = CLIPEngine()
        img = _make_pil_image()

        vec = engine.embed_image(img)
        norm = float(np.linalg.norm(vec))

        assert abs(norm - 1.0) < 1e-5, f"Vector should be L2-normalised, got norm={norm}"

    def test_embed_image_cached_on_second_call(self, mock_clip_classes):
        """Second call with the same image must not re-invoke the model."""
        CLIPEngine, mock_model, _ = mock_clip_classes
        engine = CLIPEngine()
        img = _make_pil_image()

        engine.embed_image(img)
        engine.embed_image(img)

        # get_image_features should only be called once despite two embed calls
        assert mock_model.get_image_features.call_count == 1


class TestEmbedText:
    """embed_text returns a normalised (512,) vector in image space."""

    def test_embed_text_returns_correct_shape(self, mock_clip_classes):
        CLIPEngine, _, _ = mock_clip_classes
        engine = CLIPEngine()

        vec = engine.embed_text("a photo of a cat")

        assert isinstance(vec, np.ndarray)
        assert vec.shape == (512,), f"Expected shape (512,) got {vec.shape}"

    def test_embed_text_is_l2_normalised(self, mock_clip_classes):
        CLIPEngine, _, _ = mock_clip_classes
        engine = CLIPEngine()

        vec = engine.embed_text("hello world")
        norm = float(np.linalg.norm(vec))

        assert abs(norm - 1.0) < 1e-5, f"Vector should be L2-normalised, got norm={norm}"

    def test_embed_text_cached_on_second_call(self, mock_clip_classes):
        CLIPEngine, mock_model, _ = mock_clip_classes
        engine = CLIPEngine()

        engine.embed_text("repeat query")
        engine.embed_text("repeat query")

        assert mock_model.get_text_features.call_count == 1


class TestEmbedImagesBatch:
    """embed_images processes a list of images and returns matching embeddings."""

    def test_embed_images_batch(self, mock_clip_classes):
        """3 distinct images should produce 3 result vectors of shape (512,)."""
        CLIPEngine, _, _ = mock_clip_classes
        engine = CLIPEngine()

        images = [
            _make_pil_image(colour=(255, 0, 0)),
            _make_pil_image(colour=(0, 255, 0)),
            _make_pil_image(colour=(0, 0, 255)),
        ]

        results = engine.embed_images(images)

        assert len(results) == 3, "Must return one embedding per image"
        for i, vec in enumerate(results):
            assert isinstance(vec, np.ndarray), f"Result[{i}] must be np.ndarray"
            assert vec.shape == (512,), f"Result[{i}] shape should be (512,), got {vec.shape}"

    def test_embed_images_batch_all_normalised(self, mock_clip_classes):
        CLIPEngine, _, _ = mock_clip_classes
        engine = CLIPEngine()

        images = [_make_pil_image(colour=(i * 80, 0, 0)) for i in range(1, 4)]
        results = engine.embed_images(images)

        for i, vec in enumerate(results):
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 1e-5, f"Result[{i}] norm={norm}, expected 1.0"

    def test_embed_images_uses_cache_for_duplicates(self, mock_clip_classes):
        """Duplicate images in the batch should only be embedded once."""
        CLIPEngine, mock_model, _ = mock_clip_classes
        engine = CLIPEngine()

        img = _make_pil_image(colour=(10, 20, 30))
        results = engine.embed_images([img, img, img])

        assert len(results) == 3
        # Model called only once because duplicates are served from cache
        assert mock_model.get_image_features.call_count == 1


class TestStaticHelpers:
    """Static helper methods work correctly."""

    def test_image_hash_is_deterministic(self, mock_clip_classes):
        CLIPEngine, _, _ = mock_clip_classes
        img = _make_pil_image()
        h1 = CLIPEngine._image_hash(img)
        h2 = CLIPEngine._image_hash(img)
        assert h1 == h2

    def test_text_hash_is_deterministic(self, mock_clip_classes):
        CLIPEngine, _, _ = mock_clip_classes
        h1 = CLIPEngine._text_hash("hello")
        h2 = CLIPEngine._text_hash("hello")
        assert h1 == h2

    def test_text_hash_differs_for_different_inputs(self, mock_clip_classes):
        CLIPEngine, _, _ = mock_clip_classes
        assert CLIPEngine._text_hash("foo") != CLIPEngine._text_hash("bar")

    def test_normalise_unit_vector(self, mock_clip_classes):
        CLIPEngine, _, _ = mock_clip_classes
        vec = np.array([3.0, 4.0], dtype=np.float32)
        out = CLIPEngine._normalise(vec)
        assert abs(np.linalg.norm(out) - 1.0) < 1e-6

    def test_normalise_zero_vector_no_crash(self, mock_clip_classes):
        """Zero vector should not raise; norm is clamped."""
        CLIPEngine, _, _ = mock_clip_classes
        vec = np.zeros(512, dtype=np.float32)
        out = CLIPEngine._normalise(vec)
        assert out.shape == (512,)


class TestSaveCache:
    """save_cache persists data to disk."""

    def test_save_cache_creates_file(self, mock_clip_classes, tmp_path, monkeypatch):
        CLIPEngine, _, _ = mock_clip_classes

        # Redirect cache path to tmp_path
        import src.embedding.clip_engine as clip_mod
        monkeypatch.setattr(clip_mod, "_CLIP_CACHE_PATH", tmp_path / "clip_cache.npz")

        engine = CLIPEngine()
        engine._cache["abc"] = np.ones(512, dtype=np.float32)
        engine.save_cache()

        assert (tmp_path / "clip_cache.npz").exists(), "cache file should be created"

    def test_save_cache_empty_does_not_crash(self, mock_clip_classes, tmp_path, monkeypatch):
        CLIPEngine, _, _ = mock_clip_classes

        import src.embedding.clip_engine as clip_mod
        monkeypatch.setattr(clip_mod, "_CLIP_CACHE_PATH", tmp_path / "clip_cache.npz")

        engine = CLIPEngine()
        engine.save_cache()  # empty cache — should not raise
