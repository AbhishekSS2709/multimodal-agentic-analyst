"""Tests for src/ingestion/image_loader.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_green_png(path: Path, size: tuple[int, int] = (200, 200)) -> Path:
    """Create a solid-green PNG at *path* and return it."""
    img = Image.new("RGB", size, color=(0, 200, 0))
    img.save(str(path), format="PNG")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_image_returns_documents(tmp_path: Path) -> None:
    """LoaderResult must contain exactly one Document for a valid image."""
    from src.ingestion.image_loader import load_image

    img_path = _make_green_png(tmp_path / "green.png")
    result = load_image(str(img_path))

    assert len(result.text_documents) == 1
    doc = result.text_documents[0]
    # doc_id must be a non-empty string
    assert doc.doc_id
    # text must be a string (OCR result or placeholder)
    assert isinstance(doc.text, str)
    assert len(doc.text) > 0


def test_load_image_visual_asset_exists(tmp_path: Path) -> None:
    """LoaderResult must contain exactly one visual_asset with expected keys."""
    from src.ingestion.image_loader import load_image

    img_path = _make_green_png(tmp_path / "green.png")
    result = load_image(str(img_path))

    assert len(result.visual_assets) == 1
    asset = result.visual_assets[0]

    # Required keys
    assert "image" in asset, "visual_asset must contain 'image' (PIL.Image)"
    assert isinstance(asset["image"], Image.Image)
    assert asset.get("modality") == "image"


def test_load_image_metadata(tmp_path: Path) -> None:
    """Document metadata must include modality and source fields."""
    from src.ingestion.image_loader import load_image

    img_path = _make_green_png(tmp_path / "sample.png")
    result = load_image(str(img_path))

    doc = result.text_documents[0]
    meta = doc.metadata

    assert meta.get("modality") == "image"
    assert meta.get("source") == "sample.png"
    # image_size should be a tuple/list with two positive integers
    size = meta.get("image_size")
    assert size is not None
    assert len(size) == 2
    assert size[0] > 0 and size[1] > 0
    # has_ocr_text must be a bool
    assert isinstance(meta.get("has_ocr_text"), bool)
    # file_type must be present
    assert meta.get("file_type")


def test_load_nonexistent_raises(tmp_path: Path) -> None:
    """load_image must raise FileNotFoundError for a missing file."""
    from src.ingestion.image_loader import load_image

    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "no_such_file.png"))
