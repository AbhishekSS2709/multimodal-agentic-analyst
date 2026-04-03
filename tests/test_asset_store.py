"""Tests for AssetStore — written first (TDD).

Uses pytest's tmp_path fixture so no real ASSETS_DIR is touched.
Run with: python -m pytest tests/test_asset_store.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_red_image():
    """Return a 100x100 solid-red PIL Image."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssetStoreDirectory:
    """AssetStore creates its base directory on construction."""

    def test_store_creates_directory(self, tmp_path):
        from src.ingestion.asset_store import AssetStore

        store_dir = tmp_path / "asset_store_test"
        assert not store_dir.exists()

        AssetStore(base_dir=store_dir)

        assert store_dir.is_dir()


class TestAssetStoreImage:
    """Storing and retrieving PIL images."""

    def test_store_image(self, tmp_path):
        """store_pil_image returns an asset_id and the file exists on disk."""
        from src.ingestion.asset_store import AssetStore

        store = AssetStore(base_dir=tmp_path / "assets")
        img = _make_red_image()

        asset_id = store.store_pil_image(img, doc_id="doc_A", name="red_test")

        assert asset_id, "asset_id should be a non-empty string"
        assert store.exists(asset_id), "stored asset must be findable by asset_id"

    def test_retrieve_image(self, tmp_path):
        """get_path returns a valid file path for a stored asset."""
        from src.ingestion.asset_store import AssetStore

        store = AssetStore(base_dir=tmp_path / "assets")
        img = _make_red_image()
        asset_id = store.store_pil_image(img, doc_id="doc_A", name="red_test")

        path = store.get_path(asset_id)

        assert path is not None, "get_path should return a path string"
        assert Path(path).is_file(), "path must point to an existing file"

    def test_load_image_as_pil(self, tmp_path):
        """load_image reloads a PNG and preserves its dimensions."""
        from src.ingestion.asset_store import AssetStore

        store = AssetStore(base_dir=tmp_path / "assets")
        img = _make_red_image()
        asset_id = store.store_pil_image(img, doc_id="doc_A", name="red_test")

        reloaded = store.load_image(asset_id)

        assert reloaded is not None, "load_image should return a PIL Image"
        assert reloaded.size == (100, 100), "dimensions must be preserved"


class TestAssetStoreFile:
    """Storing arbitrary binary files via store()."""

    def test_store_file(self, tmp_path):
        """store() copies a file and returns a retrievable asset_id."""
        from src.ingestion.asset_store import AssetStore

        store = AssetStore(base_dir=tmp_path / "assets")

        # Create a dummy source file
        source_file = tmp_path / "sample.bin"
        source_file.write_bytes(b"\x00\x01\x02\x03")

        asset_id = store.store(str(source_file), doc_id="doc_B")

        assert asset_id
        assert store.exists(asset_id)
        assert Path(store.get_path(asset_id)).is_file()


class TestAssetStoreIndex:
    """Index persistence and query methods."""

    def test_list_assets_for_doc(self, tmp_path):
        """list_for_doc returns the correct assets grouped by doc_id."""
        from src.ingestion.asset_store import AssetStore

        store = AssetStore(base_dir=tmp_path / "assets")
        img = _make_red_image()

        store.store_pil_image(img, doc_id="doc_A", name="img1")
        store.store_pil_image(img, doc_id="doc_A", name="img2")
        store.store_pil_image(img, doc_id="doc_B", name="img3")

        doc_a_assets = store.list_for_doc("doc_A")
        doc_b_assets = store.list_for_doc("doc_B")

        assert len(doc_a_assets) == 2, "doc_A should have 2 assets"
        assert len(doc_b_assets) == 1, "doc_B should have 1 asset"

    def test_index_persists_across_instances(self, tmp_path):
        """A new AssetStore pointed at the same dir reloads the index."""
        from src.ingestion.asset_store import AssetStore

        store_dir = tmp_path / "assets"
        store1 = AssetStore(base_dir=store_dir)
        img = _make_red_image()
        asset_id = store1.store_pil_image(img, doc_id="doc_persist", name="img")

        # Fresh instance — must reload from JSON
        store2 = AssetStore(base_dir=store_dir)

        assert store2.exists(asset_id), "index must survive across instances"

    def test_get_metadata_returns_dict(self, tmp_path):
        """get_metadata returns the metadata dict for a stored asset."""
        from src.ingestion.asset_store import AssetStore

        store = AssetStore(base_dir=tmp_path / "assets")
        img = _make_red_image()
        meta = {"source": "unit_test", "page": 1}
        asset_id = store.store_pil_image(img, doc_id="doc_A", name="img", metadata=meta)

        result = store.get_metadata(asset_id)

        assert result is not None
        assert result.get("source") == "unit_test"
        assert result.get("page") == 1


class TestAssetStoreNonexistent:
    """Edge cases for missing assets."""

    def test_get_nonexistent_returns_none(self, tmp_path):
        """get_path and load_image return None for unknown asset_id."""
        from src.ingestion.asset_store import AssetStore

        store = AssetStore(base_dir=tmp_path / "assets")
        fake_id = "00000000-0000-0000-0000-000000000000"

        assert store.get_path(fake_id) is None
        assert store.load_image(fake_id) is None
        assert store.get_metadata(fake_id) is None
        assert not store.exists(fake_id)
