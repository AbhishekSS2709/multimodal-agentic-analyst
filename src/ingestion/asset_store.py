"""Asset storage manager for images, video frames, and other binary content.

Provides a simple content-addressable store backed by the local filesystem
with a JSON index for fast lookup by asset_id or doc_id.

Usage::

    from src.ingestion.asset_store import AssetStore

    store = AssetStore()                          # uses ASSETS_DIR from config
    asset_id = store.store_pil_image(pil_img, doc_id="doc_42")
    img = store.load_image(asset_id)
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy PIL import so the module loads even without Pillow installed at the
# top level (ImportError surfaces only when image methods are called).
# ---------------------------------------------------------------------------
try:
    from PIL import Image as _PILImage  # type: ignore
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

_INDEX_FILENAME = "_index.json"


class AssetStore:
    """File-system-backed store for binary assets with a JSON index.

    Each asset is stored under *base_dir* with a UUID-based filename.
    The index is a flat JSON dict mapping ``asset_id -> entry`` where each
    entry holds ``doc_id``, ``file_path``, and optional ``metadata``.

    Args:
        base_dir: Root directory for stored files.  Defaults to
            ``ASSETS_DIR`` from ``config.settings``.
    """

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        if base_dir is None:
            from config.settings import ASSETS_DIR
            base_dir = ASSETS_DIR

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self.base_dir / _INDEX_FILENAME
        self._index: Dict[str, dict] = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        file_path: str | Path,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Copy *file_path* into the store and return a new ``asset_id``.

        Args:
            file_path: Source file to copy.
            doc_id: Identifier of the parent document.
            metadata: Optional dict of extra metadata to persist.

        Returns:
            A UUID string that uniquely identifies the stored asset.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
        """
        src = Path(file_path)
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        asset_id = str(uuid.uuid4())
        suffix = src.suffix or ""
        dest = self.base_dir / f"{asset_id}{suffix}"

        shutil.copy2(str(src), str(dest))
        logger.debug("Stored asset %s <- %s", asset_id, src.name)

        self._index[asset_id] = {
            "asset_id": asset_id,
            "doc_id": doc_id,
            "file_path": str(dest),
            "original_name": src.name,
            "metadata": metadata or {},
        }
        self._save_index()
        return asset_id

    def store_pil_image(
        self,
        image,
        doc_id: str,
        name: str = "image",
        metadata: Optional[dict] = None,
    ) -> str:
        """Save a PIL Image as PNG and return the new ``asset_id``.

        Args:
            image: A ``PIL.Image.Image`` instance.
            doc_id: Identifier of the parent document.
            name: Human-readable label stored in the index entry.
            metadata: Optional extra metadata.

        Returns:
            A UUID string that uniquely identifies the stored asset.
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required for store_pil_image(). Install it with: pip install Pillow")

        asset_id = str(uuid.uuid4())
        dest = self.base_dir / f"{asset_id}.png"

        image.save(str(dest), format="PNG")
        logger.debug("Stored PIL image asset %s (%s)", asset_id, name)

        self._index[asset_id] = {
            "asset_id": asset_id,
            "doc_id": doc_id,
            "file_path": str(dest),
            "original_name": f"{name}.png",
            "metadata": metadata or {},
        }
        self._save_index()
        return asset_id

    def exists(self, asset_id: str) -> bool:
        """Return ``True`` if *asset_id* is in the index."""
        return asset_id in self._index

    def get_path(self, asset_id: str) -> Optional[str]:
        """Return the full filesystem path for *asset_id*, or ``None``."""
        entry = self._index.get(asset_id)
        if entry is None:
            return None
        return entry["file_path"]

    def load_image(self, asset_id: str):
        """Load and return the asset as a ``PIL.Image``, or ``None``.

        Returns ``None`` if *asset_id* is unknown or Pillow is unavailable.
        """
        path = self.get_path(asset_id)
        if path is None:
            return None
        if not PIL_AVAILABLE:
            logger.warning("Pillow not installed; cannot load image for asset %s", asset_id)
            return None
        try:
            return _PILImage.open(path)
        except Exception as exc:
            logger.error("Failed to load image for asset %s: %s", asset_id, exc)
            return None

    def list_for_doc(self, doc_id: str) -> List[dict]:
        """Return all index entries associated with *doc_id*.

        Each entry is a copy of the raw dict stored in the index.
        """
        return [
            dict(entry)
            for entry in self._index.values()
            if entry.get("doc_id") == doc_id
        ]

    def get_metadata(self, asset_id: str) -> Optional[dict]:
        """Return the metadata dict for *asset_id*, or ``None``."""
        entry = self._index.get(asset_id)
        if entry is None:
            return None
        return dict(entry.get("metadata", {}))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Load the JSON index from disk, or start with an empty dict."""
        if self._index_path.is_file():
            try:
                with open(self._index_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self._index = data
                    logger.debug(
                        "Loaded asset index with %d entries from %s",
                        len(self._index),
                        self._index_path,
                    )
                else:
                    logger.warning("Asset index at %s is not a dict; resetting.", self._index_path)
                    self._index = {}
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read asset index (%s); starting fresh.", exc)
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Persist the in-memory index to ``_index.json``."""
        try:
            with open(self._index_path, "w", encoding="utf-8") as fh:
                json.dump(self._index, fh, indent=2)
        except OSError as exc:
            logger.error("Failed to save asset index: %s", exc)
