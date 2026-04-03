"""Tests for src/ingestion/video_loader.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(r: int, g: int, b: int, size: int = 64) -> np.ndarray:
    """Return a solid-colour BGR frame as a numpy array."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :, 0] = b  # OpenCV uses BGR
    frame[:, :, 1] = g
    frame[:, :, 2] = r
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_nonexistent_raises(tmp_path: Path) -> None:
    """load_video must raise FileNotFoundError for a missing file."""
    from src.ingestion.video_loader import load_video

    with pytest.raises(FileNotFoundError):
        load_video(str(tmp_path / "no_such_video.mp4"))


def test_detect_scene_changes_returns_indices() -> None:
    """Scene-change detector finds a big colour shift mid-sequence."""
    cv2_mock = pytest.importorskip("cv2", reason="cv2 not installed")

    from src.ingestion.video_loader import _detect_scene_changes

    # 10 identical red frames, then 5 identical blue frames
    red_frames = [_make_frame(200, 0, 0) for _ in range(5)]
    blue_frames = [_make_frame(0, 0, 200) for _ in range(5)]
    frames = red_frames + blue_frames

    # Use a low threshold so the big colour jump is definitely caught
    indices = _detect_scene_changes(frames, threshold=1.0)

    assert 0 in indices, "Frame 0 must always be included"
    # The scene change happens at frame 5 (first blue frame)
    assert 5 in indices, "Frame 5 (colour switch) must be detected as a scene change"


def test_load_video_with_mock(tmp_path: Path) -> None:
    """load_video returns a LoaderResult with text_documents and visual_assets (mocked cv2)."""
    # Create a dummy file so FileNotFoundError is not raised
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00" * 64)

    # Build a minimal mock for cv2.VideoCapture
    red_frame = _make_frame(200, 0, 0)
    green_frame = _make_frame(0, 200, 0)

    read_returns = [
        (True, red_frame),
        (True, green_frame),
        (False, None),  # signals end of video
    ]
    read_iter = iter(read_returns)

    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        0: 25.0,   # CAP_PROP_POS_MSEC – ignored
        5: 25.0,   # CAP_PROP_FPS
        7: 2.0,    # CAP_PROP_FRAME_COUNT
    }.get(prop, 0.0)
    mock_cap.read.side_effect = lambda: next(read_iter)
    mock_cap.release = MagicMock()

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.COLOR_BGR2RGB = 4  # arbitrary constant
    mock_cv2.cvtColor.side_effect = lambda frame, _code: frame  # no-op colour conv

    import importlib
    import src.ingestion.video_loader as vmod

    # Reload so the module-level cv2 reference is replaced by our mock
    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        importlib.reload(vmod)

    # Now patch helpers on the freshly-reloaded module
    with patch.object(vmod, "cv2", mock_cv2), \
         patch.object(vmod, "_detect_scene_changes", return_value=[0, 1]), \
         patch.object(vmod, "_transcribe_audio", return_value="Test transcript"):

        result = vmod.load_video(str(video_path))

    assert len(result.text_documents) == 1
    doc = result.text_documents[0]
    assert "transcript" in doc.text.lower() or doc.text  # text is set
    assert doc.metadata.get("modality") == "video"

    # visual_assets: one per keyframe selected
    assert isinstance(result.visual_assets, list)
    for asset in result.visual_assets:
        assert "image" in asset
        assert asset.get("modality") == "video_frame"
        assert asset.get("source") == "clip.mp4"
