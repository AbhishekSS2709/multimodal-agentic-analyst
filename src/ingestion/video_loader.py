"""Video document loader for the RAG ingestion pipeline."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore[assignment]

from src.ingestion.image_loader import LoaderResult
from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_doc_id() -> str:
    """Generate a short unique document identifier."""
    return uuid.uuid4().hex[:12]


def _detect_scene_changes(
    frames: List[Any],
    threshold: float = 30.0,
) -> List[int]:
    """Return frame indices where a scene change is detected.

    Uses per-channel histogram comparison between consecutive frames.
    Frame 0 is always included.

    Args:
        frames:    List of frames (numpy arrays in BGR format as returned by
                   ``cv2.VideoCapture.read``).
        threshold: Mean absolute difference between consecutive histograms
                   above which a scene change is declared.

    Returns:
        Sorted list of frame indices to use as keyframes.
    """
    if not frames:
        return []

    indices: List[int] = [0]

    if cv2 is None:
        # Without cv2 we cannot compare histograms – return frame 0 only.
        return indices

    def _hist(frame: Any) -> Any:
        """Compute a normalised per-channel histogram for *frame*."""
        hists = []
        for ch in range(3):
            h = cv2.calcHist([frame], [ch], None, [256], [0, 256])
            cv2.normalize(h, h)
            hists.append(h)
        return hists

    prev_hists = _hist(frames[0])

    for idx in range(1, len(frames)):
        curr_hists = _hist(frames[idx])
        diffs = [
            cv2.compareHist(prev_hists[ch], curr_hists[ch], cv2.HISTCMP_BHATTACHARYYA)
            for ch in range(3)
        ]
        mean_diff = sum(diffs) / len(diffs)
        # Scale to a 0-100 range comparable with threshold (Bhattacharyya is 0-1)
        scaled_diff = mean_diff * 100.0
        if scaled_diff >= threshold:
            indices.append(idx)
        prev_hists = curr_hists

    return sorted(set(indices))


def _transcribe_audio(video_path: str) -> str:
    """Attempt to transcribe the audio track from *video_path*.

    Delegates to :func:`src.ingestion.audio_loader._transcribe_whisper`.
    Returns an empty string on any failure.
    """
    try:
        from src.ingestion.audio_loader import _transcribe_whisper

        return _transcribe_whisper(video_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio transcription failed for %s: %s", video_path, exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_video(
    file_path: str,
    *,
    max_frames: int = 50,
    scene_threshold: float = 30.0,
    fallback_interval_sec: float = 10.0,
) -> LoaderResult:
    """Load a video file, extract keyframes, and transcribe its audio.

    Args:
        file_path:             Absolute or relative path to the video file.
        max_frames:            Maximum number of keyframes to retain.
        scene_threshold:       Sensitivity for scene-change detection (0-100).
                               Higher values mean fewer detected scene changes.
        fallback_interval_sec: When cv2 is unavailable, fall back to sampling
                               one frame every this many seconds.

    Returns:
        A :class:`LoaderResult` with:

        * ``text_documents`` – a single :class:`Document` containing the
          audio transcript (or a placeholder).
        * ``visual_assets`` – one dict per keyframe containing a PIL
          ``Image`` object and metadata.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        RuntimeError:      If cv2 is not installed.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if cv2 is None:
        raise RuntimeError(
            "cv2 (opencv-python) is required for video loading but is not installed."
        )

    source_name = path.name
    file_type = path.suffix.lstrip(".").lower() or "unknown"

    # --- Extract frames via cv2 ---------------------------------------------
    cap = cv2.VideoCapture(str(path))
    try:
        fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Decide which frame indices to read (sample evenly if video is long)
        if total_frames > 0:
            step = max(1, total_frames // (max_frames * 4))  # oversample then cull
        else:
            step = max(1, int(fps * fallback_interval_sec))

        raw_frames: List[Any] = []
        frame_indices: List[int] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                raw_frames.append(frame)
                frame_indices.append(frame_idx)
            frame_idx += 1
    finally:
        cap.release()

    # --- Scene-change detection ---------------------------------------------
    scene_indices = _detect_scene_changes(raw_frames, threshold=scene_threshold)
    # Map back to original frame positions and cap at max_frames
    selected_positions = [frame_indices[i] for i in scene_indices[:max_frames]]

    # --- Convert to PIL Images ----------------------------------------------
    from PIL import Image  # local import – PIL is a soft dependency

    visual_assets: List[Dict[str, Any]] = []
    for pos, local_idx in zip(selected_positions, scene_indices[:max_frames]):
        bgr_frame = raw_frames[local_idx]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        timestamp_sec = pos / fps if fps > 0 else 0.0

        asset_id = _generate_doc_id()
        visual_assets.append(
            {
                "image": pil_image,
                "doc_id": asset_id,
                "modality": "video_frame",
                "source": source_name,
                "frame_index": pos,
                "timestamp_sec": timestamp_sec,
                "original_path": str(path.resolve()),
            }
        )

    logger.info(
        "Extracted %d keyframe(s) from video: %s", len(visual_assets), source_name
    )

    # --- Transcribe audio ---------------------------------------------------
    transcript = _transcribe_audio(str(path))
    if not transcript:
        transcript = f"[Video: {source_name}]"

    transcript_doc = Document(
        doc_id=_generate_doc_id(),
        text=transcript,
        metadata={
            "modality": "video",
            "source": source_name,
            "file_type": file_type,
            "keyframe_count": len(visual_assets),
        },
        source=str(path.resolve()),
    )

    return LoaderResult(
        text_documents=[transcript_doc],
        visual_assets=visual_assets,
    )
