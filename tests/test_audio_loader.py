"""Tests for src/ingestion/audio_loader.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_nonexistent_raises(tmp_path: Path) -> None:
    """load_audio must raise FileNotFoundError for a missing file."""
    from src.ingestion.audio_loader import load_audio

    with pytest.raises(FileNotFoundError):
        load_audio(str(tmp_path / "no_such_file.wav"))


def test_load_audio_with_mock_whisper(tmp_path: Path) -> None:
    """load_audio returns a Document with transcribed text when _transcribe_whisper is mocked."""
    from src.ingestion.audio_loader import load_audio

    # Create a minimal dummy .wav file (just needs to exist)
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    fake_transcript = "Hello from the audio file."

    with patch(
        "src.ingestion.audio_loader._transcribe_whisper",
        return_value=fake_transcript,
    ):
        docs = load_audio(str(wav_path))

    assert len(docs) == 1
    doc = docs[0]
    assert doc.text == fake_transcript
    assert doc.doc_id  # non-empty string
    assert doc.source == str(wav_path.resolve())


def test_load_audio_metadata(tmp_path: Path) -> None:
    """Document metadata must include modality='audio' and correct file_type."""
    from src.ingestion.audio_loader import load_audio

    mp3_path = tmp_path / "clip.mp3"
    mp3_path.write_bytes(b"\xff\xfb\x90\x00" * 10)  # dummy MP3-ish bytes

    with patch(
        "src.ingestion.audio_loader._transcribe_whisper",
        return_value="Some speech here.",
    ):
        docs = load_audio(str(mp3_path))

    assert len(docs) == 1
    meta = docs[0].metadata

    assert meta.get("modality") == "audio"
    assert meta.get("file_type") == "mp3"
    assert meta.get("source") == "clip.mp3"
