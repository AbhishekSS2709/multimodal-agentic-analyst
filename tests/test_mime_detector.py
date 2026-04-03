"""Tests for src/ingestion/mime_detector.py — written before implementation (TDD)."""

import os
import json
import tempfile

import pytest

from src.ingestion.mime_detector import detect_file_type, get_loader_for_mime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tmp(suffix: str, content: bytes) -> str:
    """Create a named temporary file with given content, return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return path


# ---------------------------------------------------------------------------
# detect_file_type tests
# ---------------------------------------------------------------------------

class TestDetectTxtFile:
    """detect_file_type correctly categorises a plain-text file."""

    def test_detect_txt_file(self):
        path = _write_tmp(".txt", b"Hello, world!\nThis is plain text.\n")
        try:
            result = detect_file_type(path)
            assert result["extension"] == ".txt"
            assert result["category"] == "text"
            assert result["loader_name"] == "txt_loader"
            assert result["blocked"] is False
            assert result["mime_type"] is not None
        finally:
            os.unlink(path)


class TestDetectCsvFile:
    """detect_file_type correctly categorises a CSV file."""

    def test_detect_csv_file(self):
        path = _write_tmp(".csv", b"id,name,value\n1,Alice,100\n2,Bob,200\n")
        try:
            result = detect_file_type(path)
            assert result["extension"] == ".csv"
            assert result["category"] == "table"
            assert result["loader_name"] == "csv_loader"
            assert result["blocked"] is False
        finally:
            os.unlink(path)


class TestDetectJsonFile:
    """detect_file_type correctly categorises a JSON file."""

    def test_detect_json_file(self):
        payload = json.dumps({"key": "value", "number": 42}).encode()
        path = _write_tmp(".json", payload)
        try:
            result = detect_file_type(path)
            assert result["extension"] == ".json"
            assert result["category"] == "structured"
            assert result["loader_name"] == "json_yaml_loader"
            assert result["blocked"] is False
        finally:
            os.unlink(path)


class TestBlockedMimeType:
    """Executable files must be flagged as blocked."""

    def test_blocked_exe_extension(self):
        # Write a minimal "MZ" DOS header so content-sniffers also see it as
        # executable; the extension alone must be enough for the block flag.
        path = _write_tmp(".exe", b"MZ\x90\x00" + b"\x00" * 60)
        try:
            result = detect_file_type(path)
            assert result["extension"] == ".exe"
            assert result["blocked"] is True
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_loader_for_mime tests
# ---------------------------------------------------------------------------

class TestGetLoaderForKnownType:
    """Known MIME types resolve to the correct loader."""

    def test_get_loader_for_known_type(self):
        loader = get_loader_for_mime("application/pdf", ".pdf")
        assert loader == "pdf_loader"

    def test_get_loader_for_csv_mime(self):
        loader = get_loader_for_mime("text/csv", ".csv")
        assert loader == "csv_loader"

    def test_get_loader_for_plain_text_mime(self):
        loader = get_loader_for_mime("text/plain", ".txt")
        assert loader == "txt_loader"


class TestGetLoaderForImage:
    """Image MIME types and extensions resolve to image_loader."""

    def test_get_loader_for_image_mime(self):
        loader = get_loader_for_mime("image/png", ".png")
        assert loader == "image_loader"

    def test_get_loader_for_jpeg_mime(self):
        loader = get_loader_for_mime("image/jpeg", ".jpg")
        assert loader == "image_loader"

    def test_get_loader_for_image_extension_fallback(self):
        # Even with an unknown MIME, extension should drive the lookup.
        loader = get_loader_for_mime("application/octet-stream", ".png")
        assert loader == "image_loader"


class TestGetLoaderForUnknownReturnsNone:
    """Completely unknown MIME type + extension returns None."""

    def test_get_loader_for_unknown_returns_none(self):
        loader = get_loader_for_mime("application/x-made-up-type", ".xyz123")
        assert loader is None
