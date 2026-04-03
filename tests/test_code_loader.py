"""Tests for the code file document loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.code_loader import load_code


SAMPLE_PYTHON = '''"""Sample module for testing."""


def greet(name: str) -> str:
    """Return a greeting string."""
    return f"Hello, {name}!"


class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
'''


@pytest.fixture()
def python_file(tmp_path: Path) -> Path:
    """Write a sample Python file and return its path."""
    p = tmp_path / "sample.py"
    p.write_text(SAMPLE_PYTHON, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_code_returns_documents(python_file: Path) -> None:
    """load_code should return a non-empty list of Document objects."""
    docs = load_code(str(python_file))
    assert isinstance(docs, list)
    assert len(docs) >= 1


def test_load_code_extracts_content(python_file: Path) -> None:
    """The extracted text should contain the full source code."""
    docs = load_code(str(python_file))
    combined = " ".join(d.text for d in docs)
    assert "greet" in combined
    assert "Calculator" in combined
    assert "def add" in combined


def test_load_code_detects_language(python_file: Path) -> None:
    """Language metadata for a .py file should be 'python'."""
    docs = load_code(str(python_file))
    assert len(docs) >= 1
    assert docs[0].metadata["language"] == "python"


def test_load_code_metadata(python_file: Path) -> None:
    """Metadata must include modality='code', file_type, source, and line_count."""
    docs = load_code(str(python_file))
    assert len(docs) >= 1
    meta = docs[0].metadata
    assert meta["modality"] == "code"
    assert "file_type" in meta
    assert "source" in meta
    assert "line_count" in meta
    assert isinstance(meta["line_count"], int)
    assert meta["line_count"] > 0


def test_load_code_file_not_found() -> None:
    """load_code should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_code("/nonexistent/path/file.py")


def test_load_code_js_language(tmp_path: Path) -> None:
    """A .js file should be detected as 'javascript'."""
    p = tmp_path / "app.js"
    p.write_text("console.log('hello');", encoding="utf-8")
    docs = load_code(str(p))
    assert docs[0].metadata["language"] == "javascript"
