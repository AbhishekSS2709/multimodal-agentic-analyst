"""Tests for the HTML document loader."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.ingestion.html_loader import load_html


SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page Title</title>
    <style>
        body { color: red; }
        h1 { font-size: 2em; }
    </style>
</head>
<body>
    <header><nav>Home | About</nav></header>
    <h1>Main Heading</h1>
    <p>This is a <b>bold</b> paragraph with content.</p>
    <ul>
        <li>Item one</li>
        <li>Item two</li>
    </ul>
    <footer>Footer text</footer>
    <script>
        alert("This should not appear");
        var x = 42;
    </script>
</body>
</html>
"""


@pytest.fixture()
def html_file(tmp_path: Path) -> Path:
    """Write a sample HTML file and return its path."""
    p = tmp_path / "sample.html"
    p.write_text(SAMPLE_HTML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_html_returns_documents(html_file: Path) -> None:
    """load_html should return a non-empty list of Document objects."""
    docs = load_html(str(html_file))
    assert isinstance(docs, list)
    assert len(docs) >= 1


def test_load_html_extracts_text(html_file: Path) -> None:
    """The extracted text should contain visible content from the page."""
    docs = load_html(str(html_file))
    combined = " ".join(d.text for d in docs)
    assert "Main Heading" in combined
    assert "bold" in combined
    assert "Item one" in combined
    assert "Item two" in combined


def test_load_html_strips_scripts(html_file: Path) -> None:
    """Script and style tag contents must not appear in the output."""
    docs = load_html(str(html_file))
    combined = " ".join(d.text for d in docs)
    assert "alert" not in combined
    assert "font-size" not in combined
    assert "color: red" not in combined
    assert "var x" not in combined


def test_load_html_metadata(html_file: Path) -> None:
    """Metadata must include source, file_type='html', and the page title."""
    docs = load_html(str(html_file))
    assert len(docs) >= 1
    meta = docs[0].metadata
    assert meta["file_type"] == "html"
    assert meta["title"] == "Test Page Title"
    assert "source" in meta


def test_load_html_file_not_found() -> None:
    """load_html should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_html("/nonexistent/path/file.html")
