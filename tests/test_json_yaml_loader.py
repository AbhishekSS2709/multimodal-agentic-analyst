"""Tests for the JSON/YAML document loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.json_yaml_loader import load_json_yaml


SAMPLE_JSON = {
    "name": "enterprise-rag",
    "version": "1.0.0",
    "config": {
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 512,
        "tags": ["nlp", "retrieval", "multimodal"],
    },
}

SAMPLE_YAML_TEXT = """\
project: enterprise-rag
author: test-user
settings:
  debug: true
  max_results: 10
  allowed_types:
    - pdf
    - html
    - code
"""


@pytest.fixture()
def json_file(tmp_path: Path) -> Path:
    """Write a sample nested JSON file and return its path."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(SAMPLE_JSON, indent=2), encoding="utf-8")
    return p


@pytest.fixture()
def yaml_file(tmp_path: Path) -> Path:
    """Write a sample YAML file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE_YAML_TEXT, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# JSON tests
# ---------------------------------------------------------------------------


def test_load_json_returns_documents(json_file: Path) -> None:
    """load_json_yaml on a .json file should return a non-empty list."""
    docs = load_json_yaml(str(json_file))
    assert isinstance(docs, list)
    assert len(docs) >= 1


def test_load_json_extracts_content(json_file: Path) -> None:
    """Flattened text should contain key values from the nested JSON."""
    docs = load_json_yaml(str(json_file))
    combined = " ".join(d.text for d in docs)
    assert "enterprise-rag" in combined
    assert "1.0.0" in combined
    assert "512" in combined
    # list items should be represented
    assert "nlp" in combined


def test_load_json_metadata(json_file: Path) -> None:
    """Metadata must include modality='structured', file_type='json', and source."""
    docs = load_json_yaml(str(json_file))
    assert len(docs) >= 1
    meta = docs[0].metadata
    assert meta["modality"] == "structured"
    assert meta["file_type"] == "json"
    assert "source" in meta


# ---------------------------------------------------------------------------
# YAML tests
# ---------------------------------------------------------------------------


def test_load_yaml_returns_documents(yaml_file: Path) -> None:
    """load_json_yaml on a .yaml file should return a non-empty list."""
    docs = load_json_yaml(str(yaml_file))
    assert isinstance(docs, list)
    assert len(docs) >= 1


def test_load_yaml_extracts_content(yaml_file: Path) -> None:
    """Flattened text should contain key values from the YAML file."""
    docs = load_json_yaml(str(yaml_file))
    combined = " ".join(d.text for d in docs)
    assert "enterprise-rag" in combined
    assert "test-user" in combined
    assert "10" in combined


def test_load_json_yaml_file_not_found() -> None:
    """load_json_yaml should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_json_yaml("/nonexistent/path/data.json")
