"""Tests for MultimodalRetriever and normalize_scores — written before implementation (TDD)."""

import pytest
from unittest.mock import MagicMock

from src.retrieval.multimodal_retriever import normalize_scores, MultimodalRetriever


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:
    def test_normalize_scores_empty(self):
        assert normalize_scores([]) == []

    def test_normalize_scores_single(self):
        results = [{"score": 0.42, "doc_id": "a", "page_or_slide": 1}]
        out = normalize_scores(results)
        assert len(out) == 1
        assert out[0]["normalized_score"] == 1.0

    def test_normalize_scores_range(self):
        results = [
            {"score": 0.0, "doc_id": "a", "page_or_slide": 1},
            {"score": 0.5, "doc_id": "b", "page_or_slide": 1},
            {"score": 1.0, "doc_id": "c", "page_or_slide": 1},
        ]
        out = normalize_scores(results)
        scores = {r["doc_id"]: r["normalized_score"] for r in out}
        assert scores["a"] == pytest.approx(0.0)
        assert scores["c"] == pytest.approx(1.0)
        assert scores["b"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# MultimodalRetriever
# ---------------------------------------------------------------------------

def _make_text_results(*doc_ids):
    """Return a list of minimal text result dicts."""
    return [
        {"doc_id": did, "page_or_slide": 1, "score": float(i + 1), "type": "text"}
        for i, did in enumerate(doc_ids)
    ]


def _make_visual_results(*doc_ids):
    """Return a list of minimal visual result dicts."""
    return [
        {"doc_id": did, "page_or_slide": 1, "score": float(i + 1), "type": "image"}
        for i, did in enumerate(doc_ids)
    ]


class TestMultimodalRetriever:
    def test_retriever_text_only(self):
        """With no visual_search_fn, retriever returns text results only."""
        text_fn = MagicMock(return_value=_make_text_results("doc1", "doc2", "doc3"))
        retriever = MultimodalRetriever(text_search_fn=text_fn)

        results = retriever.retrieve("what is the policy?", top_k=3)

        text_fn.assert_called_once()
        assert len(results) <= 3
        assert all(r["type"] == "text" for r in results)

    def test_retriever_fuses_text_and_visual(self):
        """Both search functions are called and their results are merged."""
        text_fn = MagicMock(return_value=_make_text_results("docA", "docB"))
        visual_fn = MagicMock(return_value=_make_visual_results("docC", "docD"))

        # Use a mock analyzer that always returns high visual weight so visual_fn is called
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {
            "modality": "visual",
            "text_weight": 0.3,
            "visual_weight": 0.7,
            "visual_score": 3,
        }

        retriever = MultimodalRetriever(
            text_search_fn=text_fn,
            visual_search_fn=visual_fn,
            query_analyzer=mock_analyzer,
        )
        results = retriever.retrieve("show me the chart", top_k=10)

        text_fn.assert_called_once()
        visual_fn.assert_called_once()

        doc_ids = {r["doc_id"] for r in results}
        assert "docA" in doc_ids or "docB" in doc_ids  # text results present
        assert "docC" in doc_ids or "docD" in doc_ids  # visual results present

    def test_retriever_deduplicates(self):
        """When text and visual both return the same (doc_id, page_or_slide), keep the image."""
        # Both return doc_id="shared", page_or_slide=1
        text_fn = MagicMock(return_value=[
            {"doc_id": "shared", "page_or_slide": 1, "score": 0.9, "type": "caption"},
        ])
        visual_fn = MagicMock(return_value=[
            {"doc_id": "shared", "page_or_slide": 1, "score": 0.8, "type": "image"},
        ])

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {
            "modality": "visual",
            "text_weight": 0.5,
            "visual_weight": 0.5,
            "visual_score": 2,
        }

        retriever = MultimodalRetriever(
            text_search_fn=text_fn,
            visual_search_fn=visual_fn,
            query_analyzer=mock_analyzer,
        )
        results = retriever.retrieve("show me the diagram", top_k=5)

        # Only one entry for (shared, 1)
        shared = [r for r in results if r["doc_id"] == "shared" and r["page_or_slide"] == 1]
        assert len(shared) == 1
        assert shared[0]["type"] == "image"
