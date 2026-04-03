"""Tests for AnswerGenerator — written TDD style; GeminiClient is fully mocked."""

import unittest
from unittest.mock import MagicMock, patch


FAKE_KEY = "AIzaSy_test_key_generator"


def _make_image_mock(width: int = 100, height: int = 80, mode: str = "RGB") -> MagicMock:
    """Return a mock that looks like a PIL.Image."""
    img = MagicMock()
    img.size = (width, height)
    img.mode = mode
    return img


def _make_chunk(text: str, source: str = "doc.pdf", chunk_type: str = "text", image=None) -> dict:
    """Helper to build a retrieved chunk dict."""
    chunk = {"text": text, "source": source, "type": chunk_type}
    if image is not None:
        chunk["image"] = image
    return chunk


class TestAnswerGenerator(unittest.TestCase):

    # ------------------------------------------------------------------
    # test_build_context_text_only
    # ------------------------------------------------------------------

    @patch("src.gemini.generator.GeminiClient")
    def test_build_context_text_only(self, mock_client_cls):
        """_build_context() produces numbered [Source N] entries with correct format."""
        mock_client_cls.return_value = MagicMock()

        from src.gemini.generator import AnswerGenerator
        gen = AnswerGenerator(api_key=FAKE_KEY)

        chunks = [
            _make_chunk("First chunk text.", source="a.pdf", chunk_type="text"),
            _make_chunk("Second chunk text.", source="b.pdf", chunk_type="table"),
        ]

        context = gen._build_context(chunks)

        # Must contain [Source 1] and [Source 2] labels
        self.assertIn("[Source 1]", context)
        self.assertIn("[Source 2]", context)

        # Must contain source names and types
        self.assertIn("a.pdf", context)
        self.assertIn("b.pdf", context)
        self.assertIn("text", context)
        self.assertIn("table", context)

        # Must contain the actual text
        self.assertIn("First chunk text.", context)
        self.assertIn("Second chunk text.", context)

        # [Source 1] must appear before [Source 2]
        self.assertLess(context.index("[Source 1]"), context.index("[Source 2]"))

    # ------------------------------------------------------------------
    # test_build_context_limits_images
    # ------------------------------------------------------------------

    @patch("src.gemini.generator.GeminiClient")
    def test_build_context_limits_images(self, mock_client_cls):
        """_build_context_with_images() collects at most max_images PIL images."""
        mock_client_cls.return_value = MagicMock()

        from src.gemini.generator import AnswerGenerator
        gen = AnswerGenerator(api_key=FAKE_KEY, max_images=2)

        img1 = _make_image_mock(100, 100)
        img2 = _make_image_mock(200, 200)
        img3 = _make_image_mock(300, 300)

        # 4 chunks: 3 have images, 1 is text-only
        chunks = [
            _make_chunk("Text A", source="a.pdf", chunk_type="image", image=img1),
            _make_chunk("Text B", source="b.pdf", chunk_type="image", image=img2),
            _make_chunk("Text C", source="c.pdf", chunk_type="image", image=img3),
            _make_chunk("Text D", source="d.pdf", chunk_type="text"),
        ]

        context_str, images = gen._build_context_with_images(chunks)

        # max_images=2, so must not exceed 2 images
        self.assertLessEqual(len(images), 2)

        # Context string still contains all 4 source labels
        self.assertIn("[Source 1]", context_str)
        self.assertIn("[Source 2]", context_str)
        self.assertIn("[Source 3]", context_str)
        self.assertIn("[Source 4]", context_str)

    # ------------------------------------------------------------------
    # test_post_process_verifies_citations
    # ------------------------------------------------------------------

    @patch("src.gemini.generator.GeminiClient")
    def test_post_process_verifies_citations(self, mock_client_cls):
        """_post_process() sets citation_verified=True when all source_ids are valid."""
        mock_client_cls.return_value = MagicMock()

        from src.gemini.generator import AnswerGenerator
        gen = AnswerGenerator(api_key=FAKE_KEY)

        sources = [_make_chunk("The answer lies in the document text here.", source="x.pdf")]

        answer = {
            "answer": "The answer lies in the document text here.",
            "source_ids": [1],
            "confidence": 0.9,
        }

        result = gen._post_process(answer, sources)

        self.assertTrue(result["citation_verified"])
        self.assertNotIn("warning", result)  # no warning when citations are valid and confidence >= 0.5

    # ------------------------------------------------------------------
    # test_post_process_flags_invalid_citation
    # ------------------------------------------------------------------

    @patch("src.gemini.generator.GeminiClient")
    def test_post_process_flags_invalid_citation(self, mock_client_cls):
        """_post_process() sets citation_verified=False when a source_id is out of range."""
        mock_client_cls.return_value = MagicMock()

        from src.gemini.generator import AnswerGenerator
        gen = AnswerGenerator(api_key=FAKE_KEY)

        # Only 1 source, but answer cites source_id=5
        sources = [_make_chunk("Some text.", source="y.pdf")]

        answer = {
            "answer": "Some text.",
            "source_ids": [5],
            "confidence": 0.8,
        }

        result = gen._post_process(answer, sources)

        self.assertFalse(result["citation_verified"])
        # Must include a warning
        self.assertIn("warning", result)
        self.assertTrue(result["warning"])


if __name__ == "__main__":
    unittest.main()
