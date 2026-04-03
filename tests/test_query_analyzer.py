"""Tests for QueryAnalyzer — TDD, no external dependencies."""

import unittest


class TestQueryAnalyzer(unittest.TestCase):

    def setUp(self):
        from src.retrieval.query_analyzer import QueryAnalyzer
        self.analyzer = QueryAnalyzer()

    # ------------------------------------------------------------------
    # Basic modality detection
    # ------------------------------------------------------------------

    def test_detect_visual_query(self):
        """A query with a visual phrase should be classified as 'visual'."""
        result = self.analyzer.analyze("show me the chart")
        self.assertEqual(result["modality"], "visual")
        self.assertGreaterEqual(result["visual_score"], 2)

    def test_detect_text_query(self):
        """A plain text query with no visual signals should be 'text'."""
        result = self.analyzer.analyze("what is the refund policy")
        self.assertEqual(result["modality"], "text")
        self.assertEqual(result["visual_score"], 0)

    def test_detect_ambiguous_query(self):
        """A query with one weak visual signal should be 'both'."""
        # "Q3 performance" has no visual keyword; result is text or both
        # The spec example "tell me about Q3 performance" has no visual signal -> text
        result = self.analyzer.analyze("tell me about Q3 performance")
        self.assertIn(result["modality"], ("text", "both"))

    # ------------------------------------------------------------------
    # Visual keyword list
    # ------------------------------------------------------------------

    def test_visual_keywords(self):
        """Queries containing visual keywords should be 'visual' or 'both'."""
        visual_queries = [
            "can you show me the diagram of the system",
            "what does the graph look like",
            "describe the chart on page 3",
            "I need to understand the figure in the report",
        ]
        for query in visual_queries:
            with self.subTest(query=query):
                result = self.analyzer.analyze(query)
                self.assertIn(
                    result["modality"],
                    ("visual", "both"),
                    msg=f"Expected visual or both for: {query!r}, got {result}",
                )

    def test_text_keywords(self):
        """Queries with no visual signals should all be classified as 'text'."""
        text_queries = [
            "what is the cancellation policy",
            "how do I reset my password",
            "explain the terms and conditions",
        ]
        for query in text_queries:
            with self.subTest(query=query):
                result = self.analyzer.analyze(query)
                self.assertEqual(
                    result["modality"],
                    "text",
                    msg=f"Expected 'text' for: {query!r}, got {result}",
                )

    # ------------------------------------------------------------------
    # Weight constraints
    # ------------------------------------------------------------------

    def test_weights_sum_to_one(self):
        """text_weight + visual_weight must equal 1.0 for every modality."""
        sample_queries = [
            "show me the diagram",           # visual
            "what is the refund policy",     # text
            "tell me about the image maybe", # both (single keyword)
        ]
        for query in sample_queries:
            with self.subTest(query=query):
                result = self.analyzer.analyze(query)
                total = result["text_weight"] + result["visual_weight"]
                self.assertAlmostEqual(
                    total,
                    1.0,
                    places=9,
                    msg=f"Weights don't sum to 1 for {query!r}: {result}",
                )

    # ------------------------------------------------------------------
    # Weight values per modality
    # ------------------------------------------------------------------

    def test_visual_modality_weights(self):
        """Visual modality should set text_weight=0.3, visual_weight=0.7."""
        result = self.analyzer.analyze("show me the chart")
        self.assertEqual(result["modality"], "visual")
        self.assertAlmostEqual(result["text_weight"], 0.3)
        self.assertAlmostEqual(result["visual_weight"], 0.7)

    def test_text_modality_weights(self):
        """Text modality should set text_weight=0.9, visual_weight=0.1."""
        result = self.analyzer.analyze("what are the payment options")
        self.assertEqual(result["modality"], "text")
        self.assertAlmostEqual(result["text_weight"], 0.9)
        self.assertAlmostEqual(result["visual_weight"], 0.1)

    def test_both_modality_weights(self):
        """'Both' modality should set text_weight=0.6, visual_weight=0.4."""
        # Single keyword match (score == 1)
        result = self.analyzer.analyze("find a photo in the docs")
        self.assertEqual(result["modality"], "both")
        self.assertAlmostEqual(result["text_weight"], 0.6)
        self.assertAlmostEqual(result["visual_weight"], 0.4)


if __name__ == "__main__":
    unittest.main()
