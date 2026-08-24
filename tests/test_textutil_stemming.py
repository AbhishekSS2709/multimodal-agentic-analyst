"""The shared stemmer must fold -ies onto -y.

`_stem` stripped a trailing "s" but had no rule for "-ies", so:

    "deliveries" -> "deliverie"        "delivery" -> "delivery"

A question about "late deliveries" therefore could not match a document saying
"late delivery". Overlap for "What factors contribute to late deliveries in Q4?"
came to 0.200 against a 0.300 relevance threshold, so the document grader
rejected every genuinely relevant chunk and the graph abstained outright --
answer_correctness 0.0 on a question the corpus can answer.

The same tokenizer backs the grader, the synthesizer's ranking, the verifier's
groundedness check and answer_correctness, so the mismatch was consistent
everywhere.
"""

import unittest

from src.graph.textutil import _stem, overlap_ratio, tokenize


class TestPluralFolding(unittest.TestCase):

    def test_ies_folds_onto_y(self):
        for plural, singular in [("deliveries", "delivery"),
                                 ("companies", "company"),
                                 ("categories", "category"),
                                 ("discrepancies", "discrepancy")]:
            self.assertEqual(_stem(plural), _stem(singular),
                             f"{plural} should match {singular}")

    def test_plain_s_still_folds(self):
        self.assertEqual(_stem("rates"), _stem("rate"))
        self.assertEqual(_stem("suppliers"), _stem("supplier"))

    def test_double_s_is_untouched(self):
        self.assertEqual(_stem("process"), "process")

    def test_short_words_are_not_mangled(self):
        for word in ("is", "as", "his"):
            self.assertEqual(_stem(word), word)


class TestGraderThresholdCase(unittest.TestCase):
    """The exact case that made the graph abstain."""

    QUESTION = "What factors contribute to late deliveries in Q4?"
    DOCUMENT = ("DISPATCH-1026 | Status Update: DELAYED | 11-day delay | "
                "late delivery caused by supplier backlog")

    def test_relevant_document_clears_the_threshold(self):
        from src.graph.nodes.document import RELEVANCE_THRESHOLD
        score = overlap_ratio(self.QUESTION, self.DOCUMENT)
        self.assertGreaterEqual(score, RELEVANCE_THRESHOLD,
                                f"relevant document still scores {score}")

    def test_plural_question_matches_singular_document(self):
        self.assertIn("delivery", set(tokenize("late deliveries")))

    def test_unrelated_document_still_fails(self):
        """The fix must not simply make everything relevant."""
        from src.graph.nodes.document import RELEVANCE_THRESHOLD
        unrelated = "The office cafeteria menu changes on Mondays."
        self.assertLess(overlap_ratio(self.QUESTION, unrelated),
                        RELEVANCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
