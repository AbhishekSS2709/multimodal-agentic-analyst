"""Tests for LangSmith evaluators — pure functions, no network."""

import unittest


class TestEvaluators(unittest.TestCase):

    def test_faithfulness_rewards_grounded_answers(self):
        from src.evaluation.langsmith_eval import faithfulness
        run = {"answer": "Refunds are issued within 30 days.",
               "findings": [{"content": "Refunds are issued within 30 days of purchase."}]}
        self.assertGreaterEqual(faithfulness(run, {})["score"], 0.5)

    def test_faithfulness_punishes_ungrounded_answers(self):
        from src.evaluation.langsmith_eval import faithfulness
        run = {"answer": "Penguins founded the company on Jupiter.",
               "findings": [{"content": "Refunds are issued within 30 days."}]}
        self.assertLess(faithfulness(run, {})["score"], 0.5)

    def test_faithfulness_without_findings_is_zero(self):
        from src.evaluation.langsmith_eval import faithfulness
        self.assertEqual(faithfulness({"answer": "x", "findings": []}, {})["score"], 0.0)

    def test_citation_accuracy_requires_cited_sources_in_findings(self):
        from src.evaluation.langsmith_eval import citation_accuracy
        good = {"citations": [{"source": "policy.pdf"}],
                "findings": [{"source": "policy.pdf"}]}
        bad = {"citations": [{"source": "ghost.pdf"}],
               "findings": [{"source": "policy.pdf"}]}
        self.assertEqual(citation_accuracy(good, {})["score"], 1.0)
        self.assertEqual(citation_accuracy(bad, {})["score"], 0.0)

    def test_routing_accuracy_scores_specialist_overlap(self):
        from src.evaluation.langsmith_eval import routing_accuracy
        run = {"specialists": ["document", "analytics"]}
        self.assertEqual(routing_accuracy(run, {"specialists": ["analytics"]})["score"], 1.0)
        self.assertEqual(routing_accuracy(run, {"specialists": ["visual"]})["score"], 0.0)

    def test_routing_accuracy_without_expectation_is_neutral(self):
        from src.evaluation.langsmith_eval import routing_accuracy
        self.assertIsNone(routing_accuracy({"specialists": ["document"]}, {})["score"])

    def test_modality_match(self):
        from src.evaluation.langsmith_eval import modality_match
        run = {"findings": [{"modality": "image"}]}
        self.assertEqual(modality_match(run, {"modality": "image"})["score"], 1.0)

    def test_modality_match_text_expectation(self):
        from src.evaluation.langsmith_eval import modality_match
        run = {"findings": [{"modality": "text"}]}
        self.assertEqual(modality_match(run, {"modality": "text"})["score"], 1.0)

    def test_answer_correctness_uses_expected_keywords(self):
        from src.evaluation.langsmith_eval import answer_correctness
        run = {"answer": "Apex Materials is the worst supplier."}
        self.assertEqual(
            answer_correctness(run, {"expected_answer_contains": ["Apex Materials"]})["score"],
            1.0)
        self.assertEqual(
            answer_correctness(run, {"expected_answer_contains": ["Globex"]})["score"],
            0.0)

    def test_retry_efficiency_prefers_fewer_retries(self):
        from src.evaluation.langsmith_eval import retry_efficiency
        self.assertGreater(retry_efficiency({"retry_count": 0}, {})["score"],
                           retry_efficiency({"retry_count": 2}, {})["score"])

    def test_all_evaluators_registered(self):
        from src.evaluation.langsmith_eval import EVALUATORS
        for name in ("faithfulness", "citation_accuracy", "routing_accuracy",
                     "modality_match", "answer_correctness", "retry_efficiency"):
            self.assertIn(name, EVALUATORS)

    def test_every_evaluator_returns_the_standard_shape(self):
        from src.evaluation.langsmith_eval import EVALUATORS
        run = {"answer": "a", "findings": [{"content": "a", "source": "s",
                                            "modality": "text"}],
               "citations": [], "specialists": ["document"], "retry_count": 0}
        for name, fn in EVALUATORS.items():
            result = fn(run, {})
            self.assertEqual(result["key"], name)
            self.assertIn("score", result)
            self.assertIn("comment", result)


class TestDatasetBuilding(unittest.TestCase):

    def test_build_examples_from_test_cases(self):
        from src.evaluation.langsmith_eval import build_examples
        examples = build_examples()
        self.assertTrue(examples)
        self.assertIn("question", examples[0]["inputs"])

    def test_examples_carry_expectations(self):
        from src.evaluation.langsmith_eval import build_examples
        examples = build_examples()
        self.assertTrue(any("expected_answer_contains" in e["outputs"] for e in examples))
        self.assertTrue(any("modality" in e["outputs"] for e in examples))

    def test_expected_specialists_derived_from_category(self):
        from src.evaluation.langsmith_eval import expected_specialists_for
        self.assertIn("analytics", expected_specialists_for("sql"))
        self.assertIn("graph", expected_specialists_for("reasoning"))
        self.assertIn("visual", expected_specialists_for("visual_reasoning"))

    def test_push_dataset_noop_without_key(self):
        from src.evaluation.langsmith_eval import push_dataset
        self.assertIsNone(push_dataset("test-ds", client=None))

    def test_mirror_feedback_noop_without_key(self):
        from src.evaluation.langsmith_eval import mirror_feedback
        self.assertFalse(mirror_feedback("q1", 5, "great", client=None))


class TestScoreRun(unittest.TestCase):

    def test_score_run_aggregates_all_evaluators(self):
        from src.evaluation.langsmith_eval import score_run
        run = {"answer": "Refunds are issued within 30 days.",
               "findings": [{"content": "Refunds are issued within 30 days.",
                             "source": "p.pdf", "modality": "text"}],
               "citations": [{"source": "p.pdf"}],
               "specialists": ["document"], "retry_count": 0}
        scores = score_run(run, {"expected_answer_contains": ["30 days"],
                                 "modality": "text"})
        self.assertIn("faithfulness", scores)
        self.assertEqual(scores["answer_correctness"], 1.0)


if __name__ == "__main__":
    unittest.main()
