"""Tests for LangSmith evaluators — pure functions, no network."""

import os
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

    def setUp(self):
        # Strip any real LangSmith key. Without this, a developer with a key
        # configured would have these tests create datasets in their actual
        # account -- which is exactly what happened once.
        self._saved = {k: os.environ.pop(k, None)
                       for k in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_push_dataset_cannot_touch_a_real_account_in_tests(self):
        """With no key resolvable, push_dataset must be a no-op."""
        from src.evaluation.langsmith_eval import push_dataset
        self.assertIsNone(push_dataset("unit-test-should-never-exist"))

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




class TestExperimentScoping(unittest.TestCase):
    """The free Gemini tier caps daily requests, so experiments must be scopable."""

    def test_select_examples_filters_by_category(self):
        from src.evaluation.langsmith_eval import select_examples
        picked = select_examples(categories=["sql"])
        self.assertTrue(picked)
        self.assertTrue(all(e["outputs"]["category"] == "sql" for e in picked))

    def test_select_examples_accepts_multiple_categories(self):
        from src.evaluation.langsmith_eval import select_examples
        picked = select_examples(categories=["sql", "factual"])
        self.assertEqual({e["outputs"]["category"] for e in picked}, {"sql", "factual"})

    def test_select_examples_respects_limit(self):
        from src.evaluation.langsmith_eval import select_examples
        self.assertEqual(len(select_examples(limit=2)), 2)

    def test_select_examples_unknown_category_is_empty(self):
        from src.evaluation.langsmith_eval import select_examples
        self.assertEqual(select_examples(categories=["nope"]), [])

    def test_select_examples_defaults_to_everything(self):
        from src.evaluation.langsmith_eval import build_examples, select_examples
        self.assertEqual(len(select_examples()), len(build_examples()))

class TestRoutingPrecision(unittest.TestCase):
    """routing_accuracy is recall-only, so over-routing costs nothing.

    A planner that dispatches every specialist scores 1.0 on recall while doing
    several times the work, which is exactly what the LLM planner did.
    """

    def test_exact_routing_is_perfect(self):
        from src.evaluation.langsmith_eval import routing_precision
        run = {"specialists": ["document", "graph"]}
        self.assertEqual(routing_precision(run, {"specialists": ["graph"]})["score"], 1.0)

    def test_spurious_specialist_is_penalised(self):
        from src.evaluation.langsmith_eval import routing_precision
        run = {"specialists": ["document", "analytics", "graph"]}
        self.assertEqual(routing_precision(run, {"specialists": ["graph"]})["score"], 0.5)

    def test_routing_everything_is_not_free(self):
        from src.evaluation.langsmith_eval import routing_precision, routing_accuracy
        run = {"specialists": ["document", "visual", "analytics", "graph"]}
        example = {"specialists": ["graph"]}
        self.assertEqual(routing_accuracy(run, example)["score"], 1.0)
        self.assertLess(routing_precision(run, example)["score"], 0.5)

    def test_document_floor_is_not_a_false_positive(self):
        """`document` is always routed by design; it must not count against us."""
        from src.evaluation.langsmith_eval import routing_precision
        run = {"specialists": ["document"]}
        self.assertIsNone(routing_precision(run, {"specialists": ["document"]})["score"])

    def test_missing_specialist_is_recall_not_precision(self):
        from src.evaluation.langsmith_eval import routing_precision, routing_accuracy
        run = {"specialists": ["document"]}
        example = {"specialists": ["graph"]}
        self.assertEqual(routing_accuracy(run, example)["score"], 0.0)
        self.assertIsNone(routing_precision(run, example)["score"])

    def test_no_expectation_is_neutral(self):
        from src.evaluation.langsmith_eval import routing_precision
        self.assertIsNone(routing_precision({"specialists": ["document"]}, {})["score"])

    def test_registered_in_evaluators(self):
        from src.evaluation.langsmith_eval import EVALUATORS
        self.assertIn("routing_precision", EVALUATORS)


if __name__ == "__main__":
    unittest.main()
