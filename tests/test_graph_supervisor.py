"""Tests for the supervisor planning node (heuristic mode)."""

import unittest


class TestPlanHeuristic(unittest.TestCase):

    def test_numeric_question_selects_analytics(self):
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("how many orders were placed last quarter?")
        specialists = {t.specialist for t in plan.subtasks}
        self.assertIn("analytics", specialists)

    def test_visual_question_selects_visual(self):
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("show me the architecture diagram")
        self.assertIn("visual", {t.specialist for t in plan.subtasks})

    def test_causal_question_selects_graph(self):
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("why did shipments from Acme decline?")
        self.assertIn("graph", {t.specialist for t in plan.subtasks})

    def test_document_is_always_included(self):
        from src.graph.nodes.supervisor import plan_heuristic
        for q in ("what is the refund policy",
                  "how many orders were placed",
                  "show me the chart"):
            self.assertIn("document",
                          {t.specialist for t in plan_heuristic(q).subtasks}, q)

    def test_every_specialist_is_known(self):
        from src.graph.state import SPECIALISTS
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("compare Q1 and Q2 revenue trends in the chart")
        for t in plan.subtasks:
            self.assertIn(t.specialist, SPECIALISTS)

    def test_no_duplicate_specialists(self):
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("how many total orders and what is the count of returns")
        specialists = [t.specialist for t in plan.subtasks]
        self.assertEqual(len(specialists), len(set(specialists)))


class TestSupervisorNode(unittest.TestCase):

    def test_node_returns_plan_and_specialists(self):
        from src.graph.nodes.supervisor import supervisor_node
        from src.graph.state import new_state
        out = supervisor_node(new_state("what is the refund policy?"))
        self.assertIn("plan", out)
        self.assertIn("specialists", out)
        self.assertTrue(out["specialists"])
        self.assertTrue(any(t.startswith("supervisor:") for t in out["trace"]))

    def test_empty_question_still_plans(self):
        from src.graph.nodes.supervisor import supervisor_node
        from src.graph.state import new_state
        out = supervisor_node(new_state(""))
        self.assertEqual(out["specialists"], ["document"])


class TestLowConfidenceRouting(unittest.TestCase):
    """QueryRouter is 20% accurate and never exceeds 0.17 confidence on the
    suite, so its label must not be trusted to *suppress* a specialist.

    These are the questions it actually mislabelled as `factual`, which
    suppressed `graph` and left causal questions on plain text retrieval.
    """

    def _specialists(self, question):
        from src.graph.nodes.supervisor import plan_heuristic
        return [t.specialist for t in plan_heuristic(question).subtasks]

    def test_causal_question_reaches_graph(self):
        self.assertIn("graph",
                      self._specialists("How does supplier reliability affect "
                                        "order fulfilment?"))

    def test_contributing_factors_reaches_graph(self):
        self.assertIn("graph",
                      self._specialists("What factors contribute to late "
                                        "deliveries in Q4?"))

    def test_differ_across_reaches_graph(self):
        self.assertIn("graph",
                      self._specialists("How do on-time delivery rates differ "
                                        "across suppliers?"))

    def test_higher_than_reaches_graph(self):
        self.assertIn("graph",
                      self._specialists("Which region has higher fulfilment "
                                        "rates, East or West?"))

    def test_quantitative_question_reaches_analytics(self):
        self.assertIn("analytics",
                      self._specialists("What is the average order value by region?"))

    def test_top_n_reaches_analytics(self):
        self.assertIn("analytics",
                      self._specialists("List the top 5 suppliers by order volume"))

    def test_plain_lookup_does_not_over_route(self):
        """Precision guard: a simple factual lookup must stay on document."""
        picked = self._specialists("Who is the primary contact for Apex Materials?")
        self.assertEqual(picked, ["document"])

    def test_summary_does_not_over_route(self):
        picked = self._specialists("Give me an executive overview of logistics "
                                   "operations")
        self.assertEqual(picked, ["document"])

    def test_vague_question_does_not_over_route(self):
        self.assertEqual(self._specialists("What about the thing with the stuff?"),
                         ["document"])

    def test_empty_question_unchanged(self):
        self.assertEqual(self._specialists(""), ["document"])

    def test_confident_label_is_trusted_without_cues(self):
        """A confident classifier result should not be second-guessed."""
        from unittest import mock
        from src.graph.nodes import supervisor
        with mock.patch.object(supervisor, "_classify",
                               return_value={"category": "factual",
                                             "confidence": 0.95}):
            picked = self._specialists("Why are there dispatch delays?")
        self.assertEqual(picked, ["document"])


if __name__ == "__main__":
    unittest.main()
