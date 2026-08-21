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


if __name__ == "__main__":
    unittest.main()
