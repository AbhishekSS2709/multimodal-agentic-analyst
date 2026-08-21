"""Tests for LangGraph analyst state and reducers — TDD, no external deps."""

import operator
import unittest


class TestGraphState(unittest.TestCase):

    def test_new_state_initialises_all_keys(self):
        from src.graph.state import new_state
        s = new_state("what were Q3 sales?")
        self.assertEqual(s["question"], "what were Q3 sales?")
        for key in ("messages", "plan", "specialists", "findings",
                    "answer", "citations", "verification", "retry_count",
                    "approval", "trace"):
            self.assertIn(key, s)
        self.assertEqual(s["findings"], [])
        self.assertEqual(s["retry_count"], 0)
        self.assertIsNone(s["approval"])

    def test_findings_reducer_merges_parallel_writes(self):
        """Two specialists writing concurrently must both survive."""
        from src.graph.state import Finding
        a = [Finding(specialist="document", content="a", score=0.9, source="x.pdf")]
        b = [Finding(specialist="visual", content="b", score=0.8, source="y.png")]
        merged = operator.add(a, b)
        self.assertEqual(len(merged), 2)
        self.assertEqual({f.specialist for f in merged}, {"document", "visual"})

    def test_finding_defaults(self):
        from src.graph.state import Finding
        f = Finding(specialist="document", content="c", score=0.5, source="s")
        self.assertEqual(f.modality, "text")
        self.assertEqual(f.doc_id, "")
        self.assertEqual(f.metadata, {})

    def test_route_plan_round_trip(self):
        from src.graph.schemas import RoutePlan, SubTask
        p = RoutePlan(subtasks=[SubTask(description="find sales", specialist="analytics")],
                      rationale="numeric question")
        self.assertEqual(p.subtasks[0].specialist, "analytics")

    def test_specialists_constant(self):
        from src.graph.state import SPECIALISTS
        self.assertEqual(set(SPECIALISTS), {"document", "visual", "analytics", "graph"})


if __name__ == "__main__":
    unittest.main()
