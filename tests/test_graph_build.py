"""End-to-end tests for the assembled analyst graph (offline)."""

import unittest


class _StubComponents:
    """Only the document path is wired; the rest are unavailable."""

    def __init__(self, rows):
        self._rows = rows
        self.multimodal = self.sql = self.graph = self.multi_hop = self.generator = None

    @property
    def hybrid(self):
        rows = self._rows

        class _B:
            def retrieve(self, query, top_k=5, **kw):
                return rows[:top_k]

        return _B()


ROWS = [{"text": "Refunds are issued within 30 days of purchase.",
         "doc_id": "d1", "source": "policy.pdf", "score": 0.95}]


class TestGraphAssembly(unittest.TestCase):

    def test_graph_compiles(self):
        from src.graph.build import build_analyst_graph
        self.assertIsNotNone(build_analyst_graph(_StubComponents(ROWS)))

    def test_mermaid_contains_core_nodes(self):
        from src.graph.build import graph_mermaid
        m = graph_mermaid()
        for node in ("supervisor", "synthesizer", "verifier", "document"):
            self.assertIn(node, m)

    def test_end_to_end_produces_grounded_answer(self):
        from src.graph.build import run_query
        result = run_query("what is the refund window?",
                           components=_StubComponents(ROWS),
                           require_approval=False)
        self.assertIn("30 days", result["answer"])
        self.assertTrue(result["citations"])
        self.assertTrue(any(t.startswith("supervisor:") for t in result["trace"]))
        self.assertFalse(result["interrupted"])

    def test_verification_is_reported(self):
        from src.graph.build import run_query
        result = run_query("what is the refund window?",
                           components=_StubComponents(ROWS),
                           require_approval=False)
        self.assertIn("grounded", result["verification"])

    def test_parallel_findings_merge_without_loss(self):
        """Multiple specialists in one superstep must all contribute."""
        from src.graph.build import run_query
        result = run_query("show me the chart of total orders and why they fell",
                           components=_StubComponents(ROWS),
                           require_approval=False)
        specialists = {t.split(":")[0] for t in result["trace"]}
        self.assertGreaterEqual(
            len(specialists & {"document", "visual", "analytics", "graph"}), 2)

    def test_no_findings_yields_honest_answer(self):
        from src.graph.build import run_query
        result = run_query("what is the refund window?",
                           components=_StubComponents([]),
                           require_approval=False)
        self.assertRegex(result["answer"].lower(), r"could not|no relevant|not find")


class TestCheckpointing(unittest.TestCase):

    def test_memory_checkpointer_round_trips_a_thread(self):
        from src.graph.build import build_analyst_graph, get_checkpointer
        from src.graph.state import new_state
        g = build_analyst_graph(_StubComponents(ROWS),
                                checkpointer=get_checkpointer(":memory:"))
        cfg = {"configurable": {"thread_id": "t1", "require_approval": False}}
        g.invoke(new_state("what is the refund window?"), cfg)
        snapshot = g.get_state(cfg)
        self.assertEqual(snapshot.values["question"], "what is the refund window?")
        self.assertTrue(snapshot.values["answer"])


if __name__ == "__main__":
    unittest.main()
