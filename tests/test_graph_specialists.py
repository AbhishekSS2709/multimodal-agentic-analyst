"""Tests for visual, analytics, and knowledge-graph specialists."""

import unittest

from src.graph.state import new_state


class _NullComponents:
    """Every component unavailable — specialists must degrade, not crash."""
    hybrid = multimodal = sql = graph = multi_hop = generator = None


class TestApprovalGate(unittest.TestCase):

    def test_select_needs_no_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertFalse(needs_approval("SELECT * FROM orders LIMIT 10"))

    def test_cte_select_needs_no_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertFalse(needs_approval("WITH t AS (SELECT 1) SELECT * FROM t"))

    def test_delete_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval("DELETE FROM orders"))

    def test_drop_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval("DROP TABLE orders"))

    def test_multi_statement_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval("SELECT 1; DROP TABLE orders"))

    def test_empty_sql_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval(""))

    def test_update_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval("UPDATE orders SET total = 0"))


class TestSpecialistDegradation(unittest.TestCase):
    """With no indices or keys, every specialist returns empty, never raises."""

    def _run(self, node):
        out = node(new_state("q"), _NullComponents())
        self.assertEqual(out["findings"], [])
        self.assertTrue(any(t.endswith(":unavailable") for t in out["trace"]))

    def test_visual_degrades(self):
        from src.graph.nodes.visual import visual_node
        self._run(visual_node)

    def test_analytics_degrades(self):
        from src.graph.nodes.analytics import analytics_node
        self._run(analytics_node)

    def test_graph_degrades(self):
        from src.graph.nodes.graph_specialist import graph_node
        self._run(graph_node)


class TestSpecialistOutputs(unittest.TestCase):

    def test_visual_emits_image_modality(self):
        from src.graph.nodes.visual import visual_node

        class _C:
            hybrid = sql = graph = multi_hop = generator = None

            class _MM:
                def retrieve(self, query, top_k=5, **kw):
                    return [{"text": "a bar chart of Q3 revenue", "doc_id": "img1",
                             "source": "chart.png", "score": 0.8, "modality": "image"}]
            multimodal = _MM()

        out = visual_node(new_state("show me the revenue chart"), _C())
        self.assertTrue(out["findings"])
        self.assertEqual(out["findings"][0].modality, "image")

    def test_analytics_emits_table_finding_for_readonly_sql(self):
        from src.graph.nodes.analytics import analytics_node

        class _C:
            hybrid = multimodal = graph = multi_hop = generator = None

            class _SQL:
                def analyze(self, question):
                    return {"sql": "SELECT COUNT(*) AS n FROM orders",
                            "rows": [{"n": 42}], "columns": ["n"],
                            "insight": "There are 42 orders."}
            sql = _SQL()

        out = analytics_node(new_state("how many orders?"), _C())
        self.assertTrue(out["findings"])
        self.assertIn("42", out["findings"][0].content)

    def test_graph_emits_entity_findings(self):
        from src.graph.nodes.graph_specialist import graph_node

        class _C:
            hybrid = multimodal = sql = multi_hop = generator = None

            class _G:
                def augment_retrieval(self, query, vector_results):
                    return [{"text": "Acme supplies Widget Co", "doc_id": "kg1",
                             "source": "knowledge_graph", "score": 0.6}]
            graph = _G()

        out = graph_node(new_state("why did Acme shipments fall?"), _C())
        self.assertTrue(out["findings"])
        self.assertEqual(out["findings"][0].specialist, "graph")


if __name__ == "__main__":
    unittest.main()
