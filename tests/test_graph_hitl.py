"""Tests for human-in-the-loop interrupt and resume."""

import unittest


class _WriteSQLComponents:
    """SQL backend that proposes a destructive statement."""

    multimodal = graph = multi_hop = generator = hybrid = None

    class _SQL:
        def analyze(self, question):
            return {"sql": "DELETE FROM orders WHERE stale = 1",
                    "rows": [], "insight": "cleanup", "columns": []}

    sql = _SQL()


class TestHITL(unittest.TestCase):

    def setUp(self):
        from src.graph.build import build_analyst_graph, get_checkpointer
        self.graph = build_analyst_graph(_WriteSQLComponents(),
                                         checkpointer=get_checkpointer(":memory:"))

    def _cfg(self, thread_id, require_approval=True):
        return {"configurable": {"thread_id": thread_id,
                                 "require_approval": require_approval}}

    def _ask(self, cfg):
        from src.graph.state import new_state
        return self.graph.invoke(new_state("how many stale orders should be removed from the table"), cfg)

    def test_destructive_sql_interrupts(self):
        result = self._ask(self._cfg("hitl-1"))
        self.assertIn("__interrupt__", result)

    def test_interrupt_payload_describes_the_statement(self):
        result = self._ask(self._cfg("hitl-payload"))
        payload = result["__interrupt__"][0].value
        self.assertEqual(payload["type"], "sql_approval")
        self.assertIn("DELETE", payload["sql"])

    def test_resume_with_approval_completes(self):
        from langgraph.types import Command
        cfg = self._cfg("hitl-2")
        self._ask(cfg)
        final = self.graph.invoke(Command(resume="approve"), cfg)
        self.assertNotIn("__interrupt__", final)
        self.assertTrue(final["answer"])

    def test_resume_with_rejection_records_denial(self):
        from langgraph.types import Command
        cfg = self._cfg("hitl-3")
        self._ask(cfg)
        final = self.graph.invoke(Command(resume="reject"), cfg)
        self.assertTrue(any("denied" in t for t in final["trace"]))

    def test_no_interrupt_when_approval_disabled(self):
        result = self._ask(self._cfg("hitl-4", require_approval=False))
        self.assertNotIn("__interrupt__", result)


class TestRunnerHelpers(unittest.TestCase):

    def test_run_query_reports_interrupted(self):
        from src.graph.build import get_checkpointer, run_query
        result = run_query("how many stale orders should be removed from the table",
                           thread_id="hitl-runner",
                           components=_WriteSQLComponents(),
                           checkpointer=get_checkpointer(":memory:"))
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["interrupt_payload"]["type"], "sql_approval")

    def test_resume_query_completes_the_thread(self):
        from src.graph.build import get_checkpointer, resume_query, run_query
        checkpointer = get_checkpointer(":memory:")
        components = _WriteSQLComponents()
        run_query("how many stale orders should be removed from the table",
                  thread_id="hitl-runner-2", components=components,
                  checkpointer=checkpointer)
        final = resume_query("hitl-runner-2", "approve",
                             components=components, checkpointer=checkpointer)
        self.assertFalse(final["interrupted"])


if __name__ == "__main__":
    unittest.main()
