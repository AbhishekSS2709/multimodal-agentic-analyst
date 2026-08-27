"""The analytics specialist must follow the graph's LLM decision.

`SQLAgent` autodetects a provider of its own -- Ollama, then Gemini -- which is
correct for the v1 pipeline but wrong inside the analyst graph. In heuristic
mode the graph deliberately has no LLM, so a SQL agent that reached for Gemini
anyway would break the offline guarantee *and* silently contaminate the
LLM-vs-heuristic experiment, where the whole point is that the two arms differ
only in whether the graph has a model.

`SQLAgent` has a deterministic pattern-matching mode, so `use_llm=False` still
queries the warehouse -- it just writes the SQL without a model.
"""

import unittest
from unittest import mock


class _Orch:
    """Records the flag the graph asks for."""

    def __init__(self):
        self.calls = []

    def _get_sql_pipeline(self, use_llm=None):
        self.calls.append(use_llm)
        return f"pipeline(use_llm={use_llm})"


class TestSqlFollowsGraphLlmMode(unittest.TestCase):

    def _sql_with(self, available):
        from src.graph.adapters import LazyComponents

        orch = _Orch()
        with mock.patch("src.graph.llm.llm_available", return_value=available):
            result = LazyComponents(orch).sql
        return orch, result

    def test_heuristic_mode_asks_for_deterministic_sql(self):
        orch, result = self._sql_with(False)
        self.assertEqual(orch.calls, [False])
        self.assertEqual(result, "pipeline(use_llm=False)")

    def test_llm_mode_asks_for_llm_sql(self):
        orch, _ = self._sql_with(True)
        self.assertEqual(orch.calls, [True])

    def test_missing_orchestrator_still_returns_none(self):
        from src.graph.adapters import LazyComponents

        self.assertIsNone(LazyComponents(None).sql)

    def test_broken_orchestrator_degrades_rather_than_raising(self):
        from src.graph.adapters import LazyComponents

        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("no database")

        self.assertIsNone(LazyComponents(_Broken()).sql)


class TestOrchestratorPassesTheFlagThrough(unittest.TestCase):

    def test_use_llm_reaches_the_pipeline_constructor(self):
        from src.pipeline_orchestrator import EnterpriseRAGOrchestrator

        orch = EnterpriseRAGOrchestrator()
        with mock.patch("src.sql_tool.sql_pipeline.SQLAnalyticsPipeline") as cls:
            orch._get_sql_pipeline(use_llm=False)
        cls.assert_called_once_with(use_llm=False)

    def test_default_keeps_v1_autodetection(self):
        """v1 callers pass nothing and must keep SQLAgent's own detection."""
        from src.pipeline_orchestrator import EnterpriseRAGOrchestrator

        orch = EnterpriseRAGOrchestrator()
        with mock.patch("src.sql_tool.sql_pipeline.SQLAnalyticsPipeline") as cls:
            orch._get_sql_pipeline()
        cls.assert_called_once_with(use_llm=None)

    def test_switching_mode_rebuilds_rather_than_serving_a_stale_pipeline(self):
        from src.pipeline_orchestrator import EnterpriseRAGOrchestrator

        orch = EnterpriseRAGOrchestrator()
        with mock.patch("src.sql_tool.sql_pipeline.SQLAnalyticsPipeline") as cls:
            orch._get_sql_pipeline(use_llm=False)
            orch._get_sql_pipeline(use_llm=False)   # cached
            orch._get_sql_pipeline(use_llm=True)    # mode changed -> rebuild
        self.assertEqual([c.kwargs["use_llm"] for c in cls.call_args_list],
                         [False, True])


if __name__ == "__main__":
    unittest.main()
