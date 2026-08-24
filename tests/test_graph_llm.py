"""Tests for the LLM factory and LangSmith wiring — offline, no keys."""

import os
import unittest


class TestLLMFactory(unittest.TestCase):

    def setUp(self):
        # Guarantee the no-key path regardless of the developer's environment.
        self._saved = {k: os.environ.pop(k, None)
                       for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        import src.graph.llm as llm_mod
        llm_mod.get_llm.cache_clear()
        self.llm_mod = llm_mod

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
        self.llm_mod.get_llm.cache_clear()

    def test_get_llm_returns_none_without_key(self):
        self.assertIsNone(self.llm_mod.get_llm())

    def test_llm_available_false_without_key(self):
        self.assertFalse(self.llm_mod.llm_available())

    def test_get_structured_llm_returns_none_without_key(self):
        from src.graph.schemas import RoutePlan
        self.assertIsNone(self.llm_mod.get_structured_llm(RoutePlan))


class TestObservability(unittest.TestCase):

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None)
                       for k in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
                                 "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")}

    def tearDown(self):
        for k in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            os.environ.pop(k, None)
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_configure_tracing_off_without_key(self):
        from src.graph.observability import configure_tracing, tracing_enabled
        self.assertFalse(configure_tracing())
        self.assertFalse(tracing_enabled())

    def test_configure_tracing_never_raises(self):
        from src.graph.observability import configure_tracing
        configure_tracing()  # must not raise even with no key
        os.environ["LANGSMITH_API_KEY"] = "lsv2_fake_key_for_test"
        self.assertTrue(configure_tracing())
        self.assertEqual(os.environ.get("LANGSMITH_TRACING"), "true")

    def test_run_metadata_includes_flags(self):
        from src.graph.observability import run_metadata
        md = run_metadata(specialist="document")
        self.assertEqual(md["specialist"], "document")
        self.assertIn("llm_mode", md)
        self.assertIn(md["llm_mode"], ("llm", "heuristic"))


if __name__ == "__main__":
    unittest.main()


class TestSuiteIsHermetic(unittest.TestCase):
    """Guard: the suite must never reach a live LLM, key present or not.

    Without this, the tests pass offline only because no credentials happen to
    be configured -- and start making real API calls the moment they are.
    """

    def test_no_ambient_llm(self):
        from src.graph.llm import llm_available, llm_mode
        self.assertFalse(llm_available(),
                         "a live LLM leaked into the test session")
        self.assertEqual(llm_mode(), "heuristic")

    def test_no_ambient_gemini_key(self):
        import config.settings as settings
        from src.graph import llm as graph_llm
        self.assertFalse(settings.GEMINI_API_KEY)
        self.assertFalse(graph_llm.GEMINI_API_KEY)
        self.assertFalse(graph_llm._api_key())
