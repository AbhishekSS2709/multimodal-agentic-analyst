"""Provider resolution for the analyst graph's LLM.

Gemini's free tier allows 20 requests/day per model, and the graph spends ~8
calls per question, so a 25-example LLM evaluation is impossible on it. Groq's
free tier is far larger, so the factory must not be hardwired to one vendor.

Resolution is a pure function so these tests never construct a client or touch
the network.
"""

import os
import unittest
from unittest import mock


class _Env:
    """Context manager that sets exactly the given credential environment."""

    VARS = ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
            "GRAPH_LLM_PROVIDER", "GRAPH_LLM_MODEL")

    def __init__(self, **values):
        self._values = values

    def __enter__(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.VARS}
        os.environ.update({k: v for k, v in self._values.items() if v})
        import src.graph.llm as llm
        self._llm = llm
        self._saved_key = llm.GEMINI_API_KEY
        llm.GEMINI_API_KEY = ""
        llm.get_llm.cache_clear()
        return llm

    def __exit__(self, *exc):
        for k in self.VARS:
            os.environ.pop(k, None)
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
        self._llm.GEMINI_API_KEY = self._saved_key
        self._llm.get_llm.cache_clear()
        return False


class TestProviderResolution(unittest.TestCase):

    def test_no_credentials_resolves_to_nothing(self):
        with _Env() as llm:
            self.assertIsNone(llm.resolve_provider()[0])

    def test_groq_key_selects_groq(self):
        with _Env(GROQ_API_KEY="gsk_test") as llm:
            provider, model = llm.resolve_provider()
            self.assertEqual(provider, "groq")
            self.assertTrue(model)

    def test_gemini_key_selects_google(self):
        with _Env(GEMINI_API_KEY="AIza_test") as llm:
            self.assertEqual(llm.resolve_provider()[0], "google_genai")

    def test_groq_wins_when_both_present(self):
        """Groq has the larger free budget, so prefer it for evaluation runs."""
        with _Env(GROQ_API_KEY="gsk_test", GEMINI_API_KEY="AIza_test") as llm:
            self.assertEqual(llm.resolve_provider()[0], "groq")

    def test_explicit_provider_overrides_autodetection(self):
        with _Env(GROQ_API_KEY="gsk_test", GEMINI_API_KEY="AIza_test",
                  GRAPH_LLM_PROVIDER="google_genai") as llm:
            self.assertEqual(llm.resolve_provider()[0], "google_genai")

    def test_explicit_model_overrides_default(self):
        with _Env(GROQ_API_KEY="gsk_test",
                  GRAPH_LLM_MODEL="llama-3.1-8b-instant") as llm:
            self.assertEqual(llm.resolve_provider()[1], "llama-3.1-8b-instant")

    def test_legacy_LLM_PROVIDER_is_not_consulted(self):
        """.env already carries LLM_PROVIDER=huggingface for the v1 pipeline."""
        with _Env(GROQ_API_KEY="gsk_test") as llm:
            with mock.patch.dict(os.environ, {"LLM_PROVIDER": "huggingface"}):
                self.assertEqual(llm.resolve_provider()[0], "groq")

    def test_unknown_provider_degrades_instead_of_raising(self):
        with _Env(GROQ_API_KEY="gsk_test",
                  GRAPH_LLM_PROVIDER="not-a-provider") as llm:
            self.assertIsNone(llm.get_llm())


class TestModeReporting(unittest.TestCase):

    def test_llm_mode_is_heuristic_without_credentials(self):
        with _Env() as llm:
            self.assertEqual(llm.llm_mode(), "heuristic")
            self.assertFalse(llm.llm_available())


if __name__ == "__main__":
    unittest.main()
