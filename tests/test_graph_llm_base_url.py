"""Support any OpenAI-compatible endpoint, including a self-hosted one.

Free-tier budgets are the binding constraint on LLM evaluation here: Gemini
allows 20 requests/day per model and Groq 200,000 tokens/day per model, while
the graph spends ~8 calls per question. A self-hosted llama.cpp / vLLM / Ollama
server has no such ceiling, so `GRAPH_LLM_BASE_URL` points the whole graph at
one and is preferred over the metered providers when set.

Resolution stays a pure function -- these tests construct no client and make no
network call.
"""

import os
import unittest


class _Env:
    VARS = ("GRAPH_LLM_BASE_URL", "GRAPH_LLM_API_KEY", "GRAPH_LLM_MODEL",
            "GRAPH_LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY",
            "GOOGLE_API_KEY")

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


BASE = "http://10.24.6.107:8001/v1"
MODEL = "google/gemma-4-E4B-it"


class TestBaseUrlResolution(unittest.TestCase):

    def test_base_url_selects_the_openai_compatible_provider(self):
        with _Env(GRAPH_LLM_BASE_URL=BASE, GRAPH_LLM_MODEL=MODEL) as llm:
            provider, model = llm.resolve_provider()
            self.assertEqual(provider, "openai")
            self.assertEqual(model, MODEL)

    def test_self_hosted_wins_over_metered_providers(self):
        """It has no daily budget, so it should be preferred when available."""
        with _Env(GRAPH_LLM_BASE_URL=BASE, GRAPH_LLM_MODEL=MODEL,
                  GROQ_API_KEY="gsk_x", GEMINI_API_KEY="AIza_x") as llm:
            self.assertEqual(llm.resolve_provider()[0], "openai")

    def test_explicit_provider_still_overrides(self):
        with _Env(GRAPH_LLM_BASE_URL=BASE, GRAPH_LLM_MODEL=MODEL,
                  GROQ_API_KEY="gsk_x", GRAPH_LLM_PROVIDER="groq") as llm:
            self.assertEqual(llm.resolve_provider()[0], "groq")

    def test_base_url_without_a_model_does_not_pretend_to_work(self):
        """The server names its own models; guessing one would 404."""
        with _Env(GRAPH_LLM_BASE_URL=BASE) as llm:
            self.assertIsNone(llm.get_llm())

    def test_no_base_url_leaves_autodetection_unchanged(self):
        with _Env(GROQ_API_KEY="gsk_x") as llm:
            self.assertEqual(llm.resolve_provider()[0], "groq")

    def test_nothing_configured_is_still_heuristic(self):
        with _Env() as llm:
            self.assertIsNone(llm.resolve_provider()[0])
            self.assertEqual(llm.llm_mode(), "heuristic")


class TestClientKwargs(unittest.TestCase):

    def test_base_url_and_key_are_passed_through(self):
        from unittest import mock
        with _Env(GRAPH_LLM_BASE_URL=BASE, GRAPH_LLM_MODEL=MODEL) as llm:
            with mock.patch("langchain.chat_models.init_chat_model") as init:
                llm.get_llm()
            self.assertTrue(init.called)
            kwargs = init.call_args.kwargs
            self.assertEqual(kwargs.get("base_url"), BASE)
            # A local server needs no credential, but the SDK requires one.
            self.assertTrue(kwargs.get("api_key"))


if __name__ == "__main__":
    unittest.main()
