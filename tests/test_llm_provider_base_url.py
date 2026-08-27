"""The v1 provider must also reach an OpenAI-compatible self-hosted server.

`src/graph/llm.py` gained `GRAPH_LLM_BASE_URL` so the analyst graph can run
against llama.cpp / vLLM / LM Studio, which have no per-day budget unlike the
Gemini (20 requests/day) and Groq (200,000 tokens/day) free tiers. The v1
provider that `SQLAgent` calls had no equivalent, so the analytics specialist
was stuck on the metered providers even when the rest of the graph was not.

`OPENAI_BASE_URL` closes that gap. These tests construct no client and make no
network call.
"""

import unittest
from unittest import mock


class TestOpenAiBaseUrl(unittest.TestCase):

    def _call(self, base_url, api_key=""):
        import src.llm_provider as lp

        fake_client = mock.MagicMock()
        fake_client.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content=" ok "))])
        fake_openai = mock.MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with mock.patch.dict("sys.modules", {"openai": fake_openai}), \
                mock.patch.object(lp, "OPENAI_BASE_URL", base_url), \
                mock.patch.object(lp, "OPENAI_API_KEY", api_key):
            result = lp._call_openai("q", "sys")
        return fake_openai.OpenAI.call_args.kwargs, result

    def test_base_url_is_passed_to_the_client(self):
        kwargs, result = self._call("http://10.24.6.107:8001/v1")
        self.assertEqual(kwargs.get("base_url"), "http://10.24.6.107:8001/v1")
        self.assertEqual(result, "ok")

    def test_self_hosted_server_needs_no_credential(self):
        """It needs none, but the SDK refuses to construct without one."""
        kwargs, _ = self._call("http://10.24.6.107:8001/v1", api_key="")
        self.assertTrue(kwargs.get("api_key"))

    def test_a_real_key_is_preferred_over_the_placeholder(self):
        kwargs, _ = self._call("http://host/v1", api_key="sk-real")
        self.assertEqual(kwargs.get("api_key"), "sk-real")

    def test_without_a_base_url_nothing_changes(self):
        kwargs, _ = self._call("", api_key="sk-real")
        self.assertNotIn("base_url", kwargs)
        self.assertEqual(kwargs.get("api_key"), "sk-real")


class TestModelIsConfigurable(unittest.TestCase):

    def test_llm_model_reads_the_environment(self):
        """A self-hosted server names its own models, so the default would 404."""
        import importlib
        import os

        import config.settings as cs
        with mock.patch.dict(os.environ, {"LLM_MODEL": "google/gemma-4-E4B-it"}):
            reloaded = importlib.reload(cs)
            self.assertEqual(reloaded.LLM_MODEL, "google/gemma-4-E4B-it")
        importlib.reload(cs)


if __name__ == "__main__":
    unittest.main()
