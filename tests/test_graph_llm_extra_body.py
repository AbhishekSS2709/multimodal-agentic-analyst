"""Reasoning models must be able to stop reasoning into the answer.

Qwen3.5-9B puts its chain of thought in `content`, so a synthesised answer
began "Thinking Process: 1. **Analyze the Request:** ..." and was then scored
as if that were the answer. vLLM turns it off with
`{"chat_template_kwargs": {"enable_thinking": false}}`; llama.cpp and others
spell it differently, so `GRAPH_LLM_EXTRA_BODY` stays a JSON pass-through
rather than a boolean flag.

Malformed values are ignored rather than raised: a bad env var should not take
the whole graph offline when it has a working heuristic path.
"""

import json
import os
import unittest
from unittest import mock

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


class _Env:
    VARS = ("GRAPH_LLM_EXTRA_BODY", "GRAPH_LLM_BASE_URL", "GRAPH_LLM_MODEL",
            "GRAPH_LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")

    def __init__(self, **values):
        self._values = values

    def __enter__(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.VARS}
        os.environ.update({k: v for k, v in self._values.items() if v})
        import src.graph.llm as llm
        self._llm = llm
        self._key = llm.GEMINI_API_KEY
        llm.GEMINI_API_KEY = ""
        llm.get_llm.cache_clear()
        return llm

    def __exit__(self, *exc):
        for k in self.VARS:
            os.environ.pop(k, None)
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
        self._llm.GEMINI_API_KEY = self._key
        self._llm.get_llm.cache_clear()
        return False


class TestExtraBodyParsing(unittest.TestCase):

    def test_valid_json_object_is_returned(self):
        with _Env(GRAPH_LLM_EXTRA_BODY=json.dumps(NO_THINK)) as llm:
            self.assertEqual(llm._extra_body(), NO_THINK)

    def test_unset_is_empty(self):
        with _Env() as llm:
            self.assertEqual(llm._extra_body(), {})

    def test_malformed_json_is_ignored_not_raised(self):
        with _Env(GRAPH_LLM_EXTRA_BODY="{not json") as llm:
            self.assertEqual(llm._extra_body(), {})

    def test_a_json_scalar_is_rejected(self):
        """`extra_body` must be an object; a bare list would break the client."""
        with _Env(GRAPH_LLM_EXTRA_BODY="[1, 2]") as llm:
            self.assertEqual(llm._extra_body(), {})


class TestExtraBodyReachesTheClient(unittest.TestCase):

    def _kwargs(self, **env):
        with _Env(GRAPH_LLM_BASE_URL="http://10.24.6.107:8005/v1",
                  GRAPH_LLM_MODEL="Qwen/Qwen3.5-9B", **env) as llm:
            with mock.patch("langchain.chat_models.init_chat_model") as init:
                llm.get_llm()
        return init.call_args.kwargs

    def test_it_is_passed_through(self):
        kwargs = self._kwargs(GRAPH_LLM_EXTRA_BODY=json.dumps(NO_THINK))
        self.assertEqual(kwargs.get("extra_body"), NO_THINK)

    def test_absent_when_not_configured(self):
        self.assertNotIn("extra_body", self._kwargs())

    def test_malformed_does_not_reach_the_client(self):
        self.assertNotIn("extra_body", self._kwargs(GRAPH_LLM_EXTRA_BODY="oops"))


if __name__ == "__main__":
    unittest.main()
