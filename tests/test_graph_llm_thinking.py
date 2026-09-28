"""Short structured judgements run Gemini 3 at a low thinking level.

Gemini 3 defaults to "high". The graph makes five to fifteen structured calls
per question (grade each passage, route, verify), and at "high" a demo
question took 40-95 seconds. The answer itself keeps the model's default.
"""

from unittest import mock

import pytest
from pydantic import BaseModel

from src.graph import llm as graph_llm


class _Schema(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("GRAPH_LLM_THINKING_FAST", "GRAPH_LLM_MODEL", "GRAPH_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    graph_llm.get_llm.cache_clear()
    yield
    graph_llm.get_llm.cache_clear()


def _vertex(monkeypatch, model="gemini-3.8-flash"):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GRAPH_LLM_MODEL", model)


@pytest.mark.parametrize("model,expected", [
    ("gemini-3.8-flash", True),
    ("gemini-3-flash-preview", True),
    ("models/gemini-3.1-pro", True),
    ("gemini-2.5-flash", False),
    ("gemini-1.5-pro", False),
])
def test_only_gemini_3_takes_a_thinking_level(model, expected):
    assert graph_llm._supports_thinking_level("google_genai", model) is expected


def test_other_providers_never_get_one():
    assert graph_llm._supports_thinking_level("groq", "gemini-3.8-flash") is False


def test_default_fast_level_is_low_on_gemini_3(monkeypatch):
    _vertex(monkeypatch)
    assert graph_llm.fast_thinking_level() == "low"


def test_no_level_on_gemini_2_5(monkeypatch):
    _vertex(monkeypatch, model="gemini-2.5-flash")
    assert graph_llm.fast_thinking_level() is None


def test_env_override_and_off(monkeypatch):
    _vertex(monkeypatch)
    monkeypatch.setenv("GRAPH_LLM_THINKING_FAST", "minimal")
    assert graph_llm.fast_thinking_level() == "minimal"
    monkeypatch.setenv("GRAPH_LLM_THINKING_FAST", "off")
    assert graph_llm.fast_thinking_level() is None


def test_bad_value_is_ignored_not_sent(monkeypatch):
    _vertex(monkeypatch)
    monkeypatch.setenv("GRAPH_LLM_THINKING_FAST", "turbo")
    assert graph_llm.fast_thinking_level() is None


def test_no_llm_means_no_level():
    assert graph_llm.fast_thinking_level() is None


def test_structured_calls_are_built_with_the_fast_level(monkeypatch):
    _vertex(monkeypatch)
    with mock.patch("langchain.chat_models.init_chat_model",
                    return_value=mock.MagicMock()) as init:
        graph_llm.get_structured_llm(_Schema)
    assert init.call_args.kwargs["thinking_level"] == "low"


def test_the_answer_model_keeps_the_default(monkeypatch):
    _vertex(monkeypatch)
    with mock.patch("langchain.chat_models.init_chat_model",
                    return_value=mock.MagicMock()) as init:
        graph_llm.get_llm()
    assert "thinking_level" not in init.call_args.kwargs


def test_a_level_asked_of_gemini_2_5_is_not_sent(monkeypatch):
    _vertex(monkeypatch, model="gemini-2.5-flash")
    with mock.patch("langchain.chat_models.init_chat_model",
                    return_value=mock.MagicMock()) as init:
        graph_llm.get_llm(None, None, "low")
    assert "thinking_level" not in init.call_args.kwargs
