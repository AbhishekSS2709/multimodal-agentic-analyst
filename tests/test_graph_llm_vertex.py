"""Vertex AI: Gemini through the runtime's service account, no API key."""

from unittest import mock

import pytest

from src.graph import llm as graph_llm


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT",
                 "GOOGLE_CLOUD_LOCATION", "GRAPH_LLM_PROVIDER", "GRAPH_LLM_BASE_URL",
                 "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(graph_llm, "GEMINI_API_KEY", "")
    graph_llm.get_llm.cache_clear()
    yield
    graph_llm.get_llm.cache_clear()


def _vertex_env(monkeypatch, location=None):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    if location:
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", location)


def test_off_by_default():
    assert graph_llm._vertex_config() == {}
    assert graph_llm.resolve_provider() == (None, None)


def test_flag_without_project_is_ignored(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert graph_llm._vertex_config() == {}


def test_vertex_selects_gemini(monkeypatch):
    _vertex_env(monkeypatch)
    provider, model = graph_llm.resolve_provider()
    assert provider == "google_genai"
    assert model == graph_llm.GEMINI_MODEL
    assert graph_llm._vertex_config()["location"] == "us-central1"


def test_explicit_key_still_wins(monkeypatch):
    _vertex_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert graph_llm.resolve_provider()[0] == "groq"


def test_get_llm_passes_vertex_settings_without_a_key(monkeypatch):
    _vertex_env(monkeypatch, location="asia-south1")
    fake = mock.MagicMock()
    with mock.patch("langchain.chat_models.init_chat_model", return_value=fake) as init:
        assert graph_llm.get_llm() is fake
    kwargs = init.call_args.kwargs
    assert kwargs["model_provider"] == "google_genai"
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "demo-project"
    assert kwargs["location"] == "asia-south1"


def test_v1_pipeline_can_use_the_graph_model(monkeypatch):
    import src.llm_provider as lp

    reply = mock.MagicMock()
    reply.content = [{"type": "text", "text": " SELECT 1 "}]
    fake_llm = mock.MagicMock()
    fake_llm.invoke.return_value = reply
    with mock.patch("src.graph.llm.get_llm", return_value=fake_llm):
        assert lp._call_graph_llm("q", "sys") == "SELECT 1"


def test_v1_graph_provider_errors_without_a_model(monkeypatch):
    import src.llm_provider as lp

    with mock.patch("src.graph.llm.get_llm", return_value=None):
        with pytest.raises(RuntimeError):
            lp._call_graph_llm("q", "sys")
