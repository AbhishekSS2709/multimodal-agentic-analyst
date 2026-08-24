"""Make the suite hermetic with respect to credentials.

The suite used to be offline only *by accident*: with no key configured,
``get_llm()`` returned ``None`` and every node took its heuristic path.  Once a
real ``GEMINI_API_KEY`` was present in ``.env``, tests that call ``run_query``
started issuing live Gemini requests and wedged for minutes inside
``tenacity``'s retry backoff after hitting the free-tier quota.

Clearing the environment is not sufficient: ``src.graph.llm._api_key`` falls
back to ``config.settings.GEMINI_API_KEY``, which is read from ``.env`` at
import time, so the module attributes have to be blanked too.

Tests that want an LLM patch one in explicitly (see
``TestSynthesisContentShapes``); nothing here prevents that.
"""

import os

import pytest

_CREDENTIAL_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
)


@pytest.fixture(autouse=True, scope="session")
def _no_live_credentials():
    """Blank every credential path for the whole session."""
    # Import first, then clear.  config/settings.py repopulates os.environ from
    # .env via setdefault() at import time, so popping before the import just
    # hands the keys straight back.
    import config.settings as settings
    from src.graph import llm as graph_llm

    saved = {name: os.environ.pop(name, None) for name in _CREDENTIAL_VARS}

    saved_settings = settings.GEMINI_API_KEY
    saved_llm = graph_llm.GEMINI_API_KEY
    settings.GEMINI_API_KEY = ""
    graph_llm.GEMINI_API_KEY = ""
    graph_llm.get_llm.cache_clear()

    yield

    settings.GEMINI_API_KEY = saved_settings
    graph_llm.GEMINI_API_KEY = saved_llm
    graph_llm.get_llm.cache_clear()
    for name, value in saved.items():
        if value is not None:
            os.environ[name] = value
