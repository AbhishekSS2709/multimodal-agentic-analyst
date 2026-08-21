"""LLM factory for the analyst graph.

Central rule of this package: **``get_llm()`` returning ``None`` is a supported
mode, not an error.**  Every node that would call an LLM has a deterministic
heuristic fallback, so the graph runs end to end with no API key.  That keeps
the test suite offline and turns "LLM vs heuristic" into something we can
measure as a LangSmith experiment rather than assume.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE

logger = logging.getLogger(__name__)


def _api_key() -> str:
    """Resolve a Gemini key from the environment, then settings."""
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or GEMINI_API_KEY
        or ""
    ).strip()


@lru_cache(maxsize=8)
def get_llm(
    temperature: Optional[float] = None,
    model: Optional[str] = None,
) -> Optional[Any]:
    """Return a chat model, or ``None`` when no key is configured.

    Cached per (temperature, model) so nodes can call this freely.  Call
    ``get_llm.cache_clear()`` after changing the environment.
    """
    key = _api_key()
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or GEMINI_MODEL,
            temperature=GEMINI_TEMPERATURE if temperature is None else temperature,
            google_api_key=key,
        )
    except Exception as exc:  # missing extra, bad key shape, import error
        logger.warning("LLM unavailable, falling back to heuristics: %s", exc)
        return None


def llm_available() -> bool:
    """True when a real LLM is wired up."""
    return get_llm() is not None


def llm_mode() -> str:
    """``"llm"`` or ``"heuristic"`` — stamped onto LangSmith run metadata."""
    return "llm" if llm_available() else "heuristic"


def get_structured_llm(
    schema: type[BaseModel],
    temperature: Optional[float] = None,
) -> Optional[Any]:
    """Return a model constrained to ``schema``, or ``None`` in heuristic mode."""
    llm = get_llm(temperature)
    if llm is None:
        return None
    try:
        return llm.with_structured_output(schema)
    except Exception as exc:
        logger.warning("Structured output unavailable for %s: %s", schema.__name__, exc)
        return None
