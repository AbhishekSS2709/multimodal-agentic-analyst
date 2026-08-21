"""LangSmith tracing setup.

Tracing is opt-in via ``LANGSMITH_API_KEY``.  With no key the graph runs
exactly the same, just untraced — nothing here may raise, because observability
failing must never take the pipeline down with it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "multimodal-agentic-analyst"


def _langsmith_key() -> str:
    return (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()


def configure_tracing(project: Optional[str] = None) -> bool:
    """Turn LangSmith tracing on when a key is present.

    Sets both the current (``LANGSMITH_*``) and legacy (``LANGCHAIN_*``) env
    names so tracing works regardless of which the installed SDK reads.

    Returns whether tracing ended up enabled.  Never raises.
    """
    try:
        key = _langsmith_key()
        if not key:
            os.environ.pop("LANGSMITH_TRACING", None)
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            return False

        os.environ.setdefault("LANGSMITH_API_KEY", key)
        os.environ.setdefault("LANGCHAIN_API_KEY", key)
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault(
            "LANGSMITH_PROJECT", project or os.getenv("LANGSMITH_PROJECT") or DEFAULT_PROJECT
        )
        os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])
        return True
    except Exception as exc:
        logger.warning("Could not configure LangSmith tracing: %s", exc)
        return False


def tracing_enabled() -> bool:
    """True when tracing env is on and a key is present."""
    on = os.getenv("LANGSMITH_TRACING", "").lower() == "true" or \
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    return bool(on and _langsmith_key())


def get_client() -> Optional[Any]:
    """A LangSmith client, or ``None`` when unconfigured. Never raises."""
    if not _langsmith_key():
        return None
    try:
        from langsmith import Client

        return Client(api_key=_langsmith_key())
    except Exception as exc:
        logger.warning("LangSmith client unavailable: %s", exc)
        return None


def run_metadata(**kwargs: Any) -> Dict[str, Any]:
    """Metadata attached to graph runs so LangSmith is filterable.

    ``llm_mode`` in particular lets us compare LLM-driven runs against the
    heuristic fallbacks side by side in the same project.
    """
    from src.graph.llm import llm_mode

    md: Dict[str, Any] = {"llm_mode": llm_mode(), "tracing": tracing_enabled()}
    md.update(kwargs)
    return md
