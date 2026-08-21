"""Supervisor node — decompose the question and pick specialists.

This replaces the keyword dispatch in ``src/agents/query_router.py`` with a
planner that can select *several* specialists for one question.  The heuristic
fallback deliberately reuses the existing ``QueryRouter`` and ``QueryAnalyzer``
so LLM routing and keyword routing are directly comparable in a LangSmith
experiment.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.llm import get_structured_llm
from src.graph.prompts import SUPERVISOR_PROMPT
from src.graph.schemas import RoutePlan, SubTask
from src.graph.state import SPECIALISTS, AnalystState

logger = logging.getLogger(__name__)

# QueryRouter categories that map onto a specialist.
_ANALYTICS_CATEGORIES = {"analytics", "sql"}
_GRAPH_CATEGORIES = {"reasoning", "comparison"}


def _classify(question: str) -> Dict[str, Any]:
    """Existing keyword classifier; neutral result if it is unavailable."""
    try:
        from src.agents.query_router import QueryRouter

        return QueryRouter().classify(question)
    except Exception as exc:
        logger.debug("QueryRouter unavailable: %s", exc)
        return {"category": "factual", "confidence": 0.0}


def _modality(question: str) -> str:
    """Existing modality detector; ``text`` if it is unavailable."""
    try:
        from src.retrieval.query_analyzer import QueryAnalyzer

        return QueryAnalyzer().analyze(question).get("modality", "text")
    except Exception as exc:
        logger.debug("QueryAnalyzer unavailable: %s", exc)
        return "text"


def plan_heuristic(question: str) -> RoutePlan:
    """Pick specialists by keyword and modality signals.

    ``document`` is the floor: text retrieval is almost always worth running,
    and it guarantees the graph never fans out to zero specialists.
    """
    question = (question or "").strip()
    if not question:
        return RoutePlan(
            subtasks=[SubTask(description="", specialist="document")],
            rationale="empty question; defaulting to document search",
        )

    classification = _classify(question)
    category = str(classification.get("category", "factual")).lower()
    modality = _modality(question)

    chosen: List[str] = ["document"]
    reasons = [f"category={category}", f"modality={modality}"]

    if category in _ANALYTICS_CATEGORIES:
        chosen.append("analytics")
    if category in _GRAPH_CATEGORIES:
        chosen.append("graph")
    if modality in ("visual", "both"):
        chosen.append("visual")

    # Preserve SPECIALISTS order, drop duplicates.
    ordered = [s for s in SPECIALISTS if s in set(chosen)]

    return RoutePlan(
        subtasks=[SubTask(description=question, specialist=s) for s in ordered],
        rationale="; ".join(reasons),
    )


def _plan_llm(question: str) -> RoutePlan | None:
    """Ask the LLM for a plan; ``None`` if unavailable or unusable."""
    structured = get_structured_llm(RoutePlan)
    if structured is None:
        return None
    try:
        plan = structured.invoke(SUPERVISOR_PROMPT.format(question=question))
    except Exception as exc:
        logger.warning("LLM planning failed, falling back: %s", exc)
        return None

    # Drop hallucinated specialists and collapse duplicates.
    seen: set[str] = set()
    valid: List[SubTask] = []
    for task in plan.subtasks:
        name = (task.specialist or "").strip().lower()
        if name in SPECIALISTS and name not in seen:
            seen.add(name)
            valid.append(SubTask(description=task.description or question,
                                 specialist=name))
    if not valid:
        logger.warning("LLM plan had no valid specialists; falling back.")
        return None
    return RoutePlan(subtasks=valid, rationale=plan.rationale)


def supervisor_node(state: AnalystState) -> Dict[str, Any]:
    """Plan the work and record which specialists will run."""
    question = state.get("question", "")

    plan = _plan_llm(question)
    mode = "llm"
    if plan is None:
        plan = plan_heuristic(question)
        mode = "heuristic"

    specialists = [t.specialist for t in plan.subtasks]
    logger.info("Supervisor (%s) selected: %s", mode, specialists)

    return {
        "plan": plan.subtasks,
        "specialists": specialists,
        "trace": [f"supervisor:{mode}"],
    }
