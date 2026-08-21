"""Shared state for the analyst graph.

The two keys written by more than one node at a time — ``findings`` and
``trace`` — use ``operator.add`` reducers.  Specialists run in the same
LangGraph superstep, so without a reducer their writes would collide and only
the last one would survive.  Every other key is written by exactly one node,
where last-write-wins is correct.
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.graph.schemas import (  # noqa: F401  (re-exported for one import site)
    GradeDocuments,
    RewrittenQuery,
    RoutePlan,
    SubTask,
    Verification,
)

# The four specialist subgraphs the supervisor may delegate to.
SPECIALISTS: tuple[str, ...] = ("document", "visual", "analytics", "graph")


class Finding(BaseModel):
    """One piece of evidence produced by a specialist.

    All four specialists emit this same shape so the synthesizer can treat a
    text passage, an image caption, a SQL result row and a graph path
    uniformly.
    """

    specialist: str
    content: str
    score: float = 0.0
    source: str = ""
    doc_id: str = ""
    modality: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A source the final answer actually leans on."""

    source: str
    doc_id: str = ""
    snippet: str = ""
    specialist: str = ""


class AnalystState(TypedDict):
    """State threaded through the whole analyst graph."""

    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    plan: List[SubTask]
    specialists: List[str]
    findings: Annotated[List[Finding], operator.add]
    answer: str
    citations: List[Citation]
    verification: Dict[str, Any]
    retry_count: int
    approval: Optional[str]
    trace: Annotated[List[str], operator.add]


def new_state(question: str) -> AnalystState:
    """Build a fresh state for one question."""
    return AnalystState(
        messages=[],
        question=question,
        plan=[],
        specialists=[],
        findings=[],
        answer="",
        citations=[],
        verification={},
        retry_count=0,
        approval=None,
        trace=[],
    )
