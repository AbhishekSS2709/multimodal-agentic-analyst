"""Pydantic models used for structured LLM output.

These are passed to ``llm.with_structured_output(...)`` so the model is forced
to return parseable objects instead of prose we would have to regex.  Each one
also has a heuristic producer elsewhere in the package, so the graph keeps
working when no API key is configured.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """One unit of work the supervisor hands to a specialist."""

    description: str = Field(description="What this specialist should find out.")
    specialist: str = Field(
        description="One of: document, visual, analytics, graph."
    )


class RoutePlan(BaseModel):
    """The supervisor's decomposition of a question."""

    subtasks: List[SubTask] = Field(
        default_factory=list,
        description="Sub-tasks to run, at most one per specialist.",
    )
    rationale: str = Field(default="", description="Why these specialists.")


class GradeDocuments(BaseModel):
    """Relevance grade for a single retrieved document."""

    relevant: bool = Field(description="Does this document help answer the question?")
    score: float = Field(default=0.0, description="Confidence between 0 and 1.")
    reason: str = Field(default="", description="One-line justification.")


class RewrittenQuery(BaseModel):
    """A retry query produced after a weak retrieval round."""

    query: str = Field(description="Reformulated search query.")
    reason: str = Field(default="", description="What was changed and why.")


class Verification(BaseModel):
    """Final answer grading."""

    grounded: bool = Field(description="Is every claim supported by the findings?")
    relevant: bool = Field(default=True, description="Does the answer address the question?")
    score: float = Field(default=0.0, description="Groundedness score between 0 and 1.")
    reason: str = Field(default="", description="One-line justification.")
