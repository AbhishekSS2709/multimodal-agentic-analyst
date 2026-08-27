"""Pydantic models used for structured LLM output.

These are passed to ``llm.with_structured_output(...)`` so the model is forced
to return parseable objects instead of prose we would have to regex.  Each one
also has a heuristic producer elsewhere in the package, so the graph keeps
working when no API key is configured.
"""

from __future__ import annotations

from typing import Annotated, List, Union

from pydantic import BaseModel, Field, field_validator

# Some models (qwen/qwen3.6-27b among them) emit booleans as the strings "true"
# / "false".  Providers validate tool-call arguments against the JSON Schema we
# send *before* returning, so a bare `bool` field makes the whole call fail with
# "expected boolean, but got string" -- and the node then answers heuristically
# while looking like a working LLM run.  Declaring the union widens the emitted
# schema; the validator below narrows the value straight back to a real bool.
LenientBool = Union[bool, str]

# Smaller models answer the *question* rather than filling the field: asked
# whether a document is relevant, google/gemma-4-E4B-it returns "Not Relevant".
# That is a perfectly clear grade, so read it rather than discarding the call.
_TRUE = {"true", "yes", "y", "1", "relevant"}
_FALSE = {"false", "no", "n", "0", "not relevant", "irrelevant", "not_relevant"}


def _coerce_bool(value: object) -> object:
    """Narrow a string boolean back to a real bool.

    Anything unrecognised raises rather than passing through: the union would
    otherwise keep it as a truthy string, so a garbled grade would silently read
    as "relevant".  Raising makes the node fall back to its heuristic grader,
    which is the safe outcome.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"cannot interpret {value!r} as a boolean")
    return value


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

    relevant: LenientBool = Field(
        description="Does this document help answer the question?")
    score: float = Field(default=0.0, description="Confidence between 0 and 1.")
    reason: str = Field(default="", description="One-line justification.")

    _coerce = field_validator("relevant", mode="before")(_coerce_bool)


class RewrittenQuery(BaseModel):
    """A retry query produced after a weak retrieval round."""

    query: str = Field(description="Reformulated search query.")
    reason: str = Field(default="", description="What was changed and why.")


class Verification(BaseModel):
    """Final answer grading."""

    grounded: LenientBool = Field(
        description="Is every claim supported by the findings?")
    relevant: LenientBool = Field(
        default=True, description="Does the answer address the question?")
    score: float = Field(default=0.0, description="Groundedness score between 0 and 1.")
    reason: str = Field(default="", description="One-line justification.")

    _coerce = field_validator("grounded", "relevant", mode="before")(_coerce_bool)
