"""Verifier node — grade the answer for groundedness before returning it.

This is the guard against the failure mode that matters most in RAG: a fluent
answer that the retrieved evidence does not actually support.  A failed grade
sends the graph back to the synthesizer once.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.llm import get_structured_llm
from src.graph.nodes.synthesizer import NO_ANSWER, format_findings
from src.graph.prompts import VERIFY_PROMPT
from src.graph.schemas import Verification
from src.graph.state import AnalystState, Finding
from src.graph.textutil import overlap_ratio

logger = logging.getLogger(__name__)

# Fraction of the answer's content words that must appear in the findings.
GROUNDEDNESS_THRESHOLD = 0.35
RELEVANCE_THRESHOLD = 0.10
MAX_SYNTHESIS_RETRIES = 1


def grade_grounded_heuristic(answer: str, findings: List[Finding]) -> Verification:
    """Token-containment groundedness check."""
    if not findings:
        return Verification(
            grounded=False, relevant=False, score=0.0,
            reason="no findings to ground the answer in",
        )

    corpus = "\n".join(f.content for f in findings)
    score = overlap_ratio(answer, corpus)
    grounded = score >= GROUNDEDNESS_THRESHOLD
    return Verification(
        grounded=grounded,
        relevant=True,
        score=round(score, 3),
        reason=f"{score:.0%} of answer terms appear in the findings",
    )


def _grade_llm(question: str, answer: str, findings: List[Finding]) -> Verification | None:
    """LLM-as-judge groundedness grade; ``None`` when unavailable."""
    structured = get_structured_llm(Verification)
    if structured is None or not findings:
        return None
    try:
        return structured.invoke(VERIFY_PROMPT.format(
            question=question,
            findings=format_findings(findings),
            answer=answer,
        ))
    except Exception as exc:
        logger.warning("LLM verification failed, falling back: %s", exc)
        return None


def verifier_node(state: AnalystState) -> Dict[str, Any]:
    """Grade the answer and count the attempt."""
    question = state.get("question", "")
    answer = state.get("answer", "")
    findings = list(state.get("findings", []))

    # An honest "I don't know" is not a hallucination; don't retry it.
    if answer.strip() == NO_ANSWER:
        return {
            "verification": Verification(
                grounded=True, relevant=True, score=1.0,
                reason="correctly declined to answer without evidence",
            ).model_dump(),
            "retry_count": state.get("retry_count", 0),
            "trace": ["verifier:abstained"],
        }

    verification = _grade_llm(question, answer, findings)
    mode = "llm"
    if verification is None:
        verification = grade_grounded_heuristic(answer, findings)
        mode = "heuristic"

    # Relevance: does the answer engage with the question at all?
    if verification.relevant and overlap_ratio(question, answer) < RELEVANCE_THRESHOLD:
        verification.relevant = False
        verification.reason += "; answer does not address the question"

    return {
        "verification": verification.model_dump(),
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": [f"verifier:{mode}"],
    }


def should_retry(state: AnalystState) -> str:
    """Conditional edge: re-synthesize once when the grade failed."""
    verification = state.get("verification", {}) or {}
    passed = bool(verification.get("grounded")) and bool(verification.get("relevant", True))
    if passed:
        return "done"
    if state.get("retry_count", 0) < MAX_SYNTHESIS_RETRIES:
        return "retry"
    return "done"
