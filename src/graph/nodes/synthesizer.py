"""Synthesizer node — merge findings from all specialists into one cited answer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.llm import get_llm
from src.graph.prompts import SYNTHESIZE_PROMPT
from src.graph.state import AnalystState, Citation, Finding
from src.graph.textutil import overlap_ratio

logger = logging.getLogger(__name__)

NO_ANSWER = (
    "I could not find relevant information in the indexed corpus to answer "
    "that question."
)

MAX_FINDINGS_IN_ANSWER = 5
SNIPPET_CHARS = 240


def _rank(question: str, findings: List[Finding]) -> List[Finding]:
    """Order findings by retrieval score blended with question overlap."""
    def key(f: Finding) -> float:
        return 0.7 * float(f.score or 0.0) + 0.3 * overlap_ratio(question, f.content)

    return sorted(findings, key=key, reverse=True)


def format_findings(findings: List[Finding]) -> str:
    """Numbered block used in both the synthesis and verification prompts."""
    return "\n\n".join(
        f"[{i}] ({f.specialist}/{f.modality}, source: {f.source})\n{f.content}"
        for i, f in enumerate(findings, 1)
    )


def synthesize_heuristic(
    question: str,
    findings: List[Finding],
) -> Tuple[str, List[Citation]]:
    """Extractive answer: stitch the top findings together with citation markers.

    Deliberately extractive rather than generative — with no LLM available the
    honest thing is to surface the evidence, not invent prose around it.
    """
    if not findings:
        return NO_ANSWER, []

    top = _rank(question, findings)[:MAX_FINDINGS_IN_ANSWER]

    parts: List[str] = []
    citations: List[Citation] = []
    for i, f in enumerate(top, 1):
        content = f.content.strip()
        if not content:
            continue
        parts.append(f"{content} [{i}]")
        citations.append(Citation(
            source=f.source or "unknown",
            doc_id=f.doc_id,
            snippet=content[:SNIPPET_CHARS],
            specialist=f.specialist,
        ))

    if not parts:
        return NO_ANSWER, []

    return " ".join(parts), citations


def _response_text(response: object) -> str:
    """Text of a chat response, whether ``content`` is a str or a block list.

    Newer Gemini models return a list of content blocks.  Calling ``.strip()``
    on that raised, which the caller caught as "LLM unavailable" and silently
    answered from the heuristic path instead.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content if isinstance(content, str) else str(content)


def _synthesize_llm(
    question: str,
    findings: List[Finding],
) -> Tuple[str, List[Citation]] | None:
    """Generative answer over the findings; ``None`` when no LLM is configured."""
    llm = get_llm()
    if llm is None or not findings:
        return None
    try:
        response = llm.invoke(SYNTHESIZE_PROMPT.format(
            question=question,
            findings=format_findings(findings),
        ))
        answer = _response_text(response).strip()
    except Exception as exc:
        logger.warning("LLM synthesis failed, falling back: %s", exc)
        return None

    if not answer:
        return None

    # Cite every finding the model was shown; the verifier prunes unsupported ones.
    citations = [
        Citation(source=f.source or "unknown", doc_id=f.doc_id,
                 snippet=f.content[:SNIPPET_CHARS], specialist=f.specialist)
        for f in findings[:MAX_FINDINGS_IN_ANSWER]
    ]
    return answer, citations


def synthesizer_node(state: AnalystState) -> Dict[str, Any]:
    """Produce the final answer and its citations."""
    question = state.get("question", "")
    findings = list(state.get("findings", []))

    # On a retry the previous answer failed its groundedness grade, so drop to
    # the extractive path — it quotes the findings, so it is grounded by
    # construction. Without this the cycle would just regenerate the same
    # ungrounded answer.
    retrying = state.get("retry_count", 0) > 0

    result = None if retrying else _synthesize_llm(question, findings)
    mode = "llm"
    if result is None:
        result = synthesize_heuristic(question, findings)
        mode = "extractive_retry" if retrying else "heuristic"

    answer, citations = result
    return {
        "answer": answer,
        "citations": citations,
        "trace": [f"synthesizer:{mode}"],
    }
