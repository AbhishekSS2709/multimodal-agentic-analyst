"""Document specialist — corrective RAG as a cyclic subgraph.

    retrieve -> grade -+- relevant ------------> emit
                       +- weak & budget left --> rewrite -> retrieve
                       +- weak & exhausted ----> emit_low_confidence

The cycle is the point: a single-shot retriever silently returns its best bad
match, whereas this one notices the match is bad, reformulates, and tries
again, then admits low confidence rather than pretending.
"""

from __future__ import annotations

import logging
import operator
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.adapters import ChunkRetriever, documents_to_findings
from src.graph.llm import get_structured_llm
from src.graph.prompts import GRADE_DOCUMENTS_PROMPT, REWRITE_QUERY_PROMPT
from src.graph.schemas import GradeDocuments, RewrittenQuery
from src.graph.state import AnalystState, Finding
from src.graph.textutil import overlap_ratio

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.30
DEFAULT_MAX_RETRIES = 2
TOP_K = 5

# Reformulation strategies, applied in order across retry attempts.
_REWRITE_TEMPLATES = (
    "{q} definition policy details documentation",
    "{q} overview summary explanation guidelines requirements",
    "information about {q}",
)


class DocState(TypedDict):
    """State internal to the document subgraph."""

    question: str
    query: str
    documents: List[Document]
    findings: Annotated[List[Finding], operator.add]
    retries: int
    trace: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_documents_heuristic(
    subtask: str,
    docs: List[Document],
) -> List[Tuple[Document, GradeDocuments]]:
    """Grade by how much of the sub-task's vocabulary the document covers."""
    graded: List[Tuple[Document, GradeDocuments]] = []
    for doc in docs:
        score = overlap_ratio(subtask, doc.page_content)
        graded.append((doc, GradeDocuments(
            relevant=score >= RELEVANCE_THRESHOLD,
            score=round(score, 3),
            reason=f"{score:.0%} of sub-task terms present",
        )))
    return graded


def _grade_documents_llm(
    subtask: str,
    docs: List[Document],
) -> List[Tuple[Document, GradeDocuments]] | None:
    """Per-document LLM relevance grades; ``None`` when unavailable."""
    structured = get_structured_llm(GradeDocuments)
    if structured is None:
        return None
    graded: List[Tuple[Document, GradeDocuments]] = []
    try:
        for doc in docs:
            grade = structured.invoke(GRADE_DOCUMENTS_PROMPT.format(
                subtask=subtask, document=doc.page_content[:2000],
            ))
            graded.append((doc, grade))
    except Exception as exc:
        logger.warning("LLM grading failed, falling back: %s", exc)
        return None
    return graded


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

def rewrite_query_heuristic(question: str, attempt: int) -> str:
    """Expand the query with a different strategy per attempt."""
    template = _REWRITE_TEMPLATES[(attempt - 1) % len(_REWRITE_TEMPLATES)]
    return template.format(q=question.strip().rstrip("?"))


def _rewrite_query_llm(question: str, query: str, attempt: int) -> str | None:
    structured = get_structured_llm(RewrittenQuery)
    if structured is None:
        return None
    try:
        result = structured.invoke(REWRITE_QUERY_PROMPT.format(
            question=question, query=query, attempt=attempt,
        ))
        rewritten = (result.query or "").strip()
        return rewritten or None
    except Exception as exc:
        logger.warning("LLM rewrite failed, falling back: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------

def build_document_subgraph(retriever: Any, max_retries: int = DEFAULT_MAX_RETRIES):
    """Compile the corrective-RAG subgraph around ``retriever``.

    ``retriever`` is anything with ``invoke(query) -> list[Document]``, which
    is what :class:`~src.graph.adapters.ChunkRetriever` provides.
    """

    def retrieve(state: DocState) -> Dict[str, Any]:
        query = state.get("query") or state["question"]
        try:
            docs = retriever.invoke(query)
        except Exception as exc:
            logger.warning("Document retrieval failed: %s", exc)
            docs = []
        return {"documents": docs, "query": query,
                "trace": [f"document:retrieve({len(docs)})"]}

    def grade(state: DocState) -> Dict[str, Any]:
        docs = state.get("documents", [])
        if not docs:
            return {"documents": [], "trace": ["document:grade(0)"]}

        subtask = state["question"]
        graded = _grade_documents_llm(subtask, docs)
        mode = "llm"
        if graded is None:
            graded = grade_documents_heuristic(subtask, docs)
            mode = "heuristic"

        kept: List[Document] = []
        for doc, grade_result in graded:
            if grade_result.relevant:
                # Carry the grade forward so findings rank on it later.
                doc.metadata["grade"] = grade_result.score
                kept.append(doc)

        return {"documents": kept,
                "trace": [f"document:grade:{mode}({len(kept)}/{len(docs)})"]}

    def decide(state: DocState) -> str:
        if state.get("documents"):
            return "emit"
        if state.get("retries", 0) < max_retries:
            return "rewrite"
        return "low_confidence"

    def rewrite(state: DocState) -> Dict[str, Any]:
        attempt = state.get("retries", 0) + 1
        question = state["question"]
        new_query = _rewrite_query_llm(question, state.get("query", ""), attempt)
        mode = "llm"
        if new_query is None:
            new_query = rewrite_query_heuristic(question, attempt)
            mode = "heuristic"
        return {"query": new_query, "retries": attempt,
                "trace": [f"document:rewrite:{mode}({attempt})"]}

    def emit(state: DocState) -> Dict[str, Any]:
        findings = documents_to_findings(state.get("documents", []), "document")
        return {"findings": findings, "trace": [f"document:emit({len(findings)})"]}

    def low_confidence(state: DocState) -> Dict[str, Any]:
        """Retries exhausted — say so instead of returning a confident wrong answer."""
        return {"findings": [], "trace": ["document:low_confidence"]}

    builder = StateGraph(DocState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("emit", emit)
    builder.add_node("low_confidence", low_confidence)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", decide, {
        "emit": "emit",
        "rewrite": "rewrite",
        "low_confidence": "low_confidence",
    })
    builder.add_edge("rewrite", "retrieve")   # the corrective cycle
    builder.add_edge("emit", END)
    builder.add_edge("low_confidence", END)

    return builder.compile()


def document_node(state: AnalystState, components: Any) -> Dict[str, Any]:
    """Run the corrective-RAG subgraph as one node of the analyst graph."""
    backend = getattr(components, "hybrid", None)
    if backend is None:
        return {"findings": [], "trace": ["document:unavailable"]}

    subgraph = build_document_subgraph(ChunkRetriever(backend=backend, top_k=TOP_K))
    try:
        result = subgraph.invoke({
            "question": state.get("question", ""),
            "query": "", "documents": [], "findings": [], "retries": 0, "trace": [],
        })
    except Exception as exc:
        logger.warning("Document subgraph failed: %s", exc)
        return {"findings": [], "trace": ["document:unavailable"]}

    return {"findings": result.get("findings", []),
            "trace": result.get("trace", [])}
