"""Expose the existing RAG components to LangChain without rewriting them.

The retrievers in this codebase each return a different shape:

* ``Retriever.retrieve``        -> ``[(Chunk, score)]``
* ``HybridRetriever.retrieve``  -> ``[(Chunk, fused, vector, bm25)]``
* ``MultimodalRetriever``       -> ``[dict]``
* ``MultiHopRetriever``         -> ``{"results": [dict], ...}``

:func:`to_documents` is the one place that knows about all of them.  Everything
downstream of it works in ``Document``/``Finding`` and never has to care.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.graph.state import Finding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------

def _chunk_to_document(chunk: Any, score: float) -> Document:
    """Turn a Chunk-like object into a Document."""
    metadata = dict(getattr(chunk, "metadata", None) or {})
    doc_id = getattr(chunk, "doc_id", "") or metadata.get("doc_id", "")
    return Document(
        page_content=getattr(chunk, "text", "") or str(chunk),
        metadata={
            "source": metadata.get("source", doc_id) or "unknown",
            "doc_id": doc_id,
            "chunk_id": getattr(chunk, "chunk_id", ""),
            "score": float(score),
            "modality": metadata.get("modality", "text"),
            **{k: v for k, v in metadata.items()
               if k not in ("source", "doc_id", "modality")},
        },
    )


def _dict_to_document(row: Dict[str, Any]) -> Document:
    """Turn a dict result (multimodal / multi-hop) into a Document."""
    doc_id = row.get("doc_id", "")
    text = row.get("text") or row.get("content") or row.get("caption") or ""
    known = {"text", "content", "caption", "doc_id", "source", "score", "modality"}
    return Document(
        page_content=text,
        metadata={
            "source": row.get("source") or doc_id or "unknown",
            "doc_id": doc_id,
            "score": float(row.get("score", 0.0) or 0.0),
            "modality": row.get("modality", "text"),
            **{k: v for k, v in row.items() if k not in known},
        },
    )


def to_documents(results: Any) -> List[Document]:
    """Normalise any of this codebase's retrieval shapes into Documents."""
    if not results:
        return []

    # MultiHopRetriever returns a dict wrapper around its results.
    if isinstance(results, dict):
        results = results.get("results", []) or results.get("chunks", [])

    docs: List[Document] = []
    for item in results:
        try:
            if isinstance(item, dict):
                docs.append(_dict_to_document(item))
            elif isinstance(item, (tuple, list)):
                if not item:
                    continue
                # (chunk, score) or (chunk, fused, vector, bm25) — index 1 is
                # the score we rank on in both cases.
                chunk = item[0]
                score = float(item[1]) if len(item) > 1 else 0.0
                docs.append(_chunk_to_document(chunk, score))
            else:
                docs.append(_chunk_to_document(item, 0.0))
        except Exception as exc:
            logger.warning("Skipping unparseable retrieval result: %s", exc)
    return docs


def documents_to_findings(docs: List[Document], specialist: str) -> List[Finding]:
    """Convert Documents into the uniform Finding shape the synthesizer reads."""
    findings: List[Finding] = []
    for d in docs:
        md = d.metadata or {}
        findings.append(Finding(
            specialist=specialist,
            content=d.page_content,
            score=float(md.get("score", 0.0) or 0.0),
            source=str(md.get("source", "unknown")),
            doc_id=str(md.get("doc_id", "")),
            modality=str(md.get("modality", "text")),
            metadata={k: v for k, v in md.items()
                      if k not in ("source", "doc_id", "score", "modality")},
        ))
    return findings


# ---------------------------------------------------------------------------
# LangChain retriever
# ---------------------------------------------------------------------------

class ChunkRetriever(BaseRetriever):
    """Adapt any object with ``retrieve(query, top_k)`` to a LangChain retriever.

    This is what lets the existing HybridRetriever slot into LangChain and
    LangGraph tooling unchanged.
    """

    backend: Any
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,  # type: ignore[assignment]
    ) -> List[Document]:
        try:
            return to_documents(self.backend.retrieve(query, top_k=self.top_k))
        except Exception as exc:
            logger.warning("Retrieval failed for %r: %s", query[:60], exc)
            return []


# ---------------------------------------------------------------------------
# Lazy component access
# ---------------------------------------------------------------------------

class LazyComponents:
    """Holds one orchestrator and hands out its components, or ``None``.

    The orchestrator builds components lazily and raises when an index has not
    been built or a key is missing.  Specialists must degrade rather than
    crash, so every accessor here swallows that and returns ``None``.
    """

    def __init__(self, orchestrator: Any = None) -> None:
        self._orch = orchestrator

    @classmethod
    def from_orchestrator(cls) -> "LazyComponents":
        """Build against the real EnterpriseRAGOrchestrator."""
        try:
            from src.pipeline_orchestrator import EnterpriseRAGOrchestrator

            return cls(EnterpriseRAGOrchestrator())
        except Exception as exc:
            logger.warning("Orchestrator unavailable: %s", exc)
            return cls(None)

    def _get(self, accessor: str) -> Optional[Any]:
        if self._orch is None:
            return None
        try:
            return getattr(self._orch, accessor)()
        except Exception as exc:
            logger.debug("Component %s unavailable: %s", accessor, exc)
            return None

    @property
    def hybrid(self) -> Optional[Any]:
        return self._get("_get_hybrid_retriever") or self._get("_get_retriever")

    @property
    def multimodal(self) -> Optional[Any]:
        return self._get("_get_multimodal_retriever")

    @property
    def sql(self) -> Optional[Any]:
        return self._get("_get_sql_pipeline")

    @property
    def graph(self) -> Optional[Any]:
        return self._get("_get_graph_retriever")

    @property
    def multi_hop(self) -> Optional[Any]:
        return self._get("_get_multi_hop_retriever")

    @property
    def generator(self) -> Optional[Any]:
        return self._get("_get_answer_generator")


# ---------------------------------------------------------------------------
# Tools (for ReAct-style use and for LangSmith tool traces)
# ---------------------------------------------------------------------------

def build_tools(components: LazyComponents) -> List[Any]:
    """Expose the four backends as LangChain tools."""

    @tool
    def search_documents(query: str) -> List[Dict[str, Any]]:
        """Search the text corpus for passages relevant to the query."""
        backend = components.hybrid
        if backend is None:
            return []
        docs = to_documents(backend.retrieve(query, top_k=5))
        return [{"text": d.page_content, **d.metadata} for d in docs]

    @tool
    def search_visuals(query: str) -> List[Dict[str, Any]]:
        """Search images, diagrams, and video frames relevant to the query."""
        backend = components.multimodal
        if backend is None:
            return []
        docs = to_documents(backend.retrieve(query, top_k=5))
        return [{"text": d.page_content, **d.metadata} for d in docs]

    @tool
    def run_analytics(question: str) -> Dict[str, Any]:
        """Answer a quantitative question by querying the SQL warehouse."""
        backend = components.sql
        if backend is None:
            return {}
        return backend.analyze(question)

    @tool
    def search_knowledge_graph(entity: str) -> List[Dict[str, Any]]:
        """Find entities related to the given entity in the knowledge graph."""
        backend = components.graph
        if backend is None:
            return []
        return backend.augment_retrieval(entity, [])

    return [search_documents, search_visuals, run_analytics, search_knowledge_graph]
