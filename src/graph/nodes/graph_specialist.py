"""Knowledge-graph specialist — entity relationships and multi-hop reasoning.

Handles the causal and comparative questions a single vector lookup cannot:
"why did X fall", "how does A relate to B".  Prefers the existing
``MultiHopRetriever`` (iterative retrieval following entity chains) and falls
back to a single ``GraphRetriever`` augmentation pass.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.adapters import documents_to_findings, to_documents
from src.graph.state import AnalystState, Finding
from src.graph.textutil import tokenize

logger = logging.getLogger(__name__)

MAX_HOPS = 3
MAX_KG_FINDINGS = 6
_MIN_TERM_LEN = 3


def kg_findings(graph_retriever: Any, question: str) -> list:
    """Evidence read straight out of the knowledge graph.

    ``GraphRetriever._extract_query_entities`` only recognises capitalised
    names and ID patterns, so a question like "Why are there dispatch delays?"
    matches no entity and returns nothing.  Matching the question's content
    terms against the edge text instead -- subject, predicate *and* object --
    lets "delays" reach a ``delayed_because`` edge, which is exactly where the
    causal answer lives.
    """
    graph = getattr(graph_retriever, "graph", None)
    if graph is None or graph.number_of_edges() == 0:
        return []

    terms = {t for t in tokenize(question) if len(t) >= _MIN_TERM_LEN}
    if not terms:
        return []

    scored = []
    for subject, obj, data in graph.edges(data=True):
        predicate = str(data.get("predicate", "related_to"))
        sentence = f"{subject} {predicate} {obj}".replace("_", " ")
        hits = len(terms & set(tokenize(sentence)))
        if hits:
            scored.append((hits, sentence, subject, predicate))

    scored.sort(key=lambda row: -row[0])
    findings = []
    for hits, sentence, subject, predicate in scored[:MAX_KG_FINDINGS]:
        findings.append(Finding(
            specialist="graph",
            content=sentence,
            score=min(1.0, hits / max(1, len(terms))),
            source="knowledge_graph",
            doc_id=str(subject),
            modality="text",
            metadata={"predicate": predicate, "hits": hits},
        ))
    return findings


def graph_node(state: AnalystState, components: Any) -> Dict[str, Any]:
    """Traverse entity relationships to gather multi-hop evidence."""
    question = state.get("question", "")
    findings: list = []
    trace: list = []

    graph_retriever = getattr(components, "graph", None)

    # 1. The knowledge graph itself.  This runs *first* and its results are
    #    kept: multi-hop retrieval searches the vector index, so letting it
    #    return early (as it used to) meant the graph was never consulted.
    if graph_retriever is not None:
        try:
            kg = kg_findings(graph_retriever, question)
        except Exception as exc:
            logger.warning("Knowledge-graph lookup failed: %s", exc)
            kg = []
        if kg:
            findings.extend(kg)
            trace.append(f"graph:kg({len(kg)})")
        else:
            # Retrievers that return standalone results from augment_retrieval.
            try:
                augmented = graph_retriever.augment_retrieval(question, [])
                extra = documents_to_findings(to_documents(augmented), "graph")
            except Exception as exc:
                logger.warning("Graph augmentation failed: %s", exc)
                extra = []
            if extra:
                findings.extend(extra)
                trace.append(f"graph:augment({len(extra)})")

    # 2. Multi-hop passages, merged rather than preferred.
    multi_hop = getattr(components, "multi_hop", None)
    if multi_hop is not None:
        try:
            result = multi_hop.retrieve(question, max_hops=MAX_HOPS)
            hopped = documents_to_findings(to_documents(result), "graph")
            if hopped:
                findings.extend(hopped)
                hops = len(result.get("hops", [])) if isinstance(result, dict) else 0
                trace.append(f"graph:multi_hop({hops}hops,{len(hopped)})")
        except Exception as exc:
            logger.warning("Multi-hop retrieval failed: %s", exc)

    if not findings:
        return {"findings": [],
                "trace": ["graph:no_entities" if graph_retriever is not None
                          else "graph:unavailable"]}

    return {"findings": findings, "trace": trace}
