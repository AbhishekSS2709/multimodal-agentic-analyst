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
from src.graph.state import AnalystState

logger = logging.getLogger(__name__)

MAX_HOPS = 3


def graph_node(state: AnalystState, components: Any) -> Dict[str, Any]:
    """Traverse entity relationships to gather multi-hop evidence."""
    question = state.get("question", "")

    multi_hop = getattr(components, "multi_hop", None)
    if multi_hop is not None:
        try:
            result = multi_hop.retrieve(question, max_hops=MAX_HOPS)
            docs = to_documents(result)
            if docs:
                findings = documents_to_findings(docs, "graph")
                hops = len(result.get("hops", [])) if isinstance(result, dict) else 0
                return {"findings": findings,
                        "trace": [f"graph:multi_hop({hops}hops,{len(findings)})"]}
        except Exception as exc:
            logger.warning("Multi-hop retrieval failed: %s", exc)

    graph_retriever = getattr(components, "graph", None)
    if graph_retriever is None:
        return {"findings": [], "trace": ["graph:unavailable"]}

    try:
        augmented = graph_retriever.augment_retrieval(question, [])
    except Exception as exc:
        logger.warning("Graph augmentation failed: %s", exc)
        return {"findings": [], "trace": ["graph:unavailable"]}

    findings = documents_to_findings(to_documents(augmented), "graph")
    if not findings:
        return {"findings": [], "trace": ["graph:no_entities"]}

    return {"findings": findings, "trace": [f"graph:emit({len(findings)})"]}
