"""Visual specialist — search images, diagrams, slides, and video frames.

Wraps the existing ``MultimodalRetriever`` (CLIP visual index fused with the
text index) and tags everything it returns with the modality so the synthesizer
and the ``modality_match`` evaluator can tell visual evidence apart.
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

TOP_K = 5


def visual_node(state: AnalystState, components: Any) -> Dict[str, Any]:
    """Retrieve visual evidence for the question."""
    backend = getattr(components, "multimodal", None)
    if backend is None:
        return {"findings": [], "trace": ["visual:unavailable"]}

    question = state.get("question", "")
    try:
        results = backend.retrieve(question, top_k=TOP_K)
    except Exception as exc:
        logger.warning("Visual retrieval failed: %s", exc)
        return {"findings": [], "trace": ["visual:unavailable"]}

    docs = to_documents(results)
    findings = documents_to_findings(docs, "visual")

    # The multimodal retriever returns both text and visual hits; this
    # specialist only contributes the visual ones — the document specialist
    # already covers text, and duplicates would double-weight the evidence.
    visual_findings = [f for f in findings if f.modality != "text"]
    if not visual_findings:
        return {"findings": [], "trace": [f"visual:no_visual_hits({len(findings)})"]}

    return {"findings": visual_findings,
            "trace": [f"visual:emit({len(visual_findings)})"]}
