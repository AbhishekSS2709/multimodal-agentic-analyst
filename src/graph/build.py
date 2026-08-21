"""Assemble the analyst graph.

Topology:

    START -> supervisor -Send()-> {document, visual, analytics, graph}
                                       |  (all four write `findings`)
                                       v
                                  synthesizer -> verifier -+- done -> END
                                       ^                   |
                                       +----- retry -------+

The fan-out uses ``Send``, so the selected specialists execute in a single
parallel superstep.  Their concurrent writes to ``findings`` survive because
that key carries an ``operator.add`` reducer (see :mod:`src.graph.state`).
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_DIR
from src.graph.adapters import LazyComponents
from src.graph.nodes.analytics import analytics_node
from src.graph.nodes.document import document_node
from src.graph.nodes.graph_specialist import graph_node
from src.graph.nodes.supervisor import supervisor_node
from src.graph.nodes.synthesizer import synthesizer_node
from src.graph.nodes.verifier import should_retry, verifier_node
from src.graph.nodes.visual import visual_node
from src.graph.observability import configure_tracing, run_metadata
from src.graph.state import AnalystState, new_state

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(DATA_DIR) / "graph_checkpoints.db"

_SPECIALIST_NODES = {
    "document": document_node,
    "visual": visual_node,
    "graph": graph_node,
    # analytics is wired separately: it also takes `config` for the HITL gate.
}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_to_specialists(state: AnalystState) -> List[Send]:
    """Fan out to every specialist the supervisor selected."""
    specialists = state.get("specialists") or ["document"]
    return [Send(name, state) for name in specialists]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def get_checkpointer(path: Optional[str] = None) -> Any:
    """Durable SQLite checkpointer, or an in-memory one for tests.

    Pass ``":memory:"`` for a throwaway saver; pass ``None`` for the default
    on-disk database under ``data/``.
    """
    if path == ":memory:":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        target = Path(path) if path else CHECKPOINT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target), check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as exc:
        logger.warning("SQLite checkpointer unavailable (%s); using memory.", exc)
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_analyst_graph(
    components: Any = None,
    checkpointer: Any = None,
    max_retries: int = 2,
) -> Any:
    """Compile the analyst graph against ``components``."""
    if components is None:
        components = LazyComponents.from_orchestrator()

    builder = StateGraph(AnalystState)
    builder.add_node("supervisor", supervisor_node)

    for name, node in _SPECIALIST_NODES.items():
        builder.add_node(
            name,
            (lambda node=node: lambda state: node(state, components))(),
        )

    # analytics needs the RunnableConfig to read `require_approval`.
    builder.add_node(
        "analytics",
        lambda state, config: analytics_node(state, components, config),
    )

    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("verifier", verifier_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_to_specialists,
        ["document", "visual", "analytics", "graph"],
    )
    for name in ("document", "visual", "analytics", "graph"):
        builder.add_edge(name, "synthesizer")

    builder.add_edge("synthesizer", "verifier")
    builder.add_conditional_edges(
        "verifier", should_retry,
        {"retry": "synthesizer", "done": END},
    )

    return builder.compile(checkpointer=checkpointer)


def graph_mermaid() -> str:
    """Mermaid rendering of the topology, for docs and ``/api/v2/graph``."""
    return """graph TD
    START([START]) --> supervisor[supervisor<br/>plan + select specialists]
    supervisor -->|Send| document[document<br/>corrective RAG subgraph]
    supervisor -->|Send| visual[visual<br/>CLIP + vision]
    supervisor -->|Send| analytics[analytics<br/>SQL + HITL gate]
    supervisor -->|Send| graph[graph<br/>knowledge graph multi-hop]
    document --> synthesizer[synthesizer<br/>cited answer]
    visual --> synthesizer
    analytics --> synthesizer
    graph --> synthesizer
    synthesizer --> verifier[verifier<br/>groundedness grade]
    verifier -->|retry| synthesizer
    verifier -->|done| FINISH([END])"""


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def _shape_result(state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    """Normalise a raw graph result into the API/CLI response shape."""
    interrupts = state.get("__interrupt__") or []
    payload = None
    if interrupts:
        first = interrupts[0]
        payload = getattr(first, "value", first)

    def _dump(items: Any) -> List[Dict[str, Any]]:
        out = []
        for item in items or []:
            out.append(item.model_dump() if hasattr(item, "model_dump") else item)
        return out

    return {
        "answer": state.get("answer", ""),
        "citations": _dump(state.get("citations")),
        "findings": _dump(state.get("findings")),
        "verification": state.get("verification", {}),
        "specialists": state.get("specialists", []),
        "retry_count": state.get("retry_count", 0),
        "trace": state.get("trace", []),
        "thread_id": thread_id,
        "interrupted": bool(interrupts),
        "interrupt_payload": payload,
        "metadata": run_metadata(),
    }


_DEFAULT_GRAPH: Dict[str, Any] = {}


def _shared_graph(components: Any = None, checkpointer: Any = None) -> Any:
    """One compiled graph per process, so threads persist across calls."""
    if components is not None or checkpointer is not None:
        return build_analyst_graph(components, checkpointer or get_checkpointer())
    if "graph" not in _DEFAULT_GRAPH:
        configure_tracing()
        _DEFAULT_GRAPH["graph"] = build_analyst_graph(
            None, get_checkpointer()
        )
    return _DEFAULT_GRAPH["graph"]


def run_query(
    question: str,
    thread_id: Optional[str] = None,
    require_approval: bool = True,
    components: Any = None,
    checkpointer: Any = None,
) -> Dict[str, Any]:
    """Run one question through the analyst graph."""
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    graph = _shared_graph(components, checkpointer)
    config = {"configurable": {"thread_id": thread_id,
                               "require_approval": require_approval}}
    result = graph.invoke(new_state(question), config)
    return _shape_result(result, thread_id)


def resume_query(
    thread_id: str,
    decision: str,
    components: Any = None,
    checkpointer: Any = None,
    require_approval: bool = True,
) -> Dict[str, Any]:
    """Resume a thread paused at a human-approval interrupt."""
    graph = _shared_graph(components, checkpointer)
    config = {"configurable": {"thread_id": thread_id,
                               "require_approval": require_approval}}
    result = graph.invoke(Command(resume=decision), config)
    return _shape_result(result, thread_id)
