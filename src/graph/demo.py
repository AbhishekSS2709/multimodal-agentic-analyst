"""CLI demo — stream the analyst graph node by node.

Usage::

    python -m src.graph.demo "why are there dispatch delays?"
    python -m src.graph.demo --no-approval "how many orders were placed?"

Streaming makes the orchestration visible: you watch the supervisor choose
specialists, the specialists run in parallel, the document subgraph rewrite
and retry, and the verifier grade the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.graph.build import build_analyst_graph, get_checkpointer
from src.graph.llm import llm_mode
from src.graph.observability import configure_tracing, tracing_enabled
from src.graph.state import new_state

_BAR = "-" * 72


def format_update(node: str, update: Dict[str, Any]) -> str:
    """Render one streamed node update as a single readable line."""
    parts = []
    if update.get("specialists"):
        parts.append(f"specialists={update['specialists']}")
    if update.get("findings"):
        parts.append(f"findings=+{len(update['findings'])}")
    if update.get("answer"):
        parts.append(f"answer={len(update['answer'])}chars")
    if update.get("verification"):
        v = update["verification"]
        parts.append(f"grounded={v.get('grounded')} score={v.get('score')}")
    if update.get("trace"):
        parts.append(f"trace={update['trace']}")
    detail = "  ".join(parts) if parts else "(no state change)"
    return f"  {node:<14} {detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic analyst graph.")
    parser.add_argument("question", help="Question to analyse.")
    parser.add_argument("--thread-id", default="demo", help="Conversation thread id.")
    parser.add_argument("--no-approval", action="store_true",
                        help="Skip the human approval gate for write-shaped SQL.")
    args = parser.parse_args(argv)

    configure_tracing()
    print(_BAR)
    print(f"Question : {args.question}")
    print(f"LLM mode : {llm_mode()}    LangSmith tracing: {tracing_enabled()}")
    print(_BAR)

    graph = build_analyst_graph(checkpointer=get_checkpointer())
    config = {"configurable": {"thread_id": args.thread_id,
                               "require_approval": not args.no_approval}}

    final: Dict[str, Any] = {}
    for chunk in graph.stream(new_state(args.question), config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                print(f"\n  PAUSED for human approval: {update}")
                print("  Resume with: POST /api/v2/resume "
                      f'{{"thread_id": "{args.thread_id}", "decision": "approve"}}')
                return 2
            print(format_update(node, update or {}))
            final.update(update or {})

    print(_BAR)
    print("ANSWER\n")
    print(final.get("answer", "(no answer)"))

    citations = final.get("citations") or []
    if citations:
        print("\nSOURCES")
        for i, c in enumerate(citations, 1):
            source = c.source if hasattr(c, "source") else c.get("source", "?")
            print(f"  [{i}] {source}")

    verification = final.get("verification") or {}
    if verification:
        print(f"\nVERIFICATION  grounded={verification.get('grounded')} "
              f"score={verification.get('score')} — {verification.get('reason', '')}")
    print(_BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
