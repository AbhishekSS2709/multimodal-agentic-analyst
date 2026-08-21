"""LangGraph agentic analyst layer.

A supervisor-orchestrated multi-agent graph built *on top of* the existing
retrieval stack.  Nothing under ``src/ingestion``, ``src/retrieval``,
``src/embedding``, ``src/knowledge_graph``, ``src/sql_tool`` or ``src/gemini``
is modified — those components are wrapped by :mod:`src.graph.adapters`.

The original :class:`src.pipeline_orchestrator.EnterpriseRAGOrchestrator`
keeps working unchanged; this package is a parallel front door.
"""

__all__ = ["state", "schemas", "llm", "observability", "adapters", "build"]
