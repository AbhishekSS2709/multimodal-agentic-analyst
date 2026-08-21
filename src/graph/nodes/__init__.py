"""Nodes of the analyst graph.

Each node is a plain function taking state (and, for specialists, a
:class:`~src.graph.adapters.LazyComponents`) and returning a partial state
update.  Keeping them free functions rather than classes makes them trivially
unit-testable without building the graph.
"""
