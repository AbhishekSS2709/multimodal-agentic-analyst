"""Graph-augmented retrieval — query the knowledge graph and merge with vector results.

Provides :class:`GraphRetriever` which traverses a NetworkX knowledge graph
to find entities related to a query, builds a textual summary of the
relationships, and augments standard vector-retrieval results with
structured graph context.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import TOP_K

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_query_entity(name: str) -> str:
    """Normalise a user-provided entity name so it can match graph node keys."""
    name = name.strip()
    # Replace spaces / hyphens with underscores to match graph conventions
    name = re.sub(r"[\s\-]+", "_", name)
    return name


def _find_matching_nodes(graph: nx.DiGraph, entity: str) -> List[str]:
    """Find graph nodes that match *entity* (case-insensitive, partial).

    Returns
    -------
    list[str]
        Matching node keys sorted by relevance (exact match first, then
        prefix match, then substring match).
    """
    entity_norm = _normalise_query_entity(entity).lower()
    exact: List[str] = []
    prefix: List[str] = []
    substring: List[str] = []

    for node in graph.nodes:
        node_lower = node.lower()
        if node_lower == entity_norm:
            exact.append(node)
        elif node_lower.startswith(entity_norm):
            prefix.append(node)
        elif entity_norm in node_lower:
            substring.append(node)

    return exact + prefix + substring


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------

class GraphRetriever:
    """Query a knowledge graph and combine graph context with vector retrieval.

    Parameters
    ----------
    graph : nx.DiGraph
        A knowledge graph built by :class:`KnowledgeGraphBuilder`.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        if graph is None:
            raise ValueError("A NetworkX DiGraph is required.")
        self._graph = graph
        logger.info(
            "GraphRetriever initialised (nodes=%d, edges=%d).",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying NetworkX graph."""
        return self._graph

    # ------------------------------------------------------------------
    # Core graph query methods
    # ------------------------------------------------------------------

    def query_graph(
        self,
        entity: str,
        max_depth: int = 2,
    ) -> nx.DiGraph:
        """Return a subgraph of nodes/edges reachable from *entity*.

        Performs a BFS traversal up to *max_depth* hops in both directions
        (successors and predecessors) to capture the full neighbourhood.

        Parameters
        ----------
        entity : str
            The entity name to start traversal from.
        max_depth : int
            Maximum traversal depth (default 2).

        Returns
        -------
        nx.DiGraph
            A subgraph containing *entity* and all related nodes within
            *max_depth* hops, with edge data preserved.
        """
        matching_nodes = _find_matching_nodes(self._graph, entity)
        if not matching_nodes:
            logger.info("Entity '%s' not found in graph.", entity)
            return nx.DiGraph()

        visited: Set[str] = set()
        frontier: Set[str] = set(matching_nodes[:3])  # seed with top matches

        for _ in range(max_depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                # Forward neighbours
                if self._graph.has_node(node):
                    next_frontier.update(self._graph.successors(node))
                    next_frontier.update(self._graph.predecessors(node))
            frontier = next_frontier - visited

        # Include any remaining frontier nodes (they're within max_depth)
        visited.update(frontier)

        # Build the subgraph
        subgraph = self._graph.subgraph(visited).copy()
        logger.debug(
            "query_graph('%s', depth=%d): %d nodes, %d edges.",
            entity, max_depth, subgraph.number_of_nodes(), subgraph.number_of_edges(),
        )
        return subgraph

    def get_entity_context(self, entity: str) -> str:
        """Return a natural-language summary of all relationships for *entity*.

        Parameters
        ----------
        entity : str
            The entity to summarise.

        Returns
        -------
        str
            Human-readable summary of the entity's relationships, or an
            empty string if the entity is not found.
        """
        matching_nodes = _find_matching_nodes(self._graph, entity)
        if not matching_nodes:
            return ""

        lines: List[str] = []
        seen_relations: Set[str] = set()

        for node in matching_nodes[:3]:  # limit to top 3 matches
            node_data = self._graph.nodes.get(node, {})
            entity_type = node_data.get("entity_type", "Unknown")
            mention_count = node_data.get("mention_count", 1)
            display_name = node.replace("_", " ")

            lines.append(
                f"{display_name} (type: {entity_type}, mentions: {mention_count}):"
            )

            # Outgoing edges
            if self._graph.has_node(node):
                for _, target, data in self._graph.out_edges(node, data=True):
                    predicate = data.get("predicate", "related_to")
                    relation_key = f"{node}-{predicate}-{target}"
                    if relation_key not in seen_relations:
                        seen_relations.add(relation_key)
                        target_display = target.replace("_", " ")
                        lines.append(
                            f"  -> {predicate.replace('_', ' ')} {target_display}"
                        )

                # Incoming edges
                for source, _, data in self._graph.in_edges(node, data=True):
                    predicate = data.get("predicate", "related_to")
                    relation_key = f"{source}-{predicate}-{node}"
                    if relation_key not in seen_relations:
                        seen_relations.add(relation_key)
                        source_display = source.replace("_", " ")
                        lines.append(
                            f"  <- {source_display} {predicate.replace('_', ' ')}"
                        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Augmented retrieval
    # ------------------------------------------------------------------

    def augment_retrieval(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enhance vector retrieval results with knowledge-graph context.

        Extracts entity names from the *query*, looks them up in the graph,
        and appends a ``graph_context`` field to each matching result. Also
        adds a synthetic "graph summary" result at the end when graph data
        is available.

        Parameters
        ----------
        query : str
            The user's natural-language query.
        vector_results : list[dict]
            Results from a vector / BM25 retriever. Each dict must have a
            ``"text"`` key.

        Returns
        -------
        list[dict]
            The same results list, potentially enriched with ``graph_context``
            fields, plus an optional graph-summary entry appended.
        """
        if not query or not vector_results:
            return vector_results

        # Extract candidate entity names from the query
        query_entities = self._extract_query_entities(query)
        if not query_entities:
            logger.debug("No recognisable entities in query — returning vector results as-is.")
            return vector_results

        # Gather graph context for each entity
        graph_contexts: Dict[str, str] = {}
        subgraph_nodes: Set[str] = set()

        for entity in query_entities:
            context = self.get_entity_context(entity)
            if context:
                graph_contexts[entity] = context

            sub = self.query_graph(entity, max_depth=2)
            subgraph_nodes.update(sub.nodes)

        # Enrich vector results: add graph context when a result's text
        # mentions any of the entities we found in the graph
        enriched = []
        for result in vector_results:
            result = dict(result)  # shallow copy to avoid mutating originals
            text_lower = result.get("text", "").lower()
            related_contexts: List[str] = []

            for entity, ctx in graph_contexts.items():
                # Check if the entity (normalised) appears in the chunk text
                entity_variants = [
                    entity.lower(),
                    entity.replace("_", " ").lower(),
                    entity.replace(" ", "_").lower(),
                ]
                if any(v in text_lower for v in entity_variants):
                    related_contexts.append(ctx)

            if related_contexts:
                result["graph_context"] = "\n---\n".join(related_contexts)

            enriched.append(result)

        # Append a synthetic graph-summary chunk if we gathered any context
        if graph_contexts:
            summary_lines = ["[Knowledge Graph Context]"]
            for entity, ctx in graph_contexts.items():
                summary_lines.append(f"\n{ctx}")

            enriched.append({
                "text": "\n".join(summary_lines),
                "metadata": {
                    "source": "knowledge_graph",
                    "type": "graph_summary",
                    "entities_queried": list(graph_contexts.keys()),
                },
                "doc_id": "graph_summary",
                "score": 0.5,  # moderate relevance by default
                "is_graph_context": True,
            })

        logger.info(
            "augment_retrieval: found %d entity context(s) for query '%s'.",
            len(graph_contexts),
            query[:80],
        )
        return enriched

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_query_entities(self, query: str) -> List[str]:
        """Pull potential entity names from a user query.

        Uses simple heuristics:
        - Capitalised multi-word phrases
        - Known ID patterns (ORD-xxxx, MCH-xxxx, etc.)
        - Individual capitalised words longer than 2 characters

        Parameters
        ----------
        query : str
            Raw user query.

        Returns
        -------
        list[str]
            Candidate entity names (deduplicated).
        """
        entities: List[str] = []
        seen: Set[str] = set()

        def _add(e: str) -> None:
            key = e.lower().strip()
            if key and key not in seen and len(key) > 1:
                seen.add(key)
                entities.append(e.strip())

        # ID patterns (ORD-xxxx, MCH-xxxx, etc.)
        for match in re.finditer(r"\b(?:ORD|PO|SO|INV|MCH|MACH|EQP|LINE|UNIT)[-#]?\d{2,10}\b", query, re.IGNORECASE):
            _add(match.group(0))

        # Capitalised multi-word phrases (e.g. "Acme Corp")
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
            phrase = match.group(0)
            # Skip common English phrases that aren't entities
            if phrase.lower() not in {"the", "what", "which", "where", "when", "how"}:
                _add(phrase)

        # Individual capitalised words (likely proper nouns) — only if 3+ chars
        for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", query):
            word = match.group(0)
            if word.lower() not in {
                "the", "what", "which", "where", "when", "how", "why",
                "who", "did", "does", "was", "were", "are", "has", "had",
                "can", "could", "would", "should", "most", "many", "much",
            }:
                _add(word)

        # Also try splitting quoted phrases
        for match in re.finditer(r'"([^"]{2,60})"', query):
            _add(match.group(1))

        # Also scan for node keys that appear in the query directly
        query_lower = query.lower()
        for node in self._graph.nodes:
            node_variants = [node.lower(), node.replace("_", " ").lower()]
            if any(v in query_lower for v in node_variants):
                _add(node)

        return entities


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    # Build a small demo graph
    G = nx.DiGraph()
    G.add_node("Acme_Corp", entity_type="Supplier", mention_count=5)
    G.add_node("ORD-12345", entity_type="Order", mention_count=3)
    G.add_node("MCH-4401", entity_type="Machine", mention_count=2)
    G.add_node("part_shortage", entity_type="Product", mention_count=2)
    G.add_node("Dallas", entity_type="Location", mention_count=2)
    G.add_node("BetaTech", entity_type="Supplier", mention_count=1)

    G.add_edge("Acme_Corp", "ORD-12345", predicate="delayed", weight=2.0)
    G.add_edge("MCH-4401", "part_shortage", predicate="failed_due_to", weight=1.0)
    G.add_edge("Acme_Corp", "Dallas", predicate="located_in", weight=1.0)
    G.add_edge("BetaTech", "MCH-4401", predicate="resolved", weight=1.0)

    retriever = GraphRetriever(G)

    # Test query_graph
    print("=== Subgraph for 'Acme Corp' (depth=2) ===")
    sub = retriever.query_graph("Acme Corp", max_depth=2)
    for src, dst, data in sub.edges(data=True):
        print(f"  {src} --[{data.get('predicate')}]--> {dst}")

    # Test get_entity_context
    print("\n=== Entity context for 'Acme Corp' ===")
    ctx = retriever.get_entity_context("Acme Corp")
    print(ctx)

    # Test augment_retrieval
    print("\n=== Augmented retrieval ===")
    fake_vector_results = [
        {"text": "Acme Corp delayed shipment ORD-12345 by 3 weeks.", "score": 0.9},
        {"text": "Production line was down for 48 hours.", "score": 0.7},
    ]
    augmented = retriever.augment_retrieval(
        "Which supplier caused delays?",
        fake_vector_results,
    )
    for r in augmented:
        gc = r.get("graph_context", "")
        is_graph = r.get("is_graph_context", False)
        print(f"  text={r['text'][:60]}... | graph_context={'yes' if gc else 'no'} | is_graph={is_graph}")
