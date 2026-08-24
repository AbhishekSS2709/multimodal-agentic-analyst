"""Knowledge graph builder — pattern-based triple extraction and NetworkX storage.

Extracts structured (subject, predicate, object) triples from enterprise text
using regex patterns, builds them into a NetworkX directed graph, and provides
persistence and visualisation utilities.
"""

from __future__ import annotations

import logging
import pickle
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity and relationship type enums (plain strings for flexibility)
# ---------------------------------------------------------------------------

ENTITY_TYPES: Set[str] = {
    "Supplier", "Machine", "Order", "Product", "Location", "Person",
}

RELATIONSHIP_TYPES: Set[str] = {
    "delayed", "caused", "supplied", "failed", "resolved",
    "escalated", "shipped_to", "failed_due_to", "cancelled_because",
    "produced", "maintained", "reported", "affected", "located_in",
}


# ---------------------------------------------------------------------------
# Triple dataclass
# ---------------------------------------------------------------------------

@dataclass
class Triple:
    """A single (subject, predicate, object) fact with optional metadata."""

    subject: str
    predicate: str
    object: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    triple_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def as_tuple(self) -> Tuple[str, str, str, Dict[str, Any]]:
        return (self.subject, self.predicate, self.object, self.metadata)


# ---------------------------------------------------------------------------
# Pattern-based extraction rules
# ---------------------------------------------------------------------------

@dataclass
class _ExtractionRule:
    """One regex rule that, when matched, produces a Triple."""

    pattern: re.Pattern[str]
    subject_group: int
    predicate: str
    object_group: int
    subject_type: str = "Unknown"
    object_type: str = "Unknown"


# Pre-compiled extraction rules — order matters (first match wins per span).
_RULES: List[_ExtractionRule] = [
    # "Supplier X delayed shipment Y"
    _ExtractionRule(
        pattern=re.compile(
            r"(?:Supplier|Vendor|Manufacturer)\s+([A-Z][A-Za-z0-9&\s\-]{1,40}?)\s+"
            r"delayed\s+(?:shipment|delivery|order)\s+([A-Z0-9\-#]{3,20})",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="delayed",
        object_group=2,
        subject_type="Supplier",
        object_type="Order",
    ),
    # "Supplier X delayed shipment/delivery (no explicit ID)"
    _ExtractionRule(
        pattern=re.compile(
            r"(?:Supplier|Vendor)\s+([A-Z][A-Za-z0-9&\s\-]{1,40}?)\s+"
            r"delayed\s+(shipment|delivery|production|order(?:\s+\S+)?)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="delayed",
        object_group=2,
        subject_type="Supplier",
        object_type="Order",
    ),
    # "Machine Z failed due to part shortage / reason"
    _ExtractionRule(
        pattern=re.compile(
            r"(?:Machine|Equipment|Unit)\s+([A-Z0-9\-#]{2,20})\s+"
            r"failed\s+due\s+to\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="failed_due_to",
        object_group=2,
        subject_type="Machine",
        object_type="Product",
    ),
    # "Machine Z failed on DATE"
    _ExtractionRule(
        pattern=re.compile(
            r"(?:Machine|Equipment|Unit)\s+([A-Z0-9\-#]{2,20})\s+"
            r"failed\s+(?:on|at|during)\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="failed",
        object_group=2,
        subject_type="Machine",
        object_type="Order",
    ),
    # "Order was cancelled because supplier ..."
    _ExtractionRule(
        pattern=re.compile(
            r"(?:Order|PO|SO)\s+([A-Z0-9\-#]{3,20})\s+"
            r"(?:was\s+)?cancelled\s+because\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="cancelled_because",
        object_group=2,
        subject_type="Order",
        object_type="Supplier",
    ),
    # "X supplied Y"
    _ExtractionRule(
        pattern=re.compile(
            r"(?:Supplier|Vendor)\s+([A-Z][A-Za-z0-9&\s\-]{1,40}?)\s+"
            r"supplied\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="supplied",
        object_group=2,
        subject_type="Supplier",
        object_type="Product",
    ),
    # "X shipped to Y"
    _ExtractionRule(
        pattern=re.compile(
            r"([A-Z][A-Za-z0-9&\s\-]{1,40}?)\s+"
            r"shipped\s+to\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="shipped_to",
        object_group=2,
        subject_type="Supplier",
        object_type="Location",
    ),
    # "X caused Y"
    _ExtractionRule(
        pattern=re.compile(
            r"([A-Z][A-Za-z0-9&\s\-]{1,40}?)\s+"
            r"caused\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="caused",
        object_group=2,
        subject_type="Supplier",
        object_type="Order",
    ),
    # "X resolved Y" / "X resolved the issue"
    _ExtractionRule(
        pattern=re.compile(
            r"([A-Z][A-Za-z0-9&\s\-]{1,40}?)\s+"
            r"resolved\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="resolved",
        object_group=2,
        subject_type="Person",
        object_type="Order",
    ),
    # "X escalated to Y"
    _ExtractionRule(
        pattern=re.compile(
            r"([A-Z][A-Za-z0-9&\s\-#]{1,40}?)\s+"
            r"(?:was\s+)?escalated\s+to\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="escalated",
        object_group=2,
        subject_type="Order",
        object_type="Person",
    ),
    # "X located in Y" / "X is in Y"
    _ExtractionRule(
        pattern=re.compile(
            r"([A-Z][A-Za-z0-9&\s\-]{1,40}?)\s+"
            r"(?:is\s+)?(?:located\s+in|based\s+in|situated\s+in)\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        subject_group=1,
        predicate="located_in",
        object_group=2,
        subject_type="Supplier",
        object_type="Location",
    ),
]


# ---------------------------------------------------------------------------
# KnowledgeGraphBuilder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Structured-record extraction
# ---------------------------------------------------------------------------
#
# The rules above expect prose.  The enterprise corpus is pipe-delimited
# records -- "[ts] DISPATCH-1002 | Supplier: Apex Materials | Status: DELAYED"
# -- so prose rules alone extracted 4 triples from 451 documents and left the
# graph specialist with nothing to traverse.

# "[2023-01-10 07:00] DISPATCH-1002 |"  ->  DISPATCH-1002
# "[2023-03-05 16:45] MACHINE: CNC-Mill-07 |"  ->  CNC-Mill-07
_RECORD_SUBJECT = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)?(?:MACHINE\s*:\s*)?([A-Za-z][A-Za-z0-9\-_./]{2,40}?)\s*\|"
)

_RECORD_FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z ]{1,24}?)\s*:\s*(.+?)\s*$")

# Curated field -> predicate map.  Unmapped fields (ETA, Signed by, Condition,
# timestamps) are dropped on purpose: mapping everything fills the graph with
# junk nodes that dilute retrieval.
_RECORD_ENTITY_FIELDS: Dict[str, str] = {
    "supplier": "supplied_by",
    "destination": "shipped_to",
    "carrier": "carried_by",
    "warehouse": "dispatched_from",
    "location": "located_in",
    "status": "has_status",
    "status update": "has_status",
    "severity": "has_severity",
    "escalation": "escalated_to",
}

# Free-text fields: kept verbatim so the synthesizer can quote them.
_RECORD_TEXT_FIELDS: Dict[str, str] = {
    "reason": "delayed_because",
    "root cause": "failed_due_to",
    "issue": "reported_issue",
    "notes": "note",
    "note": "note",
    "production impact": "impacted",
    "items": "carries_item",
    "total delay": "delayed_by",
}

_MAX_OBJECT_CHARS = 200


def _extract_record_triples(text: str) -> List[Tuple[str, str, str, str, str]]:
    """Triples from pipe-delimited log records.

    Returns ``(subject, predicate, object, subject_type, object_type)``; the
    caller attaches metadata.  Entity fields are normalised the usual way, text
    fields are left readable.
    """
    out: List[Tuple[str, str, str, str, str]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        head = _RECORD_SUBJECT.match(line)
        if not head:
            continue
        subject = _normalise_entity(head.group(1))
        if not subject:
            continue
        subject_type = "Machine" if "MACHINE" in line[:head.end()].upper() else "Order"

        for part in line.split("|")[1:]:
            field = _RECORD_FIELD.match(part)
            if not field:
                continue
            key = field.group(1).strip().lower()
            value = field.group(2).strip()
            if not value or len(value) > _MAX_OBJECT_CHARS:
                continue
            if key in _RECORD_ENTITY_FIELDS:
                out.append((subject, _RECORD_ENTITY_FIELDS[key],
                            _normalise_entity(value), subject_type, "Unknown"))
            elif key in _RECORD_TEXT_FIELDS:
                out.append((subject, _RECORD_TEXT_FIELDS[key],
                            value.rstrip(".,;"), subject_type, "Unknown"))
    return out


def _normalise_entity(name: str) -> str:
    """Normalise an entity name for consistent graph keys."""
    name = name.strip().rstrip(".,;:!?")
    # Collapse whitespace
    name = re.sub(r"\s+", "_", name)
    return name


class KnowledgeGraphBuilder:
    """Build and manage a knowledge graph from enterprise documents.

    The graph is stored as a :class:`networkx.DiGraph` where nodes represent
    entities and edges represent relationships (triples).

    Parameters
    ----------
    graph : nx.DiGraph | None
        Optional pre-existing graph to extend. A fresh graph is created
        when *None*.
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None) -> None:
        self._graph: nx.DiGraph = graph if graph is not None else nx.DiGraph()
        self._triples: List[Triple] = []
        logger.info(
            "KnowledgeGraphBuilder initialised (nodes=%d, edges=%d).",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying NetworkX graph."""
        return self._graph

    @property
    def triples(self) -> List[Triple]:
        """Return all extracted triples."""
        return list(self._triples)

    # ------------------------------------------------------------------
    # Triple extraction
    # ------------------------------------------------------------------

    def extract_triples(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Extract (subject, predicate, object, metadata) triples from *text*.

        Parameters
        ----------
        text : str
            Raw document text to scan.
        metadata : dict | None
            Optional metadata to attach to every extracted triple
            (e.g. source file, page number).

        Returns
        -------
        list[tuple[str, str, str, dict]]
            Each element is ``(subject, predicate, object, metadata_dict)``.
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        results: List[Tuple[str, str, str, Dict[str, Any]]] = []
        # Track matched spans to avoid duplicate extraction
        matched_spans: List[Tuple[int, int]] = []

        for rule in _RULES:
            for match in rule.pattern.finditer(text):
                span = match.span()
                # Skip if this span overlaps with an already-matched span
                if any(
                    s <= span[0] < e or s < span[1] <= e
                    for s, e in matched_spans
                ):
                    continue

                subject_raw = match.group(rule.subject_group)
                object_raw = match.group(rule.object_group)

                if not subject_raw or not object_raw:
                    continue

                subject = _normalise_entity(subject_raw)
                obj = _normalise_entity(object_raw)
                predicate = rule.predicate

                if not subject or not obj:
                    continue

                triple_meta = {
                    **meta,
                    "subject_type": rule.subject_type,
                    "object_type": rule.object_type,
                    "source_span": text[max(0, span[0] - 20): span[1] + 20],
                }

                triple = Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    metadata=triple_meta,
                )
                self._triples.append(triple)
                results.append(triple.as_tuple())
                matched_spans.append(span)

        # Structured records, which the prose rules above cannot see.
        for subj, pred, obj, subj_type, obj_type in _extract_record_triples(text):
            triple = Triple(
                subject=subj,
                predicate=pred,
                object=obj,
                metadata={**meta, "subject_type": subj_type,
                          "object_type": obj_type, "extractor": "record"},
            )
            self._triples.append(triple)
            results.append(triple.as_tuple())

        logger.debug("Extracted %d triple(s) from text of length %d.", len(results), len(text))
        return results

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        documents: Sequence[Dict[str, Any]],
    ) -> nx.DiGraph:
        """Extract triples from all *documents* and build the knowledge graph.

        Parameters
        ----------
        documents : sequence of dict
            Each dict must contain at least a ``"text"`` key. Optional keys
            include ``"doc_id"``, ``"source"``, and ``"metadata"``.

        Returns
        -------
        nx.DiGraph
            The populated knowledge graph.
        """
        total_triples = 0
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue

            doc_meta = {
                "doc_id": doc.get("doc_id", "unknown"),
                "source": doc.get("source", doc.get("metadata", {}).get("source", "unknown")),
            }
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                doc_meta.update(doc["metadata"])

            triples = self.extract_triples(text, metadata=doc_meta)
            total_triples += len(triples)

            for subj, pred, obj, meta in triples:
                self._add_triple_to_graph(subj, pred, obj, meta)

        logger.info(
            "Built knowledge graph: %d nodes, %d edges from %d document(s) "
            "(%d triples extracted).",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            len(documents),
            total_triples,
        )
        return self._graph

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually add a single triple to the graph.

        Parameters
        ----------
        subject : str
            Subject entity name.
        predicate : str
            Relationship type.
        obj : str
            Object entity name.
        metadata : dict | None
            Optional metadata for the edge.
        """
        meta = metadata or {}
        subject = _normalise_entity(subject)
        obj = _normalise_entity(obj)
        triple = Triple(subject=subject, predicate=predicate, object=obj, metadata=meta)
        self._triples.append(triple)
        self._add_triple_to_graph(subject, predicate, obj, meta)

    def _add_triple_to_graph(
        self,
        subject: str,
        predicate: str,
        obj: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Add a single triple as nodes + edge on the internal graph."""
        # Add or update subject node
        if not self._graph.has_node(subject):
            self._graph.add_node(
                subject,
                entity_type=metadata.get("subject_type", "Unknown"),
                mention_count=1,
            )
        else:
            self._graph.nodes[subject]["mention_count"] = (
                self._graph.nodes[subject].get("mention_count", 0) + 1
            )

        # Add or update object node
        if not self._graph.has_node(obj):
            self._graph.add_node(
                obj,
                entity_type=metadata.get("object_type", "Unknown"),
                mention_count=1,
            )
        else:
            self._graph.nodes[obj]["mention_count"] = (
                self._graph.nodes[obj].get("mention_count", 0) + 1
            )

        # Add edge (may create parallel edges if using MultiDiGraph, but
        # DiGraph overwrites — that's fine for our use-case).
        edge_data = {
            "predicate": predicate,
            "weight": 1.0,
            **{k: v for k, v in metadata.items() if k not in ("subject_type", "object_type")},
        }
        if self._graph.has_edge(subject, obj):
            # Increment weight for repeated relationships
            self._graph[subject][obj]["weight"] = (
                self._graph[subject][obj].get("weight", 1.0) + 1.0
            )
            # Append predicate if different
            existing_pred = self._graph[subject][obj].get("predicate", "")
            if predicate not in existing_pred:
                self._graph[subject][obj]["predicate"] = f"{existing_pred},{predicate}"
        else:
            self._graph.add_edge(subject, obj, **edge_data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_graph(self, path: str | Path) -> None:
        """Persist the knowledge graph to *path* using pickle.

        Parameters
        ----------
        path : str | Path
            File path for the pickled graph.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "graph": self._graph,
            "triples": self._triples,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Knowledge graph saved to %s.", path)

    def load_graph(self, path: str | Path) -> nx.DiGraph:
        """Load a previously saved knowledge graph from *path*.

        Parameters
        ----------
        path : str | Path
            Path to a pickled graph file.

        Returns
        -------
        nx.DiGraph
            The loaded graph (also replaces the internal graph).

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {path}")

        with open(path, "rb") as fh:
            state = pickle.load(fh)  # noqa: S301

        self._graph = state["graph"]
        self._triples = state.get("triples", [])
        logger.info(
            "Knowledge graph loaded from %s (%d nodes, %d edges).",
            path,
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )
        return self._graph

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, output_path: str | Path) -> Path:
        """Generate an interactive HTML visualisation of the knowledge graph.

        Uses the `pyvis` library to produce a standalone HTML file.

        Parameters
        ----------
        output_path : str | Path
            Destination file path (should end in ``.html``).

        Returns
        -------
        Path
            Absolute path to the generated HTML file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.error("pyvis is not installed — cannot generate visualisation.")
            raise

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        net = Network(
            height="750px",
            width="100%",
            directed=True,
            notebook=False,
            bgcolor="#ffffff",
            font_color="#333333",
        )

        # Color map for entity types
        _COLORS = {
            "Supplier": "#e74c3c",
            "Machine": "#3498db",
            "Order": "#2ecc71",
            "Product": "#f39c12",
            "Location": "#9b59b6",
            "Person": "#1abc9c",
            "Unknown": "#95a5a6",
        }

        # Add nodes
        for node, data in self._graph.nodes(data=True):
            entity_type = data.get("entity_type", "Unknown")
            mention_count = data.get("mention_count", 1)
            color = _COLORS.get(entity_type, _COLORS["Unknown"])
            size = 15 + min(mention_count * 3, 30)
            net.add_node(
                node,
                label=node.replace("_", " "),
                title=f"{entity_type}\nMentions: {mention_count}",
                color=color,
                size=size,
            )

        # Add edges
        for src, dst, data in self._graph.edges(data=True):
            predicate = data.get("predicate", "related_to")
            weight = data.get("weight", 1.0)
            net.add_edge(
                src,
                dst,
                label=predicate.replace("_", " "),
                title=predicate,
                width=min(weight, 5.0),
                arrows="to",
            )

        # Physics settings for readability
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "centralGravity": 0.3,
                    "springLength": 150,
                    "springConstant": 0.04
                },
                "maxVelocity": 50,
                "minVelocity": 0.1
            },
            "edges": {
                "font": {
                    "size": 10,
                    "align": "middle"
                },
                "smooth": {
                    "type": "continuous"
                }
            },
            "nodes": {
                "font": {
                    "size": 14
                }
            }
        }
        """)

        net.save_graph(str(output_path))
        logger.info("Knowledge graph visualisation saved to %s.", output_path)
        return output_path.resolve()


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    sample_docs = [
        {
            "text": "Supplier Acme Corp delayed shipment ORD-12345 by three weeks. "
                    "Machine MCH-4401 failed due to part shortage from Acme Corp. "
                    "Order ORD-12345 was cancelled because supplier could not fulfil.",
            "doc_id": "doc_001",
            "source": "incident_report.pdf",
        },
        {
            "text": "Vendor BetaTech supplied backup components to resolve the issue. "
                    "BetaTech shipped to Dallas warehouse on 2025-07-01. "
                    "The downtime caused significant production losses.",
            "doc_id": "doc_002",
            "source": "resolution_notes.pdf",
        },
    ]

    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(sample_docs)

    print(f"\nGraph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print("\nNodes:")
    for node, data in graph.nodes(data=True):
        print(f"  {node} ({data})")
    print("\nEdges:")
    for src, dst, data in graph.edges(data=True):
        print(f"  {src} --[{data.get('predicate', '?')}]--> {dst}")

    print("\nAll extracted triples:")
    for t in builder.triples:
        print(f"  ({t.subject}, {t.predicate}, {t.object})")
