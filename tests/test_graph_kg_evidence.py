"""The graph specialist must actually read the knowledge graph.

It previously called ``augment_retrieval(question, [])``, which returns its
second argument unchanged when the vector results are empty -- so it always got
``[]`` -- and only reached that path at all when multi-hop retrieval (which
runs on the *vector* index, not the graph) returned nothing.  The knowledge
graph therefore never contributed to an answer, and growing it from 4 to 704
triples changed no score at all.
"""

import unittest

import networkx as nx

from src.graph.state import new_state


def _graph():
    """A few edges shaped like the ones extracted from the dispatch log."""
    g = nx.DiGraph()
    g.add_edge("DISPATCH-1002", "Apex_Materials", predicate="supplied_by")
    g.add_edge("DISPATCH-1002", "DELAYED", predicate="has_status")
    g.add_edge("DISPATCH-1002",
               "Apex Materials warehouse reported inventory discrepancy",
               predicate="delayed_because")
    g.add_edge("CNC-Mill-07", "Chicago_Plant", predicate="located_in")
    g.add_edge("DISPATCH-1003", "Reliable_Transport", predicate="carried_by")
    return g


class _GraphRetriever:
    def __init__(self, graph):
        self.graph = graph

    def augment_retrieval(self, query, vector_results):
        # Faithful to the real implementation: a no-op on empty input.
        if not query or not vector_results:
            return vector_results
        return vector_results


class _Components:
    hybrid = multimodal = sql = generator = None
    multi_hop = None

    def __init__(self, graph=None):
        self.graph = _GraphRetriever(graph) if graph is not None else None


class TestKnowledgeGraphEvidence(unittest.TestCase):

    def _run(self, question, graph=None):
        from src.graph.nodes.graph_specialist import graph_node
        return graph_node(new_state(question), _Components(graph))

    def test_causal_question_surfaces_the_delay_reason(self):
        out = self._run("Why are there dispatch delays?", _graph())
        text = " ".join(f.content for f in out["findings"]).lower()
        self.assertTrue(out["findings"], "graph produced no findings")
        self.assertIn("apex materials", text)

    def test_predicate_text_is_matched_not_just_node_names(self):
        """'delays' must reach the `delayed_because` edge."""
        out = self._run("Why are there dispatch delays?", _graph())
        preds = " ".join(f.content for f in out["findings"]).lower()
        self.assertIn("delayed", preds)

    def test_unrelated_question_does_not_dump_the_whole_graph(self):
        out = self._run("what is the refund policy?", _graph())
        self.assertLessEqual(len(out["findings"]), 2)

    def test_findings_are_attributed_to_the_graph_specialist(self):
        out = self._run("Why are there dispatch delays?", _graph())
        self.assertTrue(all(f.specialist == "graph" for f in out["findings"]))

    def test_empty_graph_degrades_quietly(self):
        out = self._run("Why are there dispatch delays?", nx.DiGraph())
        self.assertEqual(out["findings"], [])
        self.assertTrue(out["trace"])

    def test_no_graph_component_degrades_quietly(self):
        out = self._run("Why are there dispatch delays?", None)
        self.assertEqual(out["findings"], [])


class TestMultiHopIsMergedNotPreferred(unittest.TestCase):
    """Multi-hop runs on the vector index; it must not shadow graph evidence."""

    def test_graph_evidence_survives_alongside_multi_hop(self):
        from src.graph.nodes.graph_specialist import graph_node

        class _MH:
            def retrieve(self, query, max_hops=3):
                return {"documents": [{"text": "a passage about shipping",
                                       "doc_id": "d1", "source": "log.txt",
                                       "score": 0.5}],
                        "hops": [1]}

        comp = _Components(_graph())
        comp.multi_hop = _MH()
        out = graph_node(new_state("Why are there dispatch delays?"), comp)
        text = " ".join(f.content for f in out["findings"]).lower()
        self.assertIn("apex materials", text)


if __name__ == "__main__":
    unittest.main()
