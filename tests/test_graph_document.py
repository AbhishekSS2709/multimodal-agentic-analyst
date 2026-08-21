"""Tests for the corrective-RAG document subgraph."""

import unittest

from langchain_core.documents import Document


class _StubRetriever:
    """Returns weak docs until `flip_after` calls, then a strong one."""

    def __init__(self, weak_doc, strong_doc=None, flip_after=99):
        self.weak_doc, self.strong_doc = weak_doc, strong_doc
        self.flip_after, self.calls = flip_after, 0

    def invoke(self, query, config=None):
        self.calls += 1
        if self.strong_doc is not None and self.calls > self.flip_after:
            return [self.strong_doc]
        return [self.weak_doc]


WEAK = Document(page_content="Unrelated text about gardening tools.",
                metadata={"source": "g.pdf", "doc_id": "g1", "score": 0.1})
STRONG = Document(page_content="The refund policy allows returns within 30 days.",
                  metadata={"source": "p.pdf", "doc_id": "p1", "score": 0.9})


def _initial(question="refund policy return window"):
    return {"question": question, "query": "", "documents": [],
            "findings": [], "retries": 0, "trace": []}


class TestGrading(unittest.TestCase):

    def test_relevant_doc_graded_relevant(self):
        from src.graph.nodes.document import grade_documents_heuristic
        graded = grade_documents_heuristic("refund policy return window", [STRONG])
        self.assertTrue(graded[0][1].relevant)

    def test_irrelevant_doc_graded_irrelevant(self):
        from src.graph.nodes.document import grade_documents_heuristic
        graded = grade_documents_heuristic("refund policy return window", [WEAK])
        self.assertFalse(graded[0][1].relevant)


class TestRewrite(unittest.TestCase):

    def test_rewrite_changes_the_query(self):
        from src.graph.nodes.document import rewrite_query_heuristic
        original = "refund policy"
        self.assertNotEqual(rewrite_query_heuristic(original, 1), original)

    def test_rewrites_differ_across_attempts(self):
        from src.graph.nodes.document import rewrite_query_heuristic
        self.assertNotEqual(rewrite_query_heuristic("refund policy", 1),
                            rewrite_query_heuristic("refund policy", 2))


class TestSubgraph(unittest.TestCase):

    def test_strong_docs_need_no_retry(self):
        from src.graph.nodes.document import build_document_subgraph
        r = _StubRetriever(STRONG)
        out = build_document_subgraph(r).invoke(_initial())
        self.assertEqual(r.calls, 1)
        self.assertTrue(out["findings"])

    def test_weak_docs_trigger_rewrite_and_recover(self):
        from src.graph.nodes.document import build_document_subgraph
        r = _StubRetriever(WEAK, STRONG, flip_after=1)
        out = build_document_subgraph(r).invoke(_initial())
        self.assertGreaterEqual(r.calls, 2)
        self.assertTrue(out["findings"])
        self.assertTrue(any("rewrite" in t for t in out["trace"]))

    def test_retry_budget_terminates(self):
        """Permanently weak retrieval must stop, not loop forever."""
        from src.graph.nodes.document import build_document_subgraph
        r = _StubRetriever(WEAK)
        out = build_document_subgraph(r, max_retries=2).invoke(_initial())
        self.assertLessEqual(r.calls, 3)          # initial + 2 retries
        self.assertTrue(any("low_confidence" in t for t in out["trace"]))

    def test_document_node_degrades_without_retriever(self):
        from src.graph.nodes.document import document_node
        from src.graph.state import new_state

        class _NoComponents:
            hybrid = None

        out = document_node(new_state("q"), _NoComponents())
        self.assertEqual(out["findings"], [])
        self.assertTrue(any(t.endswith(":unavailable") for t in out["trace"]))


if __name__ == "__main__":
    unittest.main()
