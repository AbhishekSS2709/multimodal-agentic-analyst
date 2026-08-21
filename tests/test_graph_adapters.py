"""Tests for LangChain adapters over existing RAG components."""

import unittest


class _FakeBackend:
    """Minimal stand-in for HybridRetriever/Retriever."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def retrieve(self, query, top_k=5, **kw):
        self.calls.append((query, top_k))
        return self.rows[:top_k]


class _FakeChunk:
    def __init__(self, text, doc_id, metadata=None):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = doc_id + "_0"
        self.metadata = metadata or {"source": doc_id}


class TestToDocuments(unittest.TestCase):

    def test_normalises_dict_results(self):
        from src.graph.adapters import to_documents
        docs = to_documents([
            {"text": "hello", "doc_id": "d1", "source": "a.pdf", "score": 0.7},
        ])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "hello")
        self.assertEqual(docs[0].metadata["source"], "a.pdf")
        self.assertAlmostEqual(docs[0].metadata["score"], 0.7)

    def test_normalises_chunk_score_tuples(self):
        from src.graph.adapters import to_documents
        docs = to_documents([(_FakeChunk("body", "d2"), 0.42)])
        self.assertEqual(docs[0].page_content, "body")
        self.assertEqual(docs[0].metadata["doc_id"], "d2")
        self.assertAlmostEqual(docs[0].metadata["score"], 0.42)

    def test_normalises_hybrid_four_tuples(self):
        """HybridRetriever returns (chunk, fused, vec, bm25)."""
        from src.graph.adapters import to_documents
        docs = to_documents([(_FakeChunk("h", "d3"), 0.9, 0.6, 0.3)])
        self.assertEqual(docs[0].page_content, "h")
        self.assertAlmostEqual(docs[0].metadata["score"], 0.9)

    def test_empty_input(self):
        from src.graph.adapters import to_documents
        self.assertEqual(to_documents([]), [])


class TestChunkRetriever(unittest.TestCase):

    def test_conforms_to_base_retriever(self):
        from langchain_core.retrievers import BaseRetriever
        from src.graph.adapters import ChunkRetriever
        r = ChunkRetriever(backend=_FakeBackend(
            [{"text": "t", "doc_id": "d", "score": 1.0}]))
        self.assertIsInstance(r, BaseRetriever)

    def test_invoke_returns_documents(self):
        from src.graph.adapters import ChunkRetriever
        backend = _FakeBackend([{"text": "t", "doc_id": "d", "score": 1.0}])
        docs = ChunkRetriever(backend=backend, top_k=3).invoke("q")
        self.assertEqual(docs[0].page_content, "t")
        self.assertEqual(backend.calls, [("q", 3)])


class TestDocumentsToFindings(unittest.TestCase):

    def test_maps_metadata_onto_findings(self):
        from langchain_core.documents import Document
        from src.graph.adapters import documents_to_findings
        docs = [Document(page_content="body",
                         metadata={"source": "a.pdf", "doc_id": "d1",
                                   "score": 0.5, "modality": "text"})]
        findings = documents_to_findings(docs, "document")
        self.assertEqual(findings[0].specialist, "document")
        self.assertEqual(findings[0].source, "a.pdf")
        self.assertAlmostEqual(findings[0].score, 0.5)


class TestLazyComponents(unittest.TestCase):

    def test_missing_components_return_none_not_raise(self):
        """A broken/absent orchestrator must degrade, never explode."""
        from src.graph.adapters import LazyComponents

        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("no index built")

        c = LazyComponents(_Broken())
        for attr in ("hybrid", "multimodal", "sql", "graph", "multi_hop", "generator"):
            self.assertIsNone(getattr(c, attr), attr)


if __name__ == "__main__":
    unittest.main()
