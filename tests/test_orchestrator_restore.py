"""A process that did not run setup() must still retrieve real passages.

The API server, the CLI demo and a container built from a pre-indexed image
all start with an empty chunk list and an empty BM25 index. Before the
restore path existed, the hybrid retriever returned bare integer indices,
every document was graded irrelevant, and the graph declined every question
while FAISS held the whole corpus.
"""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

DIM = 8
TEXTS = [
    "Dispatch delays at the Chicago plant were caused by a conveyor failure.",
    "Supplier contract GTS-2022-001 sets a late-delivery penalty of 2 percent.",
    "Quarterly revenue rose in the North region.",
]


def _vec(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class _Chunk:
    def __init__(self, i, text):
        self.chunk_id = f"c{i}"
        self.doc_id = f"d{i}"
        self.text = text
        self.token_count = len(text.split())
        self.metadata = {"source": f"doc{i}.txt"}


class _FakeEmbedder:
    """Embeds the query onto the axis of the first text sharing a word."""

    def embed_query(self, query):
        words = set(query.lower().split())
        for i, text in enumerate(TEXTS):
            if words & set(text.lower().split()):
                return _vec(i)
        return _vec(DIM - 1)


class TestRestoreInFreshProcess(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

        from src.embedding.store_vector_db import VectorStore

        store = VectorStore(dimension=DIM, index_path=self.tmp / "i.index",
                            meta_path=self.tmp / "i_meta.json")
        store.store_embeddings([_Chunk(i, t) for i, t in enumerate(TEXTS)],
                               [_vec(i) for i in range(len(TEXTS))])
        store.save()

        graph = nx.DiGraph()
        graph.add_edge("GlobalTech Supply", "Chicago plant", relation="supplies")
        with open(self.tmp / "knowledge_graph.pkl", "wb") as fh:
            pickle.dump({"graph": graph, "triples": []}, fh)

    def tearDown(self):
        self._tmp.cleanup()

    def _fresh_orchestrator(self):
        from src.embedding.store_vector_db import VectorStore
        from src.pipeline_orchestrator import EnterpriseRAGOrchestrator

        orch = EnterpriseRAGOrchestrator()
        orch._embedding_engine = _FakeEmbedder()
        orch._vector_store = VectorStore(dimension=DIM,
                                         index_path=self.tmp / "i.index",
                                         meta_path=self.tmp / "i_meta.json")
        return orch

    def test_hybrid_retriever_returns_passages_not_indices(self):
        orch = self._fresh_orchestrator()
        results = orch._get_hybrid_retriever().retrieve("dispatch delays", top_k=2)

        self.assertTrue(results)
        top = results[0][0]
        self.assertFalse(isinstance(top, int), "got a bare index, not a chunk")
        self.assertIn("Dispatch delays", top.text)

    def test_bm25_is_rebuilt_in_vector_order(self):
        orch = self._fresh_orchestrator()
        orch._get_hybrid_retriever()
        bm25 = orch._get_bm25_search()

        self.assertEqual(bm25.corpus_size, len(TEXTS))
        idx, _ = bm25.search("penalty", top_k=1)[0]
        self.assertEqual(orch._chunks[idx].chunk_id, "c1")

    def test_knowledge_graph_is_loaded_from_disk(self):
        with patch("src.pipeline_orchestrator.VECTOR_DB_DIR", self.tmp):
            orch = self._fresh_orchestrator()
            builder = orch._get_knowledge_graph_builder()
        self.assertEqual(builder.graph.number_of_nodes(), 2)
        self.assertIsNotNone(orch._get_graph_retriever())

    def test_restore_is_a_no_op_after_setup(self):
        orch = self._fresh_orchestrator()
        sentinel = [_Chunk(9, "already built in memory")]
        orch._chunks = sentinel
        orch._restore_chunks_from_store()
        self.assertIs(orch._chunks, sentinel)
