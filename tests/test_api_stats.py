"""The sidebar counters must reflect the index the server is actually serving.

A container ships with its corpus pre-indexed, so "documents uploaded through
this server" is always zero there; the demo showed 0 documents beside 498
indexed chunks. Agentic (v2) questions were not counted either.
"""

import unittest
from unittest import mock


class _Store:
    total_vectors = 3

    def stored_chunks(self):
        return [{"source": "contracts.txt"}, {"source": "contracts.txt"},
                {"source": "dispatch_log.txt"}]


class TestStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import src.api.main as main
        cls.main = main
        cls.client = TestClient(main.app)

    def test_documents_count_indexed_sources(self):
        with mock.patch.object(self.main, "_get_vector_store", return_value=_Store()):
            body = self.client.get("/api/stats").json()
        self.assertEqual(body["total_chunks"], 3)
        self.assertEqual(body["total_docs"], 2)

    def test_agentic_queries_are_counted(self):
        before = self.main._state.get("queries_answered", 0)
        fake = {"answer": "a", "thread_id": "t"}
        with mock.patch("src.graph.build.run_query", return_value=fake):
            r = self.client.post("/api/v2/query", json={"question": "q"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.main._state["queries_answered"], before + 1)
