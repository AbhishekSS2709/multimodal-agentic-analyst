"""Tests for the v2 agentic API surface.

These cover route wiring, validation, and response shaping.  The graph itself
is exercised end-to-end in ``test_graph_build.py`` with stub components — going
through the real orchestrator here would load the BGE and CLIP models and turn
a route test into a multi-minute integration test.
"""

import unittest
from unittest import mock


def _fake_result(thread_id="t-1", interrupted=False):
    return {
        "answer": "Refunds are issued within 30 days. [1]",
        "citations": [{"source": "policy.pdf", "doc_id": "d1",
                       "snippet": "Refunds...", "specialist": "document"}],
        "findings": [{"specialist": "document", "content": "Refunds...",
                      "score": 0.9, "source": "policy.pdf", "doc_id": "d1",
                      "modality": "text", "metadata": {}}],
        "verification": {"grounded": True, "relevant": True, "score": 0.8,
                         "reason": "supported"},
        "specialists": ["document"],
        "retry_count": 1,
        "trace": ["supervisor:heuristic", "document:emit(1)"],
        "thread_id": thread_id,
        "interrupted": interrupted,
        "interrupt_payload": {"type": "sql_approval"} if interrupted else None,
        "metadata": {"llm_mode": "heuristic", "tracing": False},
    }


class TestV2Routes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from src.api.main import app
        cls.client = TestClient(app)
        cls.app = app

    def test_graph_topology_endpoint(self):
        r = self.client.get("/api/v2/graph")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("supervisor", body["mermaid"])
        self.assertIn(body["llm_mode"], ("llm", "heuristic"))
        self.assertIn("document", body["nodes"])

    def test_query_endpoint_rejects_empty_question(self):
        r = self.client.post("/api/v2/query", json={"question": ""})
        self.assertIn(r.status_code, (400, 422))

    def test_query_endpoint_returns_expected_shape(self):
        with mock.patch("src.graph.build.run_query", return_value=_fake_result()):
            r = self.client.post("/api/v2/query",
                                 json={"question": "what is the refund policy?",
                                       "require_approval": False})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("answer", "citations", "findings", "trace", "thread_id",
                    "interrupted", "verification", "specialists", "retry_count"):
            self.assertIn(key, body)
        self.assertEqual(body["thread_id"], "t-1")

    def test_query_endpoint_surfaces_interrupts(self):
        with mock.patch("src.graph.build.run_query",
                        return_value=_fake_result(interrupted=True)):
            r = self.client.post("/api/v2/query",
                                 json={"question": "how many orders to delete?"})
        self.assertTrue(r.json()["interrupted"])
        self.assertEqual(r.json()["interrupt_payload"]["type"], "sql_approval")

    def test_query_endpoint_reports_graph_failure(self):
        with mock.patch("src.graph.build.run_query", side_effect=RuntimeError("boom")):
            r = self.client.post("/api/v2/query", json={"question": "anything"})
        self.assertEqual(r.status_code, 500)

    def test_resume_endpoint_passes_decision_through(self):
        with mock.patch("src.graph.build.resume_query",
                        return_value=_fake_result()) as resume:
            r = self.client.post("/api/v2/resume",
                                 json={"thread_id": "t-1", "decision": "approve"})
        self.assertEqual(r.status_code, 200)
        resume.assert_called_once_with("t-1", "approve")

    def test_resume_unknown_thread_is_handled(self):
        with mock.patch("src.graph.build.resume_query",
                        side_effect=ValueError("no such thread")):
            r = self.client.post("/api/v2/resume",
                                 json={"thread_id": "does-not-exist",
                                       "decision": "approve"})
        self.assertEqual(r.status_code, 404)

    def test_resume_rejects_bad_decision(self):
        r = self.client.post("/api/v2/resume",
                             json={"thread_id": "t", "decision": "maybe"})
        self.assertIn(r.status_code, (400, 422))

    def test_v1_ask_endpoint_still_exists(self):
        """The original pipeline must keep working alongside v2."""
        routes = {r.path for r in self.app.routes if hasattr(r, "path")}
        self.assertIn("/api/ask", routes)
        self.assertIn("/api/v2/query", routes)


class TestDemoCLI(unittest.TestCase):

    def test_demo_module_exposes_main(self):
        from src.graph import demo
        self.assertTrue(callable(demo.main))

    def test_format_update_renders_a_line(self):
        from src.graph.demo import format_update
        line = format_update("supervisor", {"specialists": ["document"],
                                            "trace": ["supervisor:heuristic"]})
        self.assertIn("supervisor", line)
        self.assertIn("document", line)

    def test_format_update_handles_empty_update(self):
        from src.graph.demo import format_update
        self.assertIn("no state change", format_update("verifier", {}))


if __name__ == "__main__":
    unittest.main()
