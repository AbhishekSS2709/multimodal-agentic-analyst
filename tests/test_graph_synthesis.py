"""Tests for synthesis and verification nodes (heuristic mode)."""

import unittest

from src.graph.state import Finding, new_state


def _findings():
    return [
        Finding(specialist="document", content="Refunds are issued within 30 days.",
                score=0.9, source="policy.pdf", doc_id="d1"),
        Finding(specialist="document", content="Shipping takes 5 business days.",
                score=0.4, source="shipping.pdf", doc_id="d2"),
    ]


class TestSynthesizer(unittest.TestCase):

    def test_answer_uses_highest_scoring_finding(self):
        from src.graph.nodes.synthesizer import synthesize_heuristic
        answer, _ = synthesize_heuristic("what is the refund window?", _findings())
        self.assertIn("30 days", answer)

    def test_citations_reference_sources(self):
        from src.graph.nodes.synthesizer import synthesize_heuristic
        _, citations = synthesize_heuristic("refund window?", _findings())
        self.assertTrue(citations)
        self.assertIn("policy.pdf", {c.source for c in citations})

    def test_no_findings_returns_honest_answer(self):
        from src.graph.nodes.synthesizer import synthesize_heuristic
        answer, citations = synthesize_heuristic("anything?", [])
        self.assertEqual(citations, [])
        self.assertRegex(answer.lower(), r"could not|no relevant|not find")

    def test_node_populates_state_keys(self):
        from src.graph.nodes.synthesizer import synthesizer_node
        state = new_state("refund window?")
        state["findings"] = _findings()
        out = synthesizer_node(state)
        self.assertTrue(out["answer"])
        self.assertIn("citations", out)


class TestVerifier(unittest.TestCase):

    def test_grounded_answer_passes(self):
        from src.graph.nodes.verifier import grade_grounded_heuristic
        v = grade_grounded_heuristic("Refunds are issued within 30 days.", _findings())
        self.assertTrue(v.grounded)

    def test_ungrounded_answer_fails(self):
        from src.graph.nodes.verifier import grade_grounded_heuristic
        v = grade_grounded_heuristic(
            "The company was founded by penguins on Jupiter in 1823.", _findings())
        self.assertFalse(v.grounded)

    def test_no_findings_is_not_grounded(self):
        from src.graph.nodes.verifier import grade_grounded_heuristic
        self.assertFalse(grade_grounded_heuristic("Anything at all.", []).grounded)

    def test_should_retry_once_then_stops(self):
        from src.graph.nodes.verifier import should_retry
        state = new_state("q")
        state["verification"] = {"grounded": False, "relevant": False}
        state["retry_count"] = 0
        self.assertEqual(should_retry(state), "retry")
        state["retry_count"] = 1
        self.assertEqual(should_retry(state), "done")

    def test_should_not_retry_when_grounded(self):
        from src.graph.nodes.verifier import should_retry
        state = new_state("q")
        state["verification"] = {"grounded": True, "relevant": True}
        state["retry_count"] = 0
        self.assertEqual(should_retry(state), "done")

    def test_verifier_node_populates_state(self):
        from src.graph.nodes.verifier import verifier_node
        state = new_state("refund window?")
        state["findings"] = _findings()
        state["answer"] = "Refunds are issued within 30 days."
        out = verifier_node(state)
        self.assertIn("verification", out)
        self.assertIn("grounded", out["verification"])


if __name__ == "__main__":
    unittest.main()
