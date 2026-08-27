"""`document` is the floor, and it must hold for BOTH planners.

`plan_heuristic` dispatches `document` unconditionally: text retrieval is
almost always worth running, and it guarantees the graph never fans out to a
single specialist that returns nothing and leaves the synthesizer with no
evidence at all.

`_plan_llm` only *asked* for it in the prompt, and asking is not a guarantee --
the LLM dropped `document` on 6 of 25 evaluation questions, routing
`['analytics']` alone on SQL questions and `['visual']` alone on the OCR one.
`routing_precision` excludes the floor by construction, so enforcing it cannot
inflate that score; it is a robustness invariant, not a scoring trick.
"""

import unittest
from unittest import mock

from src.graph.schemas import RoutePlan, SubTask
from src.graph.state import SPECIALISTS


def _fake_structured(plan):
    llm = mock.MagicMock()
    llm.invoke.return_value = plan
    return llm


class TestLlmPlansGetTheFloor(unittest.TestCase):

    def _plan(self, specialists):
        from src.graph.nodes import supervisor

        plan = RoutePlan(
            subtasks=[SubTask(description="d", specialist=s) for s in specialists],
            rationale="r",
        )
        with mock.patch.object(supervisor, "get_structured_llm",
                               return_value=_fake_structured(plan)):
            result = supervisor._plan_llm("compare Q3 and Q4")
        return [t.specialist for t in result.subtasks]

    def test_document_is_added_when_the_model_omits_it(self):
        self.assertEqual(self._plan(["analytics"]), ["document", "analytics"])

    def test_visual_only_plan_also_gets_the_floor(self):
        self.assertEqual(self._plan(["visual"]), ["document", "visual"])

    def test_existing_document_is_not_duplicated(self):
        self.assertEqual(self._plan(["document", "graph"]), ["document", "graph"])

    def test_the_floor_carries_the_question_as_its_description(self):
        from src.graph.nodes import supervisor

        plan = RoutePlan(subtasks=[SubTask(description="d", specialist="analytics")],
                         rationale="r")
        with mock.patch.object(supervisor, "get_structured_llm",
                               return_value=_fake_structured(plan)):
            result = supervisor._plan_llm("how many orders?")
        floor = next(t for t in result.subtasks if t.specialist == "document")
        self.assertEqual(floor.description, "how many orders?")

    def test_specialists_come_back_in_canonical_order(self):
        self.assertEqual(self._plan(["graph", "visual", "analytics"]),
                         [s for s in SPECIALISTS])

    def test_a_plan_of_only_hallucinated_specialists_still_falls_back(self):
        """The floor must not rescue a plan that named nothing real."""
        self.assertIsNone(self._plan_raw(["astrologer"]))

    def _plan_raw(self, specialists):
        from src.graph.nodes import supervisor

        plan = RoutePlan(
            subtasks=[SubTask(description="d", specialist=s) for s in specialists],
            rationale="r",
        )
        with mock.patch.object(supervisor, "get_structured_llm",
                               return_value=_fake_structured(plan)):
            return supervisor._plan_llm("q")


class TestBothPlannersShareTheInvariant(unittest.TestCase):

    def test_heuristic_always_dispatches_the_floor(self):
        from src.graph.nodes.supervisor import DOCUMENT_FLOOR, plan_heuristic
        from src.evaluation.test_cases import TEST_CASES

        for case in TEST_CASES:
            picked = [t.specialist for t in plan_heuristic(case["query"]).subtasks]
            self.assertIn(DOCUMENT_FLOOR, picked, case["query"])


class TestThePromptStatesTheContract(unittest.TestCase):
    """The prompt used to contradict the code, which is why the model diverged."""

    def test_prompt_asks_for_the_floor_unconditionally(self):
        from src.graph.prompts import SUPERVISOR_PROMPT

        self.assertIn("Always include `document`.", SUPERVISOR_PROMPT)
        self.assertNotIn("unless the question is purely numeric", SUPERVISOR_PROMPT)

    def test_prompt_routes_comparison_to_the_graph_specialist(self):
        """`plan_heuristic` and the evaluator both map comparison -> graph."""
        from src.graph.prompts import SUPERVISOR_PROMPT

        self.assertIn("comparative", SUPERVISOR_PROMPT)
        self.assertIn("`graph` job", SUPERVISOR_PROMPT)


if __name__ == "__main__":
    unittest.main()
