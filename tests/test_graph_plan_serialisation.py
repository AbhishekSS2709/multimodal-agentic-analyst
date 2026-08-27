"""Checkpointed state must contain only plain data.

The supervisor wrote `state["plan"]` as a list of Pydantic ``SubTask`` objects.
LangGraph's checkpointer serialises them, but deserialising warns:

    Deserializing unregistered type src.graph.schemas.SubTask from checkpoint.
    This will be blocked in a future version. Set LANGGRAPH_STRICT_MSGPACK=true

It works today and breaks on a future LangGraph upgrade. Nothing reads the plan
-- it exists for observability -- so plain dicts carry the same information with
no custom type to register.
"""

import json
import unittest

from src.graph.state import new_state


class TestPlanIsPlainData(unittest.TestCase):

    def _plan(self, question="why are there dispatch delays?"):
        from src.graph.nodes.supervisor import supervisor_node
        return supervisor_node(new_state(question))["plan"]

    def test_plan_entries_are_dicts(self):
        plan = self._plan()
        self.assertTrue(plan)
        for entry in plan:
            self.assertIsInstance(entry, dict,
                                  "plan still carries a Pydantic model")

    def test_plan_is_json_serialisable(self):
        json.dumps(self._plan())

    def test_plan_keeps_specialist_and_description(self):
        for entry in self._plan():
            self.assertIn("specialist", entry)
            self.assertIn("description", entry)

    def test_specialists_still_match_the_plan(self):
        from src.graph.nodes.supervisor import supervisor_node
        out = supervisor_node(new_state("why are there dispatch delays?"))
        self.assertEqual([e["specialist"] for e in out["plan"]],
                         out["specialists"])

    def test_empty_question_still_produces_a_plain_plan(self):
        plan = self._plan("")
        json.dumps(plan)
        self.assertTrue(all(isinstance(e, dict) for e in plan))


class TestStrictMsgpackRoundTrip(unittest.TestCase):
    """Prove it against LangGraph's own serialiser, not just json."""

    def test_state_plan_survives_the_checkpoint_serialiser(self):
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from src.graph.nodes.supervisor import supervisor_node

        plan = supervisor_node(new_state("why are there dispatch delays?"))["plan"]
        serde = JsonPlusSerializer()
        restored = serde.loads_typed(serde.dumps_typed({"plan": plan}))
        self.assertEqual(restored["plan"], plan)


if __name__ == "__main__":
    unittest.main()
