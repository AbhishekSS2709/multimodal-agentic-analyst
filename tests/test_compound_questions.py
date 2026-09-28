"""Compound questions reach every source they need.

The demo's own example -- "Which suppliers have the most delays, and what do
the contracts say about penalties?" -- came back without the penalty clause.
Three separate faults, one test class each:

* one embedding of the whole question ranked the delay records and never the
  contract, so the document specialist now searches each part as well;
* the SQL prompt had no column values, so the model wrote
  ``status = 'Delayed'`` against ``'delayed'`` and got zero rows;
* the Sources panel listed the first five findings, not the ones cited.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.graph.schemas import GradeDocuments, RoutePlan, SubTask
from src.graph.state import Finding, new_state

COMPOUND = ("Which suppliers have the most delays, and what do the contracts "
            "say about penalties?")


class TestSplitCompound(unittest.TestCase):

    def test_second_question_becomes_its_own_search(self):
        from src.graph.nodes.document import split_compound

        self.assertEqual(split_compound(COMPOUND), [
            COMPOUND, "what do the contracts say about penalties",
        ])

    def test_simple_question_is_unchanged(self):
        from src.graph.nodes.document import split_compound

        self.assertEqual(split_compound("Why are there dispatch delays?"),
                         ["Why are there dispatch delays?"])

    def test_part_that_leans_on_the_other_half_is_not_searched_alone(self):
        from src.graph.nodes.document import split_compound

        q = "What machine failures were logged, and what caused them?"
        self.assertEqual(split_compound(q), [q])

    def test_two_sentences_split(self):
        from src.graph.nodes.document import split_compound

        q = "What is the late delivery penalty? Which supplier is worst on quality?"
        self.assertEqual(split_compound(q)[1], "Which supplier is worst on quality")

    def test_and_inside_a_clause_does_not_split(self):
        from src.graph.nodes.document import split_compound

        q = "Compare Apex and GlobalTech on quality and cost"
        self.assertEqual(split_compound(q), [q])

    def test_capped(self):
        from src.graph.nodes.document import MAX_QUERIES, split_compound

        q = ("Which supplier is late, and what is the penalty for suppliers, and "
             "which region has most orders, and what machine failed on line two?")
        self.assertLessEqual(len(split_compound(q)), MAX_QUERIES)


class TestDocumentQueries(unittest.TestCase):

    def test_supervisor_split_wins(self):
        from src.graph.nodes.document import document_queries

        state = new_state(COMPOUND)
        state["plan"] = [
            {"specialist": "document", "description": "suppliers with most delays"},
            {"specialist": "analytics", "description": "count delayed orders"},
            {"specialist": "document", "description": "contract late delivery penalties"},
        ]
        self.assertEqual(document_queries(state), [
            "suppliers with most delays", "contract late delivery penalties",
        ])

    def test_single_planned_search_falls_back_to_the_splitter(self):
        from src.graph.nodes.document import document_queries

        state = new_state(COMPOUND)
        state["plan"] = [{"specialist": "document", "description": "delays"}]
        self.assertEqual(len(document_queries(state)), 2)


class _Chunk:
    def __init__(self, text, doc_id, source):
        self.text, self.doc_id, self.chunk_id = text, doc_id, doc_id
        self.metadata = {"source": source}


class _TopicBackend:
    """Hybrid-retriever stand-in: contract text only for a penalty search."""

    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k=5):
        self.queries.append(query)
        delays = (_Chunk("Apex Materials had 30 delayed dispatches in 2023.",
                         "d1", "dispatch_log.txt"), 0.8)
        if query.lower().startswith("what do the contracts"):
            return [(_Chunk("4.1 Late Delivery Penalties: 1-5 days late: written "
                            "notification only for the contracts.", "c1",
                            "contracts.txt"), 0.9)]
        return [delays]


class _Components:
    def __init__(self, backend):
        self.hybrid = backend


class TestDocumentNodeSearchesEachPart(unittest.TestCase):

    def _run(self):
        from src.graph.nodes import document as doc_mod

        backend = _TopicBackend()
        # Heuristic grading: no LLM in tests.
        with mock.patch.object(doc_mod, "get_structured_llm", return_value=None):
            out = doc_mod.document_node(new_state(COMPOUND), _Components(backend))
        return backend, out

    def test_contract_clause_is_found(self):
        _, out = self._run()
        sources = {f.source for f in out["findings"]}
        self.assertIn("contracts.txt", sources)

    def test_each_part_was_searched(self):
        backend, _ = self._run()
        self.assertIn(COMPOUND, backend.queries)
        self.assertIn("what do the contracts say about penalties", backend.queries)

    def test_trace_names_the_search_but_keeps_the_specialist_prefix(self):
        _, out = self._run()
        self.assertTrue(all(t.split(":")[0] == "document" for t in out["trace"]))
        self.assertTrue(any(t.startswith("document:q2:") for t in out["trace"]))

    def test_same_passage_from_two_searches_is_kept_once(self):
        from src.graph.nodes.document import _merge

        a = Finding(specialist="document", content="same", source="s", doc_id="1", score=0.2)
        b = Finding(specialist="document", content="same", source="s", doc_id="1", score=0.7)
        merged = _merge([a, b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].score, 0.7)


class TestParallelGradingKeepsOrder(unittest.TestCase):

    def test_grades_line_up_with_their_documents(self):
        import time

        from langchain_core.documents import Document

        from src.graph.nodes import document as doc_mod

        class _Structured:
            def invoke(self, prompt):
                # Later documents answer first; order must still hold.
                n = int(prompt.split("DOC-")[1][0])
                time.sleep(0.01 * (5 - n))
                return GradeDocuments(relevant=n % 2 == 0, score=n / 10, reason=str(n))

        docs = [Document(page_content=f"DOC-{i}", metadata={}) for i in range(5)]
        with mock.patch.object(doc_mod, "get_structured_llm", return_value=_Structured()):
            graded = doc_mod._grade_documents_llm("anything", docs)

        self.assertEqual([g.reason for _, g in graded], ["0", "1", "2", "3", "4"])
        self.assertEqual([d.page_content for d, _ in graded],
                         [d.page_content for d in docs])

    def test_one_failed_grade_falls_back(self):
        from langchain_core.documents import Document

        from src.graph.nodes import document as doc_mod

        class _Flaky:
            def invoke(self, prompt):
                raise RuntimeError("quota")

        docs = [Document(page_content="x", metadata={})]
        with mock.patch.object(doc_mod, "get_structured_llm", return_value=_Flaky()):
            self.assertIsNone(doc_mod._grade_documents_llm("q", docs))


class TestSupervisorKeepsSeveralDocumentSearches(unittest.TestCase):

    def _plan(self, subtasks):
        from src.graph.nodes import supervisor

        llm = mock.MagicMock()
        llm.invoke.return_value = RoutePlan(
            subtasks=[SubTask(description=d, specialist=s) for s, d in subtasks],
            rationale="r",
        )
        with mock.patch.object(supervisor, "get_structured_llm", return_value=llm):
            return supervisor._plan_llm(COMPOUND)

    def test_two_document_subtasks_survive(self):
        plan = self._plan([("document", "delays"), ("analytics", "count"),
                           ("document", "contract penalties")])
        self.assertEqual([(t.specialist, t.description) for t in plan.subtasks], [
            ("document", "delays"), ("document", "contract penalties"),
            ("analytics", "count"),
        ])

    def test_duplicate_document_subtasks_collapse(self):
        plan = self._plan([("document", "Delays"), ("document", "delays")])
        self.assertEqual(len(plan.subtasks), 1)

    def test_document_subtasks_are_capped(self):
        from src.graph.nodes.supervisor import MAX_DOCUMENT_QUERIES

        plan = self._plan([("document", f"part {i}") for i in range(6)])
        self.assertEqual(len(plan.subtasks), MAX_DOCUMENT_QUERIES)

    def test_other_specialists_still_appear_once(self):
        plan = self._plan([("analytics", "a"), ("analytics", "b"), ("document", "d")])
        self.assertEqual([t.specialist for t in plan.subtasks], ["document", "analytics"])

    def test_node_sends_each_specialist_once(self):
        from src.graph.nodes import supervisor

        llm = mock.MagicMock()
        llm.invoke.return_value = RoutePlan(subtasks=[
            SubTask(description="delays", specialist="document"),
            SubTask(description="penalties", specialist="document"),
            SubTask(description="count", specialist="analytics"),
        ], rationale="r")
        with mock.patch.object(supervisor, "get_structured_llm", return_value=llm):
            out = supervisor.supervisor_node(new_state(COMPOUND))
        self.assertEqual(out["specialists"], ["document", "analytics"])
        self.assertEqual(len(out["plan"]), 3)


class TestSchemaListsCategoryValues(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE orders (order_id TEXT, status TEXT, qty INTEGER)")
        rows = [(f"ORD-{i:03d}", ("delayed", "delivered")[i % 2], i) for i in range(40)]
        con.executemany("INSERT INTO orders VALUES (?,?,?)", rows)
        con.commit()
        con.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_low_cardinality_text_values_are_listed_verbatim(self):
        from src.sql_tool.db_setup import get_schema

        schema = get_schema(self.db)
        self.assertIn("orders.status: 'delayed', 'delivered'", schema)

    def test_identifier_columns_are_not_dumped(self):
        from src.sql_tool.db_setup import get_schema

        self.assertNotIn("orders.order_id:", get_schema(self.db))

    def test_numeric_columns_are_not_listed(self):
        from src.sql_tool.db_setup import get_schema

        self.assertNotIn("orders.qty:", get_schema(self.db))

    def test_prompt_tells_the_model_to_use_them(self):
        from src.sql_tool.sql_agent import SQL_GENERATION_PROMPT

        self.assertIn("Known values", SQL_GENERATION_PROMPT)


class TestSourcesAreTheCitedOnes(unittest.TestCase):

    def _findings(self, n):
        return [Finding(specialist="document", content=f"f{i}", source=f"s{i}")
                for i in range(1, n + 1)]

    def test_markers_pick_the_sources(self):
        from src.graph.nodes.synthesizer import cited_findings

        picked = cited_findings("Apex is late [2]. Penalties are weak [6].",
                                self._findings(7))
        self.assertEqual([f.source for f in picked], ["s2", "s6"])

    def test_grouped_markers_and_repeats(self):
        from src.graph.nodes.synthesizer import cited_findings

        picked = cited_findings("A [1, 3]. B [3][2].", self._findings(4))
        self.assertEqual([f.source for f in picked], ["s1", "s3", "s2"])

    def test_out_of_range_markers_are_ignored(self):
        from src.graph.nodes.synthesizer import cited_findings

        picked = cited_findings("A [9].", self._findings(2))
        self.assertEqual([f.source for f in picked], ["s1", "s2"])

    def test_no_markers_falls_back_to_the_first_five(self):
        from src.graph.nodes.synthesizer import cited_findings

        self.assertEqual(len(cited_findings("no markers", self._findings(8))), 5)


if __name__ == "__main__":
    unittest.main()
