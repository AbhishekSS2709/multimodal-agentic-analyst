"""The analytics node reads rows from the real SQL pipeline.

Its unit tests used a fake backend that returned rows under ``"rows"``, but
``SQLAnalyticsPipeline.analyze`` returns them under ``"results"``. Every live
answer therefore carried the model's one-line insight and no data -- the trace
read ``analytics:emit(0rows)`` for a query that returned five. This runs the
real pipeline (pattern-match mode, no LLM) against the shipped warehouse.
"""

import tempfile
import unittest
from pathlib import Path

from src.graph.state import new_state

DB = Path(__file__).resolve().parent.parent / "data" / "enterprise.db"


class _Components:
    def __init__(self, pipeline):
        self.sql = pipeline


@unittest.skipUnless(DB.exists(), "sample warehouse not present")
class TestAnalyticsNodeWithRealPipeline(unittest.TestCase):

    def setUp(self):
        from src.sql_tool.sql_pipeline import SQLAnalyticsPipeline

        self.tmp = tempfile.TemporaryDirectory()
        self.pipeline = SQLAnalyticsPipeline(
            db_path=DB, use_llm=False, chart_output_dir=Path(self.tmp.name),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_reach_the_finding(self):
        from src.graph.nodes.analytics import analytics_node

        out = analytics_node(new_state("How many orders were placed per region?"),
                             _Components(self.pipeline))
        finding = out["findings"][0]
        self.assertGreater(finding.metadata["row_count"], 0)
        self.assertEqual(finding.score, 1.0)
        self.assertNotIn("(0rows)", out["trace"][0])

    def test_pipeline_still_names_the_key_this_node_reads(self):
        result = self.pipeline.analyze("How many orders were placed per region?")
        self.assertTrue(result.get("rows") or result.get("results"))


if __name__ == "__main__":
    unittest.main()
