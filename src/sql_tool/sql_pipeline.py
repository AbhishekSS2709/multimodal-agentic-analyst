"""Complete SQL analytics pipeline — question in, insight + chart out.

Combines :class:`SQLAgent` (natural-language -> SQL -> results -> insight) with
:class:`ChartGenerator` (auto-visualisation of tabular results) into a single
high-level ``analyze`` method.

Usage::

    from src.sql_tool.sql_pipeline import SQLAnalyticsPipeline

    pipeline = SQLAnalyticsPipeline()
    result = pipeline.analyze("Show monthly order trend")
    print(result["insight"])
    # result["chart_path"]["html"] -> path to interactive chart
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import SQLITE_DB_PATH

from src.sql_tool.sql_agent import SQLAgent
from src.visualization.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)


class SQLAnalyticsPipeline:
    """End-to-end analytics: question -> SQL -> results -> insight + chart.

    Args:
        db_path: Path to the SQLite database.
        use_llm: Override LLM mode (see :class:`SQLAgent`).
        chart_output_dir: Directory for saved chart files.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        use_llm: Optional[bool] = None,
        chart_output_dir: Optional[Path] = None,
    ) -> None:
        self._agent = SQLAgent(db_path=db_path, use_llm=use_llm)
        self._chart_gen = ChartGenerator()
        self._chart_dir = chart_output_dir

    def analyze(self, question: str) -> Dict[str, Any]:
        """Run the full analytics pipeline for a natural-language question.

        Args:
            question: A plain-English analytics question.

        Returns:
            A dictionary with keys:

            - **sql** (str): The generated SQL query.
            - **columns** (list[str]): Column names from the result set.
            - **results** (list[dict]): Raw result rows.
            - **insight** (str): Textual insight / summary.
            - **chart_html** (str): HTML string of the interactive chart
              (empty if no chart could be created).
            - **chart_path** (dict): ``{"html": "...", "png": "..."}``
              paths to saved chart files (empty if no chart).
        """
        # Step 1 — query via agent
        agent_result = self._agent.query(question)

        sql = agent_result["sql"]
        columns = agent_result["columns"]
        raw_results = agent_result["raw_results"]
        insight = agent_result["insight"]

        # Step 2 — auto-generate chart
        chart_html = ""
        chart_path: Dict[str, str] = {"html": "", "png": ""}

        if raw_results:
            fig = self._chart_gen.auto_chart(
                data=raw_results,
                query_context=question,
            )
            if fig is not None:
                # Inline HTML for embedding
                chart_html = fig.to_html(
                    full_html=False,
                    include_plotlyjs="cdn",
                )

                # Persist to disk
                safe_name = _slugify(question)
                chart_path = self._chart_gen.save_chart(
                    fig,
                    filename=safe_name,
                    output_dir=self._chart_dir,
                )

        logger.info(
            "Pipeline complete for '%s' — %d row(s), chart=%s.",
            question,
            len(raw_results),
            "yes" if chart_html else "no",
        )

        return {
            "sql": sql,
            "columns": columns,
            "results": raw_results,
            "insight": insight,
            "chart_html": chart_html,
            "chart_path": chart_path,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str, max_len: int = 50) -> str:
    """Convert a question string into a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "_", slug).strip("_")
    return slug[:max_len]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

    pipeline = SQLAnalyticsPipeline()

    demo_questions = [
        "Show monthly order trend",
        "Which supplier has the most delayed orders?",
        "Top 10 products by revenue",
        "Order status distribution",
    ]

    for q in demo_questions:
        print(f"\n{'='*72}")
        print(f"  Question : {q}")
        print(f"{'='*72}")

        result = pipeline.analyze(q)

        print(f"  SQL      : {result['sql']}")
        print(f"  Rows     : {len(result['results'])}")
        print(f"  Insight  : {result['insight']}")
        if result["chart_path"]["html"]:
            print(f"  Chart    : {result['chart_path']['html']}")
        else:
            print("  Chart    : (none)")

        # Show first 5 rows
        if result["results"]:
            df = pd.DataFrame(result["results"][:5])
            print(f"\n  Preview:\n{df.to_string(index=False)}")
