"""Evaluation report generator — JSON and HTML output.

Produces structured evaluation reports that can be consumed
programmatically (JSON) or reviewed visually (HTML).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    EMBEDDING_MODEL,
    EVALUATION_DIR,
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    LLM_MODEL,
    TOP_K,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate JSON and HTML evaluation reports.

    Parameters
    ----------
    output_dir : Path | str | None
        Directory where reports are written.  Defaults to
        ``EVALUATION_DIR`` from project settings.
    """

    def __init__(self, output_dir: Optional[Path | str] = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else EVALUATION_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------

    def generate_json_report(
        self,
        results: Dict[str, Any],
        filename: str = "evaluation_results.json",
    ) -> Path:
        """Write a structured JSON report to disk.

        Parameters
        ----------
        results : dict
            Output of :meth:`RAGEvaluator.evaluate_batch`.
        filename : str
            Report filename.

        Returns
        -------
        Path
            Full path of the saved file.
        """
        report: Dict[str, Any] = {
            "meta": self._build_meta(),
            "overall_metrics": results.get("aggregated", {}),
            "per_category_metrics": results.get("per_category", {}),
            "per_case_details": results.get("per_case", []),
        }

        out_path = self._output_dir / filename
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str, ensure_ascii=False)
        logger.info("JSON report saved to %s.", out_path)
        return out_path

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        results: Dict[str, Any],
        filename: str = "report.html",
    ) -> Path:
        """Generate a self-contained HTML report for visual review.

        Parameters
        ----------
        results : dict
            Output of :meth:`RAGEvaluator.evaluate_batch`.
        filename : str
            Report filename.

        Returns
        -------
        Path
            Full path of the saved file.
        """
        meta = self._build_meta()
        aggregated = results.get("aggregated", {})
        per_category = results.get("per_category", {})
        per_case = results.get("per_case", [])

        html_parts: list[str] = [
            self._html_head(meta),
            self._html_overall_section(aggregated),
            self._html_category_section(per_category),
            self._html_per_case_section(per_case),
            self._html_footer(),
        ]

        out_path = self._output_dir / filename
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(html_parts))
        logger.info("HTML report saved to %s.", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta() -> Dict[str, Any]:
        """Build metadata block for the report."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "embedding_model": EMBEDDING_MODEL,
                "llm_model": LLM_MODEL,
                "top_k": TOP_K,
                "hybrid_vector_weight": HYBRID_VECTOR_WEIGHT,
                "hybrid_bm25_weight": HYBRID_BM25_WEIGHT,
            },
        }

    @staticmethod
    def _fmt_metric(value: Any) -> str:
        """Format a metric value for display."""
        if value is None:
            return "N/A"
        if isinstance(value, dict):
            mean = value.get("mean")
            return f"{mean:.4f}" if mean is not None else "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    # ------------------------------------------------------------------
    # HTML builders
    # ------------------------------------------------------------------

    def _html_head(self, meta: Dict[str, Any]) -> str:
        ts = meta.get("timestamp", "")
        config = meta.get("configuration", {})
        config_rows = "\n".join(
            f"            <tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in config.items()
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAG Evaluation Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #1a1a2e; background: #f8f9fa; }}
    h1 {{ color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 0.5rem; }}
    h2 {{ color: #0f3460; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th, td {{ padding: 0.6rem 0.8rem; border: 1px solid #dee2e6; text-align: left; }}
    th {{ background: #16213e; color: #fff; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f1f3f5; }}
    .metric-good {{ color: #2d6a4f; font-weight: 600; }}
    .metric-ok {{ color: #e9c46a; font-weight: 600; }}
    .metric-bad {{ color: #e63946; font-weight: 600; }}
    .meta {{ font-size: 0.9rem; color: #555; }}
    .error-row {{ background: #fff0f0 !important; }}
    .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px;
              font-size: 0.8rem; font-weight: 600; }}
    .badge-factual {{ background: #d0f0c0; color: #2d6a4f; }}
    .badge-reasoning {{ background: #c0d0f0; color: #1d3557; }}
    .badge-sql {{ background: #f0e0c0; color: #7f5539; }}
    .badge-summary {{ background: #e0c0f0; color: #5a189a; }}
    .badge-comparison {{ background: #c0f0f0; color: #0077b6; }}
    .badge-edge {{ background: #f0c0c0; color: #9d0208; }}
</style>
</head>
<body>
<h1>RAG Evaluation Report</h1>
<p class="meta">Generated: {ts}</p>
<h2>Configuration</h2>
<table>
    <tr><th>Parameter</th><th>Value</th></tr>
{config_rows}
</table>"""

    def _html_overall_section(self, aggregated: Dict[str, Any]) -> str:
        num_cases = aggregated.get("num_cases", 0)
        metric_names = [
            "retrieval_precision",
            "retrieval_recall",
            "answer_relevance",
            "faithfulness",
            "keyword_hit_rate",
            "latency_ms",
        ]
        rows: list[str] = []
        for name in metric_names:
            val = aggregated.get(name)
            if val is None:
                rows.append(f"    <tr><td>{name}</td><td>N/A</td><td>N/A</td>"
                            f"<td>N/A</td><td>N/A</td><td>N/A</td></tr>")
                continue
            if isinstance(val, dict):
                cls = self._color_class(val.get("mean", 0), name)
                rows.append(
                    f"    <tr><td>{name}</td>"
                    f"<td class=\"{cls}\">{val.get('mean', 'N/A'):.4f}</td>"
                    f"<td>{val.get('median', 'N/A'):.4f}</td>"
                    f"<td>{val.get('min', 'N/A'):.4f}</td>"
                    f"<td>{val.get('max', 'N/A'):.4f}</td>"
                    f"<td>{val.get('std', 'N/A'):.4f}</td></tr>"
                )

        return f"""
<h2>Overall Metrics ({num_cases} cases)</h2>
<table>
    <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Std</th></tr>
{"".join(rows)}
</table>"""

    def _html_category_section(self, per_category: Dict[str, Any]) -> str:
        if not per_category:
            return "<h2>Per-Category Metrics</h2><p>No category data.</p>"

        metric_names = [
            "retrieval_precision",
            "retrieval_recall",
            "answer_relevance",
            "faithfulness",
            "keyword_hit_rate",
        ]

        rows: list[str] = []
        for cat, agg in sorted(per_category.items()):
            badge = f'<span class="badge badge-{cat}">{cat}</span>'
            cells = [f"<td>{badge}</td>", f"<td>{agg.get('num_cases', 0)}</td>"]
            for m in metric_names:
                val = agg.get(m)
                if val and isinstance(val, dict) and val.get("mean") is not None:
                    cls = self._color_class(val["mean"], m)
                    cells.append(f'<td class="{cls}">{val["mean"]:.4f}</td>')
                else:
                    cells.append("<td>N/A</td>")
            rows.append(f"    <tr>{''.join(cells)}</tr>")

        header_cells = "".join(f"<th>{m}</th>" for m in metric_names)
        return f"""
<h2>Per-Category Metrics</h2>
<table>
    <tr><th>Category</th><th>Cases</th>{header_cells}</tr>
{"".join(rows)}
</table>"""

    def _html_per_case_section(self, per_case: List[Dict[str, Any]]) -> str:
        if not per_case:
            return "<h2>Per-Case Details</h2><p>No case data.</p>"

        rows: list[str] = []
        for i, case in enumerate(per_case, 1):
            if "error" in case:
                rows.append(
                    f'    <tr class="error-row"><td>{i}</td>'
                    f'<td>{_esc(case.get("query", ""))[:80]}</td>'
                    f'<td>{case.get("category", "")}</td>'
                    f'<td colspan="5">ERROR: {_esc(case["error"])}</td></tr>'
                )
                continue

            cat = case.get("category", "")
            badge = f'<span class="badge badge-{cat}">{cat}</span>' if cat else ""
            rp = self._fmt_metric(case.get("retrieval_precision"))
            rr = self._fmt_metric(case.get("retrieval_recall"))
            ar = self._fmt_metric(case.get("answer_relevance"))
            ff = self._fmt_metric(case.get("faithfulness"))
            kh = self._fmt_metric(case.get("keyword_hit_rate"))
            rows.append(
                f"    <tr><td>{i}</td>"
                f"<td>{_esc(case.get('query', ''))[:80]}</td>"
                f"<td>{badge}</td>"
                f"<td>{rp}</td><td>{rr}</td><td>{ar}</td><td>{ff}</td><td>{kh}</td></tr>"
            )

        return f"""
<h2>Per-Case Details</h2>
<table>
    <tr><th>#</th><th>Query</th><th>Category</th>
    <th>Precision</th><th>Recall</th><th>Relevance</th>
    <th>Faithfulness</th><th>Keyword Hit</th></tr>
{"".join(rows)}
</table>"""

    @staticmethod
    def _html_footer() -> str:
        return """
<hr>
<p class="meta">Report generated by RAG Evaluation Pipeline</p>
</body>
</html>"""

    @staticmethod
    def _color_class(value: float, metric_name: str) -> str:
        """Return a CSS class based on the metric score."""
        if metric_name == "latency_ms":
            # Lower is better for latency
            if value < 500:
                return "metric-good"
            if value < 2000:
                return "metric-ok"
            return "metric-bad"
        # Higher is better for quality metrics
        if value >= 0.7:
            return "metric-good"
        if value >= 0.4:
            return "metric-ok"
        return "metric-bad"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
