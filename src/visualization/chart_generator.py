"""Automatic chart generation — picks the right visualization for query results.

Uses Plotly for interactive, professional charts and supports export to both
HTML (interactive) and PNG (static) formats.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

CHARTS_DIR = PROJECT_ROOT / "static" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Professional theme
# ---------------------------------------------------------------------------

_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#333333"),
    title_font=dict(size=18, color="#1a1a2e"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(255,255,255,1)",
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5,
    ),
    colorway=[
        "#4361ee", "#3a0ca3", "#7209b7", "#f72585",
        "#4cc9f0", "#06d6a0", "#ffd166", "#ef476f",
    ],
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the standard professional layout to a Plotly figure."""
    fig.update_layout(**_LAYOUT_DEFAULTS)
    return fig


# ---------------------------------------------------------------------------
# Chart type detection
# ---------------------------------------------------------------------------

# Heuristic keywords in query text or column names
_TIME_KEYWORDS = re.compile(
    r"(month|year|date|week|quarter|daily|weekly|monthly|yearly|trend|time\s*series)",
    re.IGNORECASE,
)
_PROPORTION_KEYWORDS = re.compile(
    r"(distribution|proportion|share|percentage|breakdown|pie|composition)",
    re.IGNORECASE,
)
_RANKING_KEYWORDS = re.compile(
    r"(top|bottom|rank|best|worst|highest|lowest|most|least)",
    re.IGNORECASE,
)


def _detect_chart_type(
    df: pd.DataFrame,
    query_context: str = "",
) -> str:
    """Heuristically decide the best chart type for the data.

    Returns one of: ``"line"``, ``"bar"``, ``"pie"``, ``"histogram"``.
    """
    columns_lower = [c.lower() for c in df.columns]

    # 1. Explicit proportion language -> pie (only if few categories)
    if _PROPORTION_KEYWORDS.search(query_context) and len(df) <= 12:
        return "pie"

    # 2. Time series: at least one column looks like a date / month
    has_time_col = any(
        kw in col for col in columns_lower for kw in ("month", "year", "date", "week", "quarter")
    )
    if has_time_col or _TIME_KEYWORDS.search(query_context):
        return "line"

    # 3. Ranking / categorical comparison
    if _RANKING_KEYWORDS.search(query_context):
        return "bar"

    # 4. Small number of categorical rows -> bar; very few -> pie
    if len(df) <= 6 and len(df.columns) >= 2:
        # If there's a count/amount column, pie might work
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 1:
            return "pie"
        return "bar"

    if len(df) <= 30:
        return "bar"

    # 5. Large numeric column with no obvious grouping -> histogram
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) >= 1 and len(df) > 30:
        return "histogram"

    return "bar"


# ---------------------------------------------------------------------------
# ChartGenerator
# ---------------------------------------------------------------------------


class ChartGenerator:
    """Generates professional Plotly charts from query result data.

    Typical usage::

        cg = ChartGenerator()
        fig = cg.auto_chart(data=rows, query_context="monthly order trend")
        cg.save_chart(fig, "monthly_trend")
    """

    # ---- individual chart builders ----------------------------------------

    @staticmethod
    def generate_bar_chart(
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "Bar Chart",
        color: Optional[str] = None,
    ) -> go.Figure:
        """Create a styled horizontal or vertical bar chart."""
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            text_auto=".2s",
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(title_text=x.replace("_", " ").title())
        fig.update_yaxes(title_text=y.replace("_", " ").title())
        return _apply_theme(fig)

    @staticmethod
    def generate_line_chart(
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "Line Chart",
        color: Optional[str] = None,
    ) -> go.Figure:
        """Create a styled line chart (ideal for time series)."""
        fig = px.line(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            markers=True,
        )
        fig.update_xaxes(title_text=x.replace("_", " ").title())
        fig.update_yaxes(title_text=y.replace("_", " ").title())
        return _apply_theme(fig)

    @staticmethod
    def generate_pie_chart(
        data: pd.DataFrame,
        labels: str,
        values: str,
        title: str = "Pie Chart",
    ) -> go.Figure:
        """Create a styled pie / donut chart."""
        fig = px.pie(
            data,
            names=labels,
            values=values,
            title=title,
            hole=0.35,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )
        return _apply_theme(fig)

    @staticmethod
    def generate_histogram(
        data: pd.DataFrame,
        x: str,
        title: str = "Distribution",
        nbins: int = 30,
    ) -> go.Figure:
        """Create a styled histogram."""
        fig = px.histogram(
            data,
            x=x,
            title=title,
            nbins=nbins,
        )
        fig.update_xaxes(title_text=x.replace("_", " ").title())
        fig.update_yaxes(title_text="Count")
        return _apply_theme(fig)

    # ---- auto chart -------------------------------------------------------

    def auto_chart(
        self,
        data: List[Dict[str, Any]] | pd.DataFrame,
        query_context: str = "",
        title: Optional[str] = None,
    ) -> Optional[go.Figure]:
        """Automatically detect the best chart type and generate it.

        Args:
            data: Query results as a list of dicts or a DataFrame.
            query_context: The original natural-language question (used for
                heuristic chart-type detection).
            title: Override chart title.  If *None*, one is derived from
                *query_context*.

        Returns:
            A Plotly ``Figure``, or *None* if the data is unsuitable for
            charting (e.g. empty or a single scalar).
        """
        if isinstance(data, list):
            if not data:
                logger.warning("auto_chart received empty data; returning None.")
                return None
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        if df.empty:
            logger.warning("auto_chart received an empty DataFrame; returning None.")
            return None

        chart_title = title or _derive_title(query_context)
        chart_type = _detect_chart_type(df, query_context)

        # Identify x (categorical / time) and y (numeric) columns
        numeric_cols = list(df.select_dtypes(include="number").columns)
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        if not numeric_cols:
            logger.warning("No numeric columns found; cannot chart.")
            return None

        x_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
        y_col = numeric_cols[0]

        logger.info(
            "auto_chart: type=%s, x=%s, y=%s, rows=%d",
            chart_type, x_col, y_col, len(df),
        )

        if chart_type == "line":
            return self.generate_line_chart(df, x=x_col, y=y_col, title=chart_title)
        elif chart_type == "pie":
            return self.generate_pie_chart(df, labels=x_col, values=y_col, title=chart_title)
        elif chart_type == "histogram":
            return self.generate_histogram(df, x=y_col, title=chart_title)
        else:  # bar (default)
            return self.generate_bar_chart(df, x=x_col, y=y_col, title=chart_title)

    # ---- save / export ----------------------------------------------------

    @staticmethod
    def save_chart(
        fig: go.Figure,
        filename: str,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        """Save a Plotly figure as both HTML (interactive) and PNG (static).

        Args:
            fig: The Plotly figure to save.
            filename: Base filename (without extension).
            output_dir: Directory for output files.  Defaults to
                ``static/charts/``.

        Returns:
            A dict ``{"html": "<path>", "png": "<path>"}``.
        """
        out = output_dir or CHARTS_DIR
        out.mkdir(parents=True, exist_ok=True)

        # Sanitise filename
        safe_name = re.sub(r"[^\w\-]", "_", filename)

        html_path = out / f"{safe_name}.html"
        png_path = out / f"{safe_name}.png"

        # HTML — always works
        pio.write_html(fig, str(html_path), auto_open=False)
        logger.info("Chart saved (HTML): %s", html_path)

        # PNG — requires kaleido; degrade gracefully
        try:
            pio.write_image(fig, str(png_path), width=1000, height=600, scale=2)
            logger.info("Chart saved (PNG): %s", png_path)
        except Exception as exc:
            logger.warning(
                "PNG export failed (install kaleido for static images): %s", exc
            )
            png_path = None  # type: ignore[assignment]

        return {
            "html": str(html_path),
            "png": str(png_path) if png_path else "",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_title(query_context: str) -> str:
    """Create a clean chart title from the user's question."""
    if not query_context:
        return "Query Results"
    # Strip leading question words
    title = re.sub(
        r"^(show|display|give|get|find|list|what|which|how)\s+(me\s+)?",
        "",
        query_context,
        flags=re.IGNORECASE,
    ).strip()
    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]
    return title or "Query Results"


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick demo with synthetic data
    sample_data = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "revenue": [12000, 15000, 13500, 17000, 16200, 19000],
        "orders": [120, 145, 130, 160, 155, 180],
    })

    cg = ChartGenerator()
    fig = cg.auto_chart(sample_data, query_context="monthly revenue trend")
    if fig:
        paths = cg.save_chart(fig, "demo_monthly_trend")
        print(f"Charts saved: {paths}")
