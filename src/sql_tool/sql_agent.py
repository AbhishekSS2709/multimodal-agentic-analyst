"""SQL query agent — translates natural language to SQL, executes, and generates insights.

Supports two modes:
  1. **LLM mode** — uses OpenAI to generate SQL and summarise results.
  2. **Pattern-matching fallback** — when no API key is available, maps common
     question patterns to pre-built SQL templates.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    SQL_MAX_ROWS,
    SQLITE_DB_PATH,
)
from src.sql_tool.db_setup import get_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = """\
Given this schema:
{schema}

Generate a SQL query for: {question}
Return ONLY the SQL query, nothing else.
Important rules:
- Use only SELECT statements (no INSERT, UPDATE, DELETE, DROP, etc.)
- Use SQLite-compatible syntax (e.g. strftime for dates)
- Limit results to {max_rows} rows unless the question asks for a specific limit
"""

INSIGHT_PROMPT = """\
A user asked: "{question}"

The SQL query executed was:
{sql}

The results are (showing up to 20 rows):
{results_preview}

Provide a concise, data-driven insight (2-4 sentences) summarising the key findings.
If the results are empty, say so clearly and suggest a possible reason.
"""

# ---------------------------------------------------------------------------
# SQL safety
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|GRANT|REVOKE|EXEC|EXECUTE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)

_MULTIPLE_STATEMENTS = re.compile(r";\s*\S")


def _validate_sql(sql: str) -> Tuple[bool, str]:
    """Validate that a SQL string is a safe, read-only SELECT query.

    Returns:
        A tuple ``(is_valid, reason)``.
    """
    sql_stripped = sql.strip().rstrip(";").strip()

    if not sql_stripped:
        return False, "Empty SQL query."

    if not sql_stripped.upper().startswith("SELECT"):
        return False, "Only SELECT queries are permitted."

    if _FORBIDDEN_KEYWORDS.search(sql_stripped):
        match = _FORBIDDEN_KEYWORDS.search(sql_stripped)
        return False, f"Forbidden keyword detected: {match.group() if match else 'unknown'}."

    if _MULTIPLE_STATEMENTS.search(sql_stripped):
        return False, "Multiple SQL statements are not allowed."

    return True, "OK"


# ---------------------------------------------------------------------------
# Pattern-matching fallback
# ---------------------------------------------------------------------------

_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # Monthly trend
    (
        re.compile(r"monthly\s*(order)?\s*trend", re.IGNORECASE),
        (
            "SELECT strftime('%Y-%m', order_date) AS month, "
            "COUNT(*) AS order_count, "
            "ROUND(SUM(total_amount), 2) AS revenue "
            "FROM orders GROUP BY month ORDER BY month"
        ),
        "Monthly order trend with count and revenue.",
    ),
    # By supplier
    (
        re.compile(r"by\s+supplier", re.IGNORECASE),
        (
            "SELECT supplier, COUNT(*) AS order_count, "
            "ROUND(SUM(total_amount), 2) AS revenue "
            "FROM orders GROUP BY supplier ORDER BY revenue DESC"
        ),
        "Order count and revenue grouped by supplier.",
    ),
    # Delayed orders
    (
        re.compile(r"delayed\s*(order)?", re.IGNORECASE),
        (
            "SELECT supplier, region, COUNT(*) AS delayed_count "
            "FROM orders WHERE LOWER(status) = 'delayed' "
            "GROUP BY supplier, region ORDER BY delayed_count DESC"
        ),
        "Delayed orders broken down by supplier and region.",
    ),
    # Top products
    (
        re.compile(r"top\s*(\d+)?\s*products?", re.IGNORECASE),
        (
            "SELECT product, ROUND(SUM(total_amount), 2) AS revenue, "
            "SUM(quantity) AS total_qty "
            "FROM orders GROUP BY product ORDER BY revenue DESC LIMIT {limit}"
        ),
        "Top products ranked by total revenue.",
    ),
    # Status distribution
    (
        re.compile(r"status\s*(distribution|breakdown|summary)", re.IGNORECASE),
        (
            "SELECT status, COUNT(*) AS order_count, "
            "ROUND(SUM(total_amount), 2) AS total_revenue "
            "FROM orders GROUP BY status ORDER BY order_count DESC"
        ),
        "Order status distribution.",
    ),
    # Revenue by region
    (
        re.compile(r"(revenue|sales)\s*(by|per)\s*region", re.IGNORECASE),
        (
            "SELECT region, ROUND(SUM(total_amount), 2) AS revenue, "
            "COUNT(*) AS order_count "
            "FROM orders GROUP BY region ORDER BY revenue DESC"
        ),
        "Revenue and order count by region.",
    ),
    # Priority breakdown
    (
        re.compile(r"priority", re.IGNORECASE),
        (
            "SELECT priority, COUNT(*) AS order_count, "
            "ROUND(AVG(total_amount), 2) AS avg_order_value "
            "FROM orders GROUP BY priority ORDER BY order_count DESC"
        ),
        "Order distribution by priority level.",
    ),
    # Generic fallback — recent orders
    (
        re.compile(r".*", re.IGNORECASE),
        (
            "SELECT * FROM orders ORDER BY order_date DESC LIMIT 20"
        ),
        "Showing the 20 most recent orders.",
    ),
]


def _pattern_match(question: str) -> Tuple[str, str]:
    """Match a natural-language question to a canned SQL query.

    Returns:
        ``(sql, description)``
    """
    for pattern, sql_template, description in _PATTERNS:
        match = pattern.search(question)
        if match:
            # Handle dynamic limit in "top N products"
            sql = sql_template
            if "{limit}" in sql:
                groups = match.groups()
                limit = int(groups[0]) if groups and groups[0] else 10
                sql = sql.format(limit=limit)
            return sql, description

    # Should never reach here because of the catch-all pattern
    return "SELECT * FROM orders LIMIT 20", "Showing sample orders."


# ---------------------------------------------------------------------------
# SQL Agent
# ---------------------------------------------------------------------------


class SQLAgent:
    """Translates natural-language questions into SQL, executes them, and
    optionally generates a textual insight via an LLM.

    Args:
        db_path: Path to the SQLite database file.
        use_llm: Force LLM mode on/off.  When *None* (default), LLM mode
            is enabled automatically if ``OPENAI_API_KEY`` is set.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        use_llm: Optional[bool] = None,
    ) -> None:
        self._db_path = db_path or SQLITE_DB_PATH
        self._engine = create_engine(f"sqlite:///{self._db_path}", echo=False)
        self._schema: Optional[str] = None

        # Decide whether to use the LLM
        if use_llm is not None:
            self._use_llm = use_llm
        else:
            try:
                from src.llm_provider import get_active_provider
                self._use_llm = get_active_provider() != "none"
            except Exception:
                self._use_llm = bool(OPENAI_API_KEY)

        logger.info(
            "SQLAgent initialised (mode=%s, db=%s).",
            "LLM" if self._use_llm else "pattern-match",
            self._db_path,
        )

    # ---- helpers ----------------------------------------------------------

    @property
    def schema(self) -> str:
        """Lazily fetch and cache the database schema string."""
        if self._schema is None:
            self._schema = get_schema(self._db_path)
        return self._schema

    def _call_llm(self, prompt: str) -> str:
        """Call the best available LLM provider."""
        from src.llm_provider import call_llm
        return call_llm(prompt, "You are a SQL expert. Return only valid SQLite SQL queries.")

    def _generate_sql_llm(self, question: str) -> str:
        """Use the LLM to generate a SQL query."""
        prompt = SQL_GENERATION_PROMPT.format(
            schema=self.schema,
            question=question,
            max_rows=SQL_MAX_ROWS,
        )
        raw = self._call_llm(prompt)
        # Strip markdown fences if present
        sql = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.MULTILINE)
        sql = re.sub(r"```\s*$", "", sql, flags=re.MULTILINE)
        return sql.strip()

    def _generate_insight_llm(
        self, question: str, sql: str, results_preview: str
    ) -> str:
        """Use the LLM to produce a human-readable insight."""
        prompt = INSIGHT_PROMPT.format(
            question=question,
            sql=sql,
            results_preview=results_preview,
        )
        return self._call_llm(prompt)

    def _execute_sql(self, sql: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Execute a validated SQL query and return ``(columns, rows)``.

        Raises:
            RuntimeError: On execution errors.
        """
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(sql))
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return columns, rows
        except Exception as exc:
            logger.error("SQL execution failed: %s\nQuery: %s", exc, sql)
            raise RuntimeError(f"SQL execution failed: {exc}") from exc

    # ---- public API -------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """Answer a natural-language question by generating and executing SQL.

        Args:
            question: The user's question in plain English.

        Returns:
            A dictionary with keys:
              - **sql** (str): The executed SQL query.
              - **columns** (list[str]): Column names of the result set.
              - **raw_results** (list[dict]): Result rows as dictionaries.
              - **insight** (str): A textual summary / insight.
        """
        # Step 1 — generate SQL
        if self._use_llm:
            sql = self._generate_sql_llm(question)
        else:
            sql, fallback_description = _pattern_match(question)

        # Step 2 — validate
        is_valid, reason = _validate_sql(sql)
        if not is_valid:
            logger.warning("Generated SQL rejected (%s): %s", reason, sql)
            return {
                "sql": sql,
                "columns": [],
                "raw_results": [],
                "insight": f"The generated SQL was rejected: {reason}",
            }

        # Step 3 — execute
        try:
            columns, raw_results = self._execute_sql(sql)
        except RuntimeError as exc:
            return {
                "sql": sql,
                "columns": [],
                "raw_results": [],
                "insight": str(exc),
            }

        # Step 4 — generate insight
        if self._use_llm and raw_results:
            preview_df = pd.DataFrame(raw_results[:20])
            results_preview = preview_df.to_string(index=False)
            try:
                insight = self._generate_insight_llm(question, sql, results_preview)
            except Exception as exc:
                logger.warning("Insight generation failed: %s", exc)
                insight = f"Query returned {len(raw_results)} row(s)."
        elif raw_results:
            # Pattern-match mode: use the canned description
            insight = fallback_description if not self._use_llm else ""
            insight += f" ({len(raw_results)} row(s) returned.)"
        else:
            insight = "The query returned no results."

        logger.info(
            "Query answered — %d row(s) returned for: %s", len(raw_results), question
        )
        return {
            "sql": sql,
            "columns": columns,
            "raw_results": raw_results,
            "insight": insight,
        }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = SQLAgent()

    demo_questions = [
        "Show monthly order trend",
        "Which supplier has the most delayed orders?",
        "Top 10 products by revenue",
        "Order status distribution",
    ]

    for q in demo_questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        result = agent.query(q)
        print(f"SQL: {result['sql']}")
        print(f"Rows: {len(result['raw_results'])}")
        print(f"Insight: {result['insight']}")
