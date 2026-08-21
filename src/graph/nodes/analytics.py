"""Analytics specialist — SQL warehouse queries behind a human approval gate.

An agent that can emit SQL is an agent that can emit ``DELETE``.  Rather than
trusting the generator or hard-blocking writes, this node inspects the
statement and calls ``interrupt()`` for anything that is not a plain read.
The graph pauses, the checkpointer persists the state, and a human resumes it
with an approve/reject decision.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.state import AnalystState, Finding

logger = logging.getLogger(__name__)

# Statements that only read. Anything else needs a human.
_READ_ONLY_PREFIXES = ("select", "with", "explain", "pragma")

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|grant|revoke|attach)\b",
    re.IGNORECASE,
)

MAX_ROWS_IN_FINDING = 20


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql.strip()


def needs_approval(sql: str) -> bool:
    """True when ``sql`` is anything other than a single read-only statement.

    Errs toward requiring approval: empty, unparseable, multi-statement, or
    write-shaped SQL all return True.
    """
    cleaned = _strip_sql_comments(sql or "")
    if not cleaned:
        return True

    # Multiple statements — a trailing semicolon alone is fine.
    body = cleaned.rstrip(";")
    if ";" in body:
        return True

    if not body.lower().startswith(_READ_ONLY_PREFIXES):
        return True

    return bool(_WRITE_KEYWORDS.search(body))


def _format_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no rows returned)"
    shown = rows[:MAX_ROWS_IN_FINDING]
    lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in shown]
    if len(rows) > MAX_ROWS_IN_FINDING:
        lines.append(f"... and {len(rows) - MAX_ROWS_IN_FINDING} more rows")
    return "\n".join(lines)


def _require_approval(config: Optional[Dict[str, Any]]) -> bool:
    return bool((config or {}).get("configurable", {}).get("require_approval", True))


def analytics_node(
    state: AnalystState,
    components: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Answer a quantitative question from the SQL warehouse."""
    backend = getattr(components, "sql", None)
    if backend is None:
        return {"findings": [], "trace": ["analytics:unavailable"]}

    question = state.get("question", "")
    try:
        result = backend.analyze(question) or {}
    except Exception as exc:
        logger.warning("SQL analytics failed: %s", exc)
        return {"findings": [], "trace": ["analytics:unavailable"]}

    sql = str(result.get("sql", "") or "")

    # --- human-in-the-loop gate ------------------------------------------
    if _require_approval(config) and needs_approval(sql):
        decision = interrupt({
            "type": "sql_approval",
            "question": question,
            "sql": sql,
            "reason": "Statement is not a plain read-only SELECT.",
        })
        if str(decision).strip().lower() not in ("approve", "approved", "yes", "y"):
            return {
                "findings": [],
                "approval": "denied",
                "trace": ["analytics:denied"],
            }
        state = dict(state)  # type: ignore[assignment]

    rows = result.get("rows", []) or []
    insight = str(result.get("insight", "") or "").strip()

    content_parts = []
    if insight:
        content_parts.append(insight)
    if rows:
        content_parts.append(_format_rows(rows))
    if sql:
        content_parts.append(f"SQL: {sql}")

    if not content_parts:
        return {"findings": [], "trace": ["analytics:no_results"]}

    finding = Finding(
        specialist="analytics",
        content="\n".join(content_parts),
        score=1.0 if rows else 0.3,
        source="sql_warehouse",
        doc_id="sql",
        modality="table",
        metadata={"sql": sql, "row_count": len(rows),
                  "columns": result.get("columns", [])},
    )
    return {
        "findings": [finding],
        "approval": "approved" if sql else None,
        "trace": [f"analytics:emit({len(rows)}rows)"],
    }
