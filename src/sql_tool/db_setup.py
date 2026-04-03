"""SQLite database setup — loads CSV data into a queryable orders table.

Sample queries (for reference / LLM context):
  -- Total revenue by supplier
  SELECT supplier, SUM(total_amount) AS revenue
  FROM orders GROUP BY supplier ORDER BY revenue DESC;

  -- Monthly order trend
  SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS order_count,
         SUM(total_amount) AS revenue
  FROM orders GROUP BY month ORDER BY month;

  -- Delayed orders by region
  SELECT region, COUNT(*) AS delayed_count
  FROM orders WHERE status = 'Delayed'
  GROUP BY region ORDER BY delayed_count DESC;

  -- Top 10 products by revenue
  SELECT product, SUM(total_amount) AS revenue
  FROM orders GROUP BY product ORDER BY revenue DESC LIMIT 10;

  -- Average order value by priority
  SELECT priority, ROUND(AVG(total_amount), 2) AS avg_order_value
  FROM orders GROUP BY priority;
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
    text,
)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import SQLITE_DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

metadata_obj = MetaData()

orders_table = Table(
    "orders",
    metadata_obj,
    Column("order_id", String, primary_key=True),
    Column("customer_name", String, nullable=False),
    Column("product", String, nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("unit_price", Float, nullable=False),
    Column("total_amount", Float, nullable=False),
    Column("order_date", String, nullable=True),      # stored as ISO text for strftime
    Column("delivery_date", String, nullable=True),
    Column("status", String, nullable=True),
    Column("supplier", String, nullable=True),
    Column("region", String, nullable=True),
    Column("priority", String, nullable=True),
)

# Indexes on commonly queried columns
_indexes = [
    Index("idx_orders_status", orders_table.c.status),
    Index("idx_orders_supplier", orders_table.c.supplier),
    Index("idx_orders_region", orders_table.c.region),
    Index("idx_orders_priority", orders_table.c.priority),
    Index("idx_orders_order_date", orders_table.c.order_date),
    Index("idx_orders_product", orders_table.c.product),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _engine(db_path: Optional[Path] = None):
    """Create a SQLAlchemy engine for the SQLite database."""
    path = db_path or SQLITE_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


def setup_database(
    csv_path: str,
    *,
    db_path: Optional[Path] = None,
    if_exists: str = "replace",
) -> Path:
    """Load a CSV file into the SQLite ``orders`` table.

    Args:
        csv_path: Path to the source CSV file.
        db_path: Override for the database file location (defaults to
            ``config.settings.SQLITE_DB_PATH``).
        if_exists: Pandas ``to_sql`` behaviour — ``"replace"`` (default),
            ``"append"``, or ``"fail"``.

    Returns:
        The resolved path to the created database file.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If the CSV has no rows.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(str(csv_file))
    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # Normalise column names to snake_case expected by our schema
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    engine = _engine(db_path)

    # Create schema (tables + indexes)
    metadata_obj.create_all(engine, checkfirst=True)

    # Load data
    df.to_sql("orders", con=engine, if_exists=if_exists, index=False)

    resolved = (db_path or SQLITE_DB_PATH).resolve()
    row_count = len(df)
    logger.info(
        "Database ready at %s — loaded %d rows into 'orders' table.",
        resolved,
        row_count,
    )
    return resolved


def get_schema(db_path: Optional[Path] = None) -> str:
    """Return a human-readable description of every table and column.

    The output is designed to be injected into an LLM prompt so the model
    understands the available data.

    Returns:
        A multi-line string describing the database schema.
    """
    engine = _engine(db_path)
    inspector = inspect(engine)

    lines: list[str] = []
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        col_descriptions = []
        for col in columns:
            col_type = str(col["type"])
            nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
            pk = " PRIMARY KEY" if col.get("primary_key", False) or col.get("autoincrement", False) else ""
            col_descriptions.append(f"    {col['name']}  {col_type}  {nullable}{pk}")

        indexes = inspector.get_indexes(table_name)
        idx_lines = []
        for idx in indexes:
            idx_lines.append(f"    INDEX {idx['name']} ON ({', '.join(idx['column_names'])})")

        lines.append(f"TABLE {table_name} (")
        lines.extend(col_descriptions)
        lines.append(")")
        if idx_lines:
            lines.append("Indexes:")
            lines.extend(idx_lines)
        lines.append("")

    # Append a quick row-count summary
    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            lines.append(f"-- {table_name}: {count} rows")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    csv = _sys.argv[1] if len(_sys.argv) > 1 else str(DATA_DIR / "orders.csv")
    db = setup_database(csv)
    print(f"Database created at: {db}")
    print()
    print(get_schema())
