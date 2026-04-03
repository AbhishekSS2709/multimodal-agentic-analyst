"""Feedback collection and storage backed by SQLite.

Persists user ratings, comments, and retrieval logs so that the learning
loop can mine patterns and continuously improve retrieval quality.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import FEEDBACK_DB_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    query           TEXT    NOT NULL,
    retrieved_chunk_ids TEXT,          -- JSON array of chunk IDs
    answer          TEXT    NOT NULL,
    user_rating     INTEGER NOT NULL CHECK (user_rating BETWEEN 1 AND 5),
    user_comment    TEXT,
    query_category  TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    query           TEXT    NOT NULL,
    chunk_id        TEXT    NOT NULL,
    rank            INTEGER NOT NULL,
    score           REAL    NOT NULL,
    was_relevant    INTEGER              -- 1 = relevant, 0 = not, NULL = unknown
);

CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(user_rating);
CREATE INDEX IF NOT EXISTS idx_feedback_category ON feedback(query_category);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_query ON retrieval_logs(query);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_chunk ON retrieval_logs(chunk_id);
"""


# ---------------------------------------------------------------------------
# FeedbackStore
# ---------------------------------------------------------------------------

class FeedbackStore:
    """Collect and query user feedback and retrieval logs.

    Parameters
    ----------
    db_path : Path | str | None
        SQLite database file.  Defaults to ``FEEDBACK_DB_PATH`` from
        project settings.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._db_path = Path(db_path) if db_path else FEEDBACK_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("FeedbackStore ready (db=%s).", self._db_path)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a SQLite connection with WAL mode and row-factory."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript(_SCHEMA_SQL)
        logger.debug("Database schema initialised.")

    # ------------------------------------------------------------------
    # Record feedback
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        query: str,
        answer: str,
        chunks: Sequence[str],
        rating: int,
        comment: Optional[str] = None,
        category: Optional[str] = None,
    ) -> int:
        """Store a user feedback entry.

        Parameters
        ----------
        query : str
            The original user query.
        answer : str
            The generated answer.
        chunks : Sequence[str]
            IDs (or texts) of retrieved chunks.
        rating : int
            User satisfaction rating (1 = terrible ... 5 = excellent).
        comment : str | None
            Free-text user comment.
        category : str | None
            Query category label.

        Returns
        -------
        int
            The ``feedback.id`` of the inserted row.
        """
        import json

        if not 1 <= rating <= 5:
            raise ValueError(f"rating must be between 1 and 5, got {rating}")

        ts = datetime.now(timezone.utc).isoformat()
        chunk_ids_json = json.dumps(list(chunks))

        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback
                    (timestamp, query, retrieved_chunk_ids, answer,
                     user_rating, user_comment, query_category)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, query, chunk_ids_json, answer, rating, comment, category),
            )
            feedback_id: int = cursor.lastrowid  # type: ignore[assignment]

        logger.info("Recorded feedback id=%d (rating=%d, category=%s).",
                     feedback_id, rating, category)
        return feedback_id

    # ------------------------------------------------------------------
    # Record retrieval log
    # ------------------------------------------------------------------

    def record_retrieval(
        self,
        query: str,
        chunk_id: str,
        rank: int,
        score: float,
        relevant: Optional[bool] = None,
    ) -> int:
        """Log a single retrieval event.

        Parameters
        ----------
        query : str
            The user query.
        chunk_id : str
            ID of the retrieved chunk.
        rank : int
            Position in the result list (1-based).
        score : float
            Retrieval score.
        relevant : bool | None
            Whether the chunk was marked relevant (None = unknown).

        Returns
        -------
        int
            The ``retrieval_logs.id`` of the inserted row.
        """
        ts = datetime.now(timezone.utc).isoformat()
        was_relevant: Optional[int] = None
        if relevant is not None:
            was_relevant = 1 if relevant else 0

        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO retrieval_logs
                    (timestamp, query, chunk_id, rank, score, was_relevant)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, query, chunk_id, rank, score, was_relevant),
            )
            log_id: int = cursor.lastrowid  # type: ignore[assignment]

        logger.debug("Recorded retrieval log id=%d (chunk=%s, rank=%d).",
                      log_id, chunk_id, rank)
        return log_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Return aggregated feedback statistics.

        Returns
        -------
        dict
            Keys include ``total_feedback``, ``avg_rating``,
            ``rating_distribution``, ``category_stats``, etc.
        """
        with self._connection() as conn:
            # Total count and average
            row = conn.execute(
                "SELECT COUNT(*) AS cnt, AVG(user_rating) AS avg_r FROM feedback"
            ).fetchone()
            total = row["cnt"]
            avg_rating = round(row["avg_r"], 2) if row["avg_r"] is not None else None

            # Rating distribution
            dist_rows = conn.execute(
                "SELECT user_rating, COUNT(*) AS cnt FROM feedback "
                "GROUP BY user_rating ORDER BY user_rating"
            ).fetchall()
            rating_distribution = {r["user_rating"]: r["cnt"] for r in dist_rows}

            # Per-category stats
            cat_rows = conn.execute(
                "SELECT query_category, COUNT(*) AS cnt, AVG(user_rating) AS avg_r "
                "FROM feedback WHERE query_category IS NOT NULL "
                "GROUP BY query_category ORDER BY avg_r"
            ).fetchall()
            category_stats = {
                r["query_category"]: {
                    "count": r["cnt"],
                    "avg_rating": round(r["avg_r"], 2),
                }
                for r in cat_rows
            }

            # Retrieval stats
            ret_row = conn.execute(
                "SELECT COUNT(*) AS cnt, AVG(score) AS avg_s, "
                "       SUM(CASE WHEN was_relevant = 1 THEN 1 ELSE 0 END) AS rel, "
                "       SUM(CASE WHEN was_relevant = 0 THEN 1 ELSE 0 END) AS irrel "
                "FROM retrieval_logs"
            ).fetchone()
            retrieval_count = ret_row["cnt"]
            retrieval_precision = None
            if ret_row["rel"] is not None and ret_row["irrel"] is not None:
                judged = ret_row["rel"] + ret_row["irrel"]
                if judged > 0:
                    retrieval_precision = round(ret_row["rel"] / judged, 4)

        return {
            "total_feedback": total,
            "avg_rating": avg_rating,
            "rating_distribution": rating_distribution,
            "category_stats": category_stats,
            "total_retrieval_logs": retrieval_count,
            "retrieval_precision": retrieval_precision,
        }

    def get_low_rated_queries(self, threshold: int = 3) -> List[Dict[str, Any]]:
        """Return queries whose rating is at or below *threshold*.

        Parameters
        ----------
        threshold : int
            Maximum rating to include (default 3).

        Returns
        -------
        list[dict]
            Each dict has ``query``, ``answer``, ``rating``, ``comment``,
            ``category``, ``timestamp``.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT query, answer, user_rating, user_comment, "
                "       query_category, timestamp "
                "FROM feedback WHERE user_rating <= ? "
                "ORDER BY user_rating ASC, timestamp DESC",
                (threshold,),
            ).fetchall()

        return [
            {
                "query": r["query"],
                "answer": r["answer"],
                "rating": r["user_rating"],
                "comment": r["user_comment"],
                "category": r["query_category"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    def export_training_data(self) -> List[Dict[str, Any]]:
        """Export feedback data formatted for retriever fine-tuning.

        Produces ``(query, positive_chunks, negative_chunks)`` triplets
        derived from relevance judgments in retrieval logs, enriched with
        user ratings from the feedback table.

        Returns
        -------
        list[dict]
            Each dict contains ``query``, ``positive_chunk_ids``,
            ``negative_chunk_ids``, and ``avg_rating``.
        """
        import json
        from collections import defaultdict

        training_data: list[Dict[str, Any]] = []

        with self._connection() as conn:
            # Gather retrieval relevance judgments grouped by query
            rows = conn.execute(
                "SELECT query, chunk_id, was_relevant FROM retrieval_logs "
                "WHERE was_relevant IS NOT NULL ORDER BY query"
            ).fetchall()

            query_map: Dict[str, Dict[str, list[str]]] = defaultdict(
                lambda: {"positive": [], "negative": []}
            )
            for r in rows:
                key = "positive" if r["was_relevant"] == 1 else "negative"
                query_map[r["query"]][key].append(r["chunk_id"])

            # Enrich with average user rating
            for query, chunks in query_map.items():
                rating_row = conn.execute(
                    "SELECT AVG(user_rating) AS avg_r FROM feedback WHERE query = ?",
                    (query,),
                ).fetchone()
                avg_r = round(rating_row["avg_r"], 2) if rating_row["avg_r"] else None

                training_data.append({
                    "query": query,
                    "positive_chunk_ids": chunks["positive"],
                    "negative_chunk_ids": chunks["negative"],
                    "avg_rating": avg_r,
                })

        logger.info("Exported %d training-data records.", len(training_data))
        return training_data
