"""FastAPI backend for the Enterprise RAG System.

Provides REST endpoints for question answering, document upload, analytics,
feedback, health checks, system statistics, and evaluation.

All pipeline components are lazily initialised on first request so the server
starts quickly and degrades gracefully when optional dependencies are missing.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_DIR,
    EVALUATION_DIR,
    FEEDBACK_DB_PATH,
    SQLITE_DB_PATH,
    VECTOR_DB_DIR,
)

from src.api.models import (
    AnalyticsResponse,
    AnalystQueryRequest,
    AnalystResponse,
    AnalystResumeRequest,
    AskRequest,
    AskResponse,
    EvaluationMetric,
    EvaluationResponse,
    FeedbackRequest,
    FeedbackResponse,
    GraphTopologyResponse,
    HealthResponse,
    SourceDocument,
    StatsResponse,
    UploadResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise RAG System",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public-demo guard rails (no-op unless DEMO_MODE=1).
from src.api.demo_guard import DemoGuardMiddleware, demo_mode_enabled

if demo_mode_enabled():
    app.add_middleware(DemoGuardMiddleware)

# Mount static directory if it exists
_static_dir = PROJECT_ROOT / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ---------------------------------------------------------------------------
# Global state (lazy-initialised singletons)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "ingest_pipeline": None,
    "chunker": None,
    "embedding_engine": None,
    "vector_store": None,
    "retriever": None,
    "rag_pipeline": None,
    "sql_analytics_pipeline": None,
    "feedback_store": None,
    "queries_answered": 0,
    "docs_uploaded": 0,
}

# ---------------------------------------------------------------------------
# Lazy component initialisation helpers
# ---------------------------------------------------------------------------

def _get_ingest_pipeline():
    """Return (or create) the ingestion pipeline."""
    if _state["ingest_pipeline"] is None:
        try:
            from src.ingestion.pipeline import IngestPipeline
            _state["ingest_pipeline"] = IngestPipeline()
            logger.info("IngestPipeline initialised.")
        except Exception as exc:
            logger.error("Failed to initialise IngestPipeline: %s", exc)
            raise HTTPException(status_code=503, detail=f"Ingestion pipeline unavailable: {exc}")
    return _state["ingest_pipeline"]


def _get_chunker():
    """Return (or create) the semantic chunker."""
    if _state["chunker"] is None:
        try:
            from src.chunking.chunker import SemanticChunker
            _state["chunker"] = SemanticChunker()
            logger.info("SemanticChunker initialised.")
        except Exception as exc:
            logger.error("Failed to initialise SemanticChunker: %s", exc)
            raise HTTPException(status_code=503, detail=f"Chunker unavailable: {exc}")
    return _state["chunker"]


def _get_embedding_engine():
    """Return (or create) the embedding engine."""
    if _state["embedding_engine"] is None:
        try:
            from src.embedding.embed_chunks import EmbeddingEngine
            _state["embedding_engine"] = EmbeddingEngine()
            logger.info("EmbeddingEngine initialised.")
        except Exception as exc:
            logger.error("Failed to initialise EmbeddingEngine: %s", exc)
            raise HTTPException(status_code=503, detail=f"Embedding engine unavailable: {exc}")
    return _state["embedding_engine"]


def _get_vector_store():
    """Return (or create) the vector store."""
    if _state["vector_store"] is None:
        try:
            from src.embedding.store_vector_db import VectorStore
            _state["vector_store"] = VectorStore()
            logger.info("VectorStore initialised.")
        except Exception as exc:
            logger.error("Failed to initialise VectorStore: %s", exc)
            raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}")
    return _state["vector_store"]


def _get_retriever():
    """Return (or create) the Retriever."""
    if _state["retriever"] is None:
        try:
            from src.embedding.retrieve import Retriever
            ee = _get_embedding_engine()
            vs = _get_vector_store()
            _state["retriever"] = Retriever(embedding_engine=ee, vector_store=vs)
            logger.info("Retriever initialised.")
        except Exception as exc:
            logger.error("Failed to initialise Retriever: %s", exc)
            raise HTTPException(status_code=503, detail=f"Retriever unavailable: {exc}")
    return _state["retriever"]


def _get_rag_pipeline():
    """Return (or create) the full RAG QA pipeline."""
    if _state["rag_pipeline"] is None:
        try:
            from src.qa_pipeline import RAGPipeline
            retriever = _get_retriever()
            _state["rag_pipeline"] = RAGPipeline(retriever=retriever)
            logger.info("RAGPipeline initialised.")
        except Exception as exc:
            logger.warning("RAGPipeline unavailable: %s. Will use direct retrieval + LLM.", exc)
            return None
    return _state["rag_pipeline"]


def _get_sql_analytics_pipeline():
    """Return (or create) the SQL analytics pipeline."""
    if _state["sql_analytics_pipeline"] is None:
        try:
            from src.sql_tool.sql_pipeline import SQLAnalyticsPipeline
            _state["sql_analytics_pipeline"] = SQLAnalyticsPipeline()
            logger.info("SQLAnalyticsPipeline initialised.")
        except Exception as exc:
            logger.warning("SQLAnalyticsPipeline unavailable: %s", exc)
            return None
    return _state["sql_analytics_pipeline"]


def _get_feedback_store():
    """Return (or create) the feedback store."""
    if _state["feedback_store"] is None:
        try:
            from src.feedback.feedback_store import FeedbackStore
            _state["feedback_store"] = FeedbackStore()
            logger.info("FeedbackStore initialised.")
        except Exception as exc:
            logger.warning("FeedbackStore unavailable: %s. Using fallback.", exc)
            return None
    return _state["feedback_store"]


# ---------------------------------------------------------------------------
# Feedback fallback helpers (used when FeedbackStore is unavailable)
# ---------------------------------------------------------------------------

def _ensure_feedback_db() -> None:
    """Create a minimal feedback table if the full FeedbackStore is unavailable."""
    FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _save_feedback_fallback(query_id: str, rating: int, comment: str) -> str:
    """Insert a feedback record using the fallback SQLite table."""
    _ensure_feedback_db()
    feedback_id = uuid.uuid4().hex[:12]
    conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
    try:
        conn.execute(
            "INSERT INTO feedback (id, query_id, rating, comment) VALUES (?, ?, ?, ?)",
            (feedback_id, query_id, rating, comment),
        )
        conn.commit()
    finally:
        conn.close()
    return feedback_id


def _get_feedback_count() -> int:
    """Return total feedback entries from either FeedbackStore or fallback."""
    # Try the full FeedbackStore first
    store = _get_feedback_store()
    if store is not None:
        try:
            stats = store.get_feedback_stats()
            return stats.get("total_feedback", 0)
        except Exception:
            pass

    # Fallback: query the DB directly
    if not FEEDBACK_DB_PATH.exists():
        return 0
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
        cur = conn.execute("SELECT COUNT(*) FROM feedback")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# LLM helpers (graceful when OpenAI is unavailable)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, system: str = "You are a helpful enterprise assistant.") -> str:
    """Call the best available LLM provider (free or paid)."""
    try:
        from src.llm_provider import call_llm
        return call_llm(prompt, system)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return f"[LLM unavailable: {exc}]"


def _classify_query(question: str) -> str:
    """Classify a question into a query type using keyword heuristics."""
    q_lower = question.lower()

    sql_keywords = [
        "how many", "total", "count", "average", "sum", "revenue",
        "top", "most", "least", "trend", "by region", "by supplier",
        "by month", "compare", "percentage", "growth", "decline",
    ]
    if any(kw in q_lower for kw in sql_keywords):
        return "analytics"

    graph_keywords = [
        "related to", "connected", "relationship", "entity", "link",
        "graph", "knowledge graph", "network",
    ]
    if any(kw in q_lower for kw in graph_keywords):
        return "knowledge_graph"

    return "retrieval"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components: Dict[str, str] = {}

    # Check vector store
    try:
        from src.embedding.store_vector_db import VectorStore
        components["vector_store"] = "ok"
    except Exception:
        components["vector_store"] = "unavailable"

    # Check embedding model availability
    try:
        from src.embedding.embed_chunks import EmbeddingEngine
        components["embedding_engine"] = "ok"
    except Exception:
        components["embedding_engine"] = "unavailable"

    # Check SQLite database
    if SQLITE_DB_PATH.exists():
        components["sql_database"] = "ok"
    else:
        components["sql_database"] = "no database"

    # Check feedback database
    try:
        _ensure_feedback_db()
        components["feedback_db"] = "ok"
    except Exception:
        components["feedback_db"] = "unavailable"

    # Check LLM provider
    try:
        from src.llm_provider import get_active_provider
        provider = get_active_provider()
        components["llm"] = f"ok ({provider})" if provider != "none" else "no provider"
    except Exception:
        components["llm"] = "unavailable"

    overall = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components)


@app.get("/api/stats", response_model=StatsResponse)
async def system_stats():
    """Return system-level statistics."""
    total_chunks = 0
    try:
        vs = _get_vector_store()
        total_chunks = vs.total_vectors
    except Exception:
        pass

    total_docs = _state.get("docs_uploaded", 0)
    queries_answered = _state.get("queries_answered", 0)
    feedback_count = _get_feedback_count()

    return StatsResponse(
        total_docs=total_docs,
        total_chunks=total_chunks,
        queries_answered=queries_answered,
        feedback_count=feedback_count,
    )


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Answer a question using the RAG pipeline."""
    start_time = time.time()
    question = request.question.strip()
    query_type = _classify_query(question)
    query_id = uuid.uuid4().hex[:12]

    logger.info("Ask request: query_type=%s, question='%s'", query_type, question[:100])

    # --- Analytics path -------------------------------------------------------
    if query_type == "analytics":
        try:
            analytics_result = _run_analytics(question)
            _state["queries_answered"] = _state.get("queries_answered", 0) + 1
            return AskResponse(
                answer=analytics_result.get("insight", "See the analytics tab for detailed results."),
                sources=[],
                confidence=0.85,
                query_type="analytics",
                chart_html=analytics_result.get("chart_html"),
            )
        except Exception as exc:
            logger.warning("Analytics path failed, falling back to retrieval: %s", exc)
            query_type = "retrieval"

    # --- Try the full RAGPipeline first (if available) ------------------------
    rag = _get_rag_pipeline()
    if rag is not None:
        try:
            result = rag.answer(question)
            sources: List[SourceDocument] = []
            if request.include_sources:
                for src in result.get("sources", []):
                    sources.append(
                        SourceDocument(
                            text=src.get("text_preview", "")[:500],
                            source=src.get("source", "unknown"),
                            score=round(float(src.get("score", 0.0)), 4),
                            metadata={
                                "chunk_id": src.get("chunk_id", ""),
                                "doc_id": src.get("doc_id", ""),
                            },
                        )
                    )

            _state["queries_answered"] = _state.get("queries_answered", 0) + 1
            elapsed = time.time() - start_time
            confidence = result.get("confidence", 0.0)
            logger.info("Ask completed via RAGPipeline in %.2fs (confidence=%.2f)", elapsed, confidence)

            return AskResponse(
                answer=result.get("answer", ""),
                sources=sources,
                confidence=round(confidence, 3),
                query_type=query_type,
                chart_html=None,
            )
        except Exception as exc:
            logger.warning("RAGPipeline.answer() failed: %s. Falling back to direct retrieval.", exc)

    # --- Fallback: direct retrieval + LLM ------------------------------------
    sources = []
    context_texts: List[str] = []
    top_score = 0.0

    try:
        retriever = _get_retriever()
        results = retriever.retrieve(question, top_k=request.top_k)

        for chunk, score in results:
            if request.include_sources:
                sources.append(
                    SourceDocument(
                        text=chunk.text[:500],
                        source=chunk.metadata.get("source", "unknown"),
                        score=round(float(score), 4),
                        metadata=chunk.metadata,
                    )
                )
            context_texts.append(chunk.text)
            if score > top_score:
                top_score = float(score)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)

    # Build prompt with retrieved context
    if context_texts:
        context_block = "\n\n---\n\n".join(context_texts[:5])
        prompt = (
            f"Answer the following question using ONLY the context provided below.\n"
            f"If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    else:
        prompt = (
            f"The retrieval system returned no relevant documents for this question.\n"
            f"Please answer to the best of your ability, or explain that no relevant "
            f"documents were found.\n\nQuestion: {question}\n\nAnswer:"
        )

    answer = _call_llm(prompt)
    confidence = min(top_score, 1.0) if top_score > 0 else 0.3

    _state["queries_answered"] = _state.get("queries_answered", 0) + 1

    elapsed = time.time() - start_time
    logger.info("Ask completed (fallback) in %.2fs (confidence=%.2f)", elapsed, confidence)

    return AskResponse(
        answer=answer,
        sources=sources,
        confidence=round(confidence, 3),
        query_type=query_type,
        chart_html=None,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF, CSV, or TXT)."""
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    allowed_extensions = {
        ".pdf", ".csv", ".txt", ".log", ".md", ".eml",
        ".docx", ".pptx", ".xlsx", ".xls",
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
        ".mp4", ".avi", ".mov",
        ".mp3", ".wav", ".m4a",
        ".html", ".htm",
        ".py", ".js", ".ts", ".java", ".go", ".rs",
        ".json", ".yaml", ".yml",
    }
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    # Write uploaded file to a temporary location
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{filename}"

    try:
        temp_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Run through ingestion pipeline
    try:
        pipeline = _get_ingest_pipeline()
        documents = pipeline.ingest(str(temp_path))
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", filename, exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    if not documents:
        return UploadResponse(
            status="warning",
            doc_id="",
            chunks_created=0,
        )

    doc_id = documents[0].doc_id

    # Chunk the documents
    chunks_created = 0
    try:
        chunker = _get_chunker()
        all_chunks = chunker.chunk_documents(documents)
        chunks_created = len(all_chunks)

        if all_chunks:
            # Embed and store
            ee = _get_embedding_engine()
            embedded = ee.embed_chunks(all_chunks)
            chunks_only = [c for c, _ in embedded]
            embeddings_only = [e for _, e in embedded]

            vs = _get_vector_store()
            vs.store_embeddings(chunks_only, embeddings_only)
            vs.save()
            ee.save_cache()

            logger.info(
                "Uploaded %s: %d documents, %d chunks stored.",
                filename, len(documents), chunks_created,
            )
    except Exception as exc:
        logger.error("Chunking/embedding failed for %s: %s", filename, exc)
        return UploadResponse(
            status="partial",
            doc_id=doc_id,
            chunks_created=0,
        )

    _state["docs_uploaded"] = _state.get("docs_uploaded", 0) + 1

    return UploadResponse(
        status="success",
        doc_id=doc_id,
        chunks_created=chunks_created,
    )


@app.get("/api/analytics", response_model=AnalyticsResponse)
async def run_analytics(question: str = Query(..., min_length=1)):
    """Run an analytics query against the SQL database."""
    try:
        result = _run_analytics(question)
        return AnalyticsResponse(
            sql=result.get("sql", ""),
            results=result.get("results", []),
            insight=result.get("insight", ""),
            chart_html=result.get("chart_html", ""),
        )
    except Exception as exc:
        logger.error("Analytics query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analytics query failed: {exc}")


def _run_analytics(question: str) -> Dict[str, Any]:
    """Internal helper: run an analytics query and return structured results.

    Attempts to use SQLAnalyticsPipeline first (which includes SQLAgent +
    ChartGenerator). Falls back to direct LLM-based SQL generation.
    """
    if not SQLITE_DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="SQL database not available. Upload a CSV or run database setup first.",
        )

    # --- Try the integrated SQLAnalyticsPipeline first ---
    pipeline = _get_sql_analytics_pipeline()
    if pipeline is not None:
        try:
            result = pipeline.analyze(question)
            return {
                "sql": result.get("sql", ""),
                "results": result.get("results", [])[:100],
                "insight": result.get("insight", ""),
                "chart_html": result.get("chart_html", ""),
            }
        except Exception as exc:
            logger.warning("SQLAnalyticsPipeline failed: %s. Using fallback.", exc)

    # --- Fallback: direct LLM + SQL execution ---
    try:
        from src.sql_tool.db_setup import get_schema
        schema = get_schema()
    except Exception:
        schema = (
            "TABLE orders (order_id, customer_name, product, quantity, "
            "unit_price, total_amount, order_date, delivery_date, status, "
            "supplier, region, priority)"
        )

    sql_prompt = (
        f"Given this SQLite database schema:\n\n{schema}\n\n"
        f"Write a SQL query to answer: {question}\n\n"
        f"Return ONLY the SQL query, nothing else. No markdown, no explanation."
    )
    sql_query = _call_llm(
        sql_prompt,
        system="You are a SQL expert. Return only valid SQLite SQL queries.",
    ).strip()

    # Clean up markdown code fences
    if sql_query.startswith("```"):
        lines = sql_query.split("\n")
        sql_query = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    # Execute the SQL
    results: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql_query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        results = [dict(zip(columns, row)) for row in rows]
        conn.close()
    except Exception as exc:
        logger.error("SQL execution failed: %s\nQuery: %s", exc, sql_query)
        return {
            "sql": sql_query,
            "results": [],
            "insight": f"SQL execution failed: {exc}",
            "chart_html": "",
        }

    # Generate insight
    results_preview = json.dumps(results[:20], default=str, indent=2)
    insight_prompt = (
        f"Question: {question}\n"
        f"SQL: {sql_query}\n"
        f"Results (first 20 rows):\n{results_preview}\n\n"
        f"Provide a concise business insight (2-3 sentences) based on these results."
    )
    insight = _call_llm(insight_prompt)

    # Generate chart HTML using ChartGenerator if available
    chart_html = _generate_chart_html(question, results)

    return {
        "sql": sql_query,
        "results": results[:100],
        "insight": insight,
        "chart_html": chart_html,
    }


def _generate_chart_html(question: str, results: List[Dict]) -> str:
    """Generate a Plotly chart HTML string from analytics results."""
    if not results:
        return ""

    # Try the ChartGenerator from the visualization module
    try:
        from src.visualization.chart_generator import ChartGenerator
        cg = ChartGenerator()
        fig = cg.auto_chart(data=results, query_context=question)
        if fig is not None:
            return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as exc:
        logger.warning("ChartGenerator failed: %s. Trying plotly directly.", exc)

    # Fallback: basic plotly chart
    try:
        import pandas as pd
        import plotly.express as px
        import plotly.io as pio

        df = pd.DataFrame(results)
        columns = list(df.columns)

        if len(columns) < 2:
            return ""

        q_lower = question.lower()
        x_col = columns[0]

        # Find best numeric y-column
        numeric_cols = [c for c in columns if pd.to_numeric(df[c], errors="coerce").notna().all()]
        y_col = columns[1]
        if numeric_cols:
            y_col = numeric_cols[0] if numeric_cols[0] != x_col else (
                numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            )

        if any(kw in q_lower for kw in ["trend", "over time", "monthly", "by month", "by year"]):
            fig = px.line(df, x=x_col, y=y_col, title=question[:80], markers=True)
        elif any(kw in q_lower for kw in ["distribution", "breakdown", "proportion", "percentage"]):
            fig = px.pie(df, names=x_col, values=y_col, title=question[:80])
        elif len(df) <= 20:
            fig = px.bar(df, x=x_col, y=y_col, title=question[:80])
        else:
            fig = px.bar(df.head(20), x=x_col, y=y_col, title=question[:80] + " (top 20)")

        fig.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    except Exception as exc:
        logger.warning("Chart generation fallback failed: %s", exc)
        return ""


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a previous query."""
    try:
        # Try the full FeedbackStore first
        store = _get_feedback_store()
        if store is not None:
            store.record_feedback(
                query=request.query_id,
                answer="",
                chunks=[],
                rating=request.rating,
                comment=request.comment or None,
            )
            logger.info("Feedback saved via FeedbackStore: query=%s, rating=%d", request.query_id, request.rating)
        else:
            # Use the fallback
            _save_feedback_fallback(request.query_id, request.rating, request.comment)
            logger.info("Feedback saved via fallback: query=%s, rating=%d", request.query_id, request.rating)

        return FeedbackResponse(status="success")
    except Exception as exc:
        logger.error("Failed to save feedback: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {exc}")


@app.get("/api/assets/{asset_id}")
async def get_asset(asset_id: str):
    try:
        from src.ingestion.asset_store import AssetStore
        store = AssetStore()
        path = store.get_path(asset_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/quota")
async def gemini_quota():
    try:
        from config.settings import GEMINI_API_KEY
        from src.gemini.client import GeminiClient
        if not GEMINI_API_KEY:
            return {"daily_remaining": 0, "rpm_remaining": 0, "daily_used": 0, "status": "no_api_key"}
        client = GeminiClient(api_key=GEMINI_API_KEY)
        return client.quota_remaining()
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/metrics")
async def system_metrics():
    total_vectors = 0
    visual_vectors = 0
    try:
        vs = _get_vector_store()
        total_vectors = vs.total_vectors
    except Exception:
        pass
    try:
        from src.embedding.visual_store import VisualVectorStore
        vvs = VisualVectorStore()
        visual_vectors = vvs.total_vectors
    except Exception:
        pass
    gemini_quota = {}
    try:
        from config.settings import GEMINI_API_KEY
        if GEMINI_API_KEY:
            from src.gemini.client import GeminiClient
            client = GeminiClient(api_key=GEMINI_API_KEY)
            gemini_quota = client.quota_remaining()
    except Exception:
        pass
    return {
        "queries_answered": _state.get("queries_answered", 0),
        "docs_uploaded": _state.get("docs_uploaded", 0),
        "total_vectors": total_vectors,
        "visual_vectors": visual_vectors,
        "gemini_quota": gemini_quota,
    }


@app.get("/api/evaluation", response_model=EvaluationResponse)
async def run_evaluation():
    """Run the evaluation pipeline and return metrics."""
    metrics: List[Dict[str, Any]] = []
    summary = ""
    per_category: Dict[str, Any] = {}

    # Try the full evaluation module
    try:
        from src.evaluation.eval_pipeline import RAGEvaluator
        from src.evaluation.test_cases import TEST_CASES

        evaluator = RAGEvaluator()
        eval_results = evaluator.evaluate_batch(TEST_CASES)

        # Extract aggregated metrics
        aggregated = eval_results.get("aggregated", {})
        for metric_name in [
            "retrieval_precision", "retrieval_recall",
            "answer_relevance", "faithfulness", "keyword_hit_rate",
        ]:
            val = aggregated.get(metric_name)
            if val is not None and isinstance(val, dict):
                metrics.append({
                    "name": metric_name,
                    "value": float(val.get("mean", 0.0)),
                    "category": "overall",
                })

        num_cases = aggregated.get("num_cases", 0)
        summary = f"Evaluated {num_cases} test cases across {len(eval_results.get('per_category', {}))} categories."

        # Per-category breakdown
        for cat_name, cat_agg in eval_results.get("per_category", {}).items():
            cat_metrics = {}
            for m_name in ["retrieval_precision", "retrieval_recall", "answer_relevance", "faithfulness"]:
                val = cat_agg.get(m_name)
                if val and isinstance(val, dict):
                    cat_metrics[m_name] = val.get("mean", 0.0)
            cat_metrics["num_cases"] = cat_agg.get("num_cases", 0)
            score = sum(
                v for k, v in cat_metrics.items()
                if k != "num_cases" and isinstance(v, (int, float))
            ) / max(sum(1 for k, v in cat_metrics.items() if k != "num_cases" and isinstance(v, (int, float))), 1)
            per_category[cat_name] = {
                "score": round(score, 4),
                "checks": list(cat_metrics.keys()),
                **cat_metrics,
            }

    except Exception as exc:
        logger.warning("Full evaluation unavailable: %s. Running basic checks.", exc)
        basic = _run_basic_evaluation()
        metrics = basic["metrics"]
        summary = basic["summary"]
        per_category = basic["per_category"]

    return EvaluationResponse(
        status="ok",
        metrics=[EvaluationMetric(**m) for m in metrics],
        summary=summary,
        per_category=per_category,
    )


def _run_basic_evaluation() -> Dict[str, Any]:
    """Run basic system health evaluation as a fallback."""
    metrics = []
    checks = {"passed": 0, "total": 0}

    # Check 1: Vector store has data
    checks["total"] += 1
    try:
        vs = _get_vector_store()
        has_vectors = vs.total_vectors > 0
        metrics.append({
            "name": "vector_store_populated",
            "value": 1.0 if has_vectors else 0.0,
            "category": "data_readiness",
        })
        if has_vectors:
            checks["passed"] += 1
    except Exception:
        metrics.append({"name": "vector_store_populated", "value": 0.0, "category": "data_readiness"})

    # Check 2: SQL database exists
    checks["total"] += 1
    db_exists = SQLITE_DB_PATH.exists()
    metrics.append({
        "name": "sql_database_available",
        "value": 1.0 if db_exists else 0.0,
        "category": "data_readiness",
    })
    if db_exists:
        checks["passed"] += 1

    # Check 3: Embedding engine loadable
    checks["total"] += 1
    try:
        _get_embedding_engine()
        metrics.append({"name": "embedding_engine_available", "value": 1.0, "category": "components"})
        checks["passed"] += 1
    except Exception:
        metrics.append({"name": "embedding_engine_available", "value": 0.0, "category": "components"})

    # Check 4: LLM reachable
    checks["total"] += 1
    llm_test = _call_llm("Reply with 'ok'.")
    llm_ok = "ok" in llm_test.lower() and "[" not in llm_test
    metrics.append({
        "name": "llm_reachable",
        "value": 1.0 if llm_ok else 0.0,
        "category": "components",
    })
    if llm_ok:
        checks["passed"] += 1

    score = checks["passed"] / max(checks["total"], 1)
    summary = f"Basic evaluation: {checks['passed']}/{checks['total']} checks passed ({score:.0%} readiness)."

    per_category = {
        "data_readiness": {
            "checks": ["vector_store_populated", "sql_database_available"],
            "score": sum(
                m["value"] for m in metrics if m["category"] == "data_readiness"
            ) / max(
                sum(1 for m in metrics if m["category"] == "data_readiness"), 1
            ),
        },
        "components": {
            "checks": ["embedding_engine_available", "llm_reachable"],
            "score": sum(
                m["value"] for m in metrics if m["category"] == "components"
            ) / max(
                sum(1 for m in metrics if m["category"] == "components"), 1
            ),
        },
    }

    return {"metrics": metrics, "summary": summary, "per_category": per_category}


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple landing page."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Enterprise RAG System</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 60px auto; text-align: center;">
        <h1>Enterprise RAG System</h1>
        <p>API is running. Visit <a href="/docs">/docs</a> for the interactive API documentation.</p>
        <p>Streamlit UI runs on <a href="http://localhost:8501">http://localhost:8501</a>.</p>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# v2 — LangGraph agentic analyst
#
# The v1 endpoints above continue to serve the original linear pipeline.
# These add the supervisor-orchestrated multi-agent graph alongside it.
# ---------------------------------------------------------------------------

@app.get("/api/v2/graph", response_model=GraphTopologyResponse)
async def analyst_graph_topology():
    """Return the compiled analyst graph topology and its current mode."""
    from src.graph.build import graph_mermaid
    from src.graph.llm import llm_mode
    from src.graph.observability import tracing_enabled

    return GraphTopologyResponse(
        mermaid=graph_mermaid(),
        nodes=["supervisor", "document", "visual", "analytics", "graph",
               "synthesizer", "verifier"],
        llm_mode=llm_mode(),
        tracing=tracing_enabled(),
    )


@app.post("/api/v2/query", response_model=AnalystResponse)
async def analyst_query(request: AnalystQueryRequest):
    """Run a question through the multi-agent analyst graph."""
    from src.graph.build import run_query

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = run_query(
            question,
            thread_id=request.thread_id,
            require_approval=request.require_approval,
        )
    except Exception as exc:
        logger.error("Analyst graph failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analyst graph failed: {exc}")

    return AnalystResponse(**result)


@app.post("/api/v2/resume", response_model=AnalystResponse)
async def analyst_resume(request: AnalystResumeRequest):
    """Resume a thread that paused for human approval."""
    from src.graph.build import resume_query

    decision = request.decision.strip().lower()
    if decision not in ("approve", "approved", "yes", "y", "reject", "rejected", "no", "n"):
        raise HTTPException(
            status_code=400,
            detail="Decision must be 'approve' or 'reject'.",
        )

    try:
        result = resume_query(request.thread_id, decision)
    except Exception as exc:
        logger.warning("Resume failed for thread %s: %s", request.thread_id, exc)
        raise HTTPException(
            status_code=404,
            detail=f"No resumable thread '{request.thread_id}': {exc}",
        )

    return AnalystResponse(**result)
