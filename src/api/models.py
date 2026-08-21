"""Pydantic models for all API request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request body for POST /api/ask."""

    question: str = Field(..., min_length=1, description="The question to ask the RAG system.")
    use_hybrid: bool = Field(True, description="Use hybrid (vector + BM25) retrieval.")
    include_sources: bool = Field(True, description="Include source documents in the response.")
    top_k: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve.")


class FeedbackRequest(BaseModel):
    """Request body for POST /api/feedback."""

    query_id: str = Field(..., min_length=1, description="Identifier of the query being rated.")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent).")
    comment: str = Field("", description="Optional free-text feedback.")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SourceDocument(BaseModel):
    """A single source document returned with an answer."""

    text: str = Field(..., description="Chunk text.")
    source: str = Field("", description="Originating file name or path.")
    score: float = Field(0.0, description="Relevance score.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata.")


class AskResponse(BaseModel):
    """Response body for POST /api/ask."""

    answer: str = Field(..., description="Generated answer.")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents.")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score.")
    query_type: str = Field("unknown", description="Classified query type.")
    chart_html: Optional[str] = Field(None, description="Optional Plotly chart HTML.")


class UploadResponse(BaseModel):
    """Response body for POST /api/upload."""

    status: str = Field(..., description="Upload status message.")
    doc_id: str = Field("", description="Assigned document identifier.")
    chunks_created: int = Field(0, description="Number of chunks produced.")


class AnalyticsResponse(BaseModel):
    """Response body for GET /api/analytics."""

    sql: str = Field("", description="Generated SQL query.")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Query result rows.")
    insight: str = Field("", description="LLM-generated insight.")
    chart_html: str = Field("", description="Plotly chart as HTML.")


class FeedbackResponse(BaseModel):
    """Response body for POST /api/feedback."""

    status: str = Field(..., description="Feedback submission status.")


class HealthResponse(BaseModel):
    """Response body for GET /api/health."""

    status: str = Field("ok")
    components: Dict[str, str] = Field(default_factory=dict, description="Component health statuses.")


class StatsResponse(BaseModel):
    """Response body for GET /api/stats."""

    total_docs: int = Field(0, description="Total documents ingested.")
    total_chunks: int = Field(0, description="Total chunks in vector store.")
    queries_answered: int = Field(0, description="Total queries answered.")
    feedback_count: int = Field(0, description="Total feedback entries.")


class EvaluationMetric(BaseModel):
    """A single evaluation metric."""

    name: str
    value: float
    category: str = ""


class EvaluationResponse(BaseModel):
    """Response body for GET /api/evaluation."""

    status: str = Field("ok")
    metrics: List[EvaluationMetric] = Field(default_factory=list)
    summary: str = Field("")
    per_category: Dict[str, Any] = Field(default_factory=dict)


class QuotaResponse(BaseModel):
    daily_remaining: int = 0
    rpm_remaining: int = 0
    daily_used: int = 0
    status: str = "ok"


class MetricsResponse(BaseModel):
    queries_answered: int = 0
    docs_uploaded: int = 0
    total_vectors: int = 0
    visual_vectors: int = 0
    gemini_quota: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# v2 — LangGraph agentic analyst
# ---------------------------------------------------------------------------

class AnalystQueryRequest(BaseModel):
    """Request body for POST /api/v2/query."""

    question: str = Field(..., min_length=1, description="The question to analyse.")
    thread_id: Optional[str] = Field(
        None, description="Conversation thread to continue; a new one is created if omitted."
    )
    require_approval: bool = Field(
        True, description="Pause for human approval before non-read-only SQL."
    )


class AnalystResumeRequest(BaseModel):
    """Request body for POST /api/v2/resume."""

    thread_id: str = Field(..., min_length=1, description="Thread paused at an interrupt.")
    decision: str = Field(..., min_length=1, description="'approve' or 'reject'.")


class AnalystResponse(BaseModel):
    """Response body for the v2 analyst endpoints."""

    answer: str = Field("", description="Final cited answer.")
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    verification: Dict[str, Any] = Field(default_factory=dict)
    specialists: List[str] = Field(default_factory=list)
    retry_count: int = Field(0, description="Synthesis attempts made.")
    trace: List[str] = Field(default_factory=list, description="Node-by-node breadcrumbs.")
    thread_id: str = Field("", description="Thread this run belongs to.")
    interrupted: bool = Field(False, description="True when paused for human approval.")
    interrupt_payload: Optional[Dict[str, Any]] = Field(
        None, description="What the human is being asked to approve."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphTopologyResponse(BaseModel):
    """Response body for GET /api/v2/graph."""

    mermaid: str = Field("", description="Mermaid rendering of the graph.")
    nodes: List[str] = Field(default_factory=list)
    llm_mode: str = Field("heuristic", description="'llm' or 'heuristic'.")
    tracing: bool = Field(False, description="Whether LangSmith tracing is active.")
