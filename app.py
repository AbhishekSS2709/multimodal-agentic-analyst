"""Streamlit frontend for the Enterprise RAG System.

Provides a professional multi-tab interface for question answering,
analytics, knowledge graph visualization, evaluation, and feedback.
Communicates with the FastAPI backend at http://localhost:8000.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"
API_TIMEOUT = 120  # seconds

st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Overall page */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* Header */
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 1rem;
    }

    /* Confidence badge */
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Query type badge */
    .query-badge {
        background-color: #e2e3f1;
        color: #383874;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-left: 8px;
    }

    /* Source card */
    .source-card {
        background-color: #f8f9fa;
        border-left: 3px solid #4a6cf7;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }

    /* Stats card */
    .stat-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
    }

    /* Metric card for evaluation */
    .metric-pass {
        background-color: #d4edda;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin: 4px;
    }
    .metric-fail {
        background-color: #f8d7da;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin: 4px;
    }

    /* Sidebar section headers */
    .sidebar-section {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "history": [],           # list of {question, answer, sources, confidence, query_type, query_id}
    "use_hybrid": True,
    "top_k": 5,
    "last_query_id": None,
    "stats": None,
    "analytics_result": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """Issue a GET request to the backend. Returns None on failure."""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot reach the API server. Make sure it is running on port 8000.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _api_post(endpoint: str, json_body: Optional[Dict] = None, files=None) -> Optional[Dict]:
    """Issue a POST request to the backend. Returns None on failure."""
    try:
        if files:
            resp = requests.post(f"{API_BASE}{endpoint}", files=files, timeout=API_TIMEOUT)
        else:
            resp = requests.post(f"{API_BASE}{endpoint}", json=json_body, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot reach the API server. Make sure it is running on port 8000.")
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = str(exc)
        st.error(f"API error: {detail}")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _load_stats() -> Dict:
    """Fetch system statistics from the backend."""
    data = _api_get("/api/stats")
    return data if data else {"total_docs": 0, "total_chunks": 0, "queries_answered": 0, "feedback_count": 0}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar():
    with st.sidebar:
        st.markdown('<div class="header-title">Enterprise RAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-subtitle">Retrieval-Augmented Generation System</div>', unsafe_allow_html=True)

        st.divider()

        # --- Upload section ---
        st.markdown('<div class="sidebar-section">Upload Documents</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "csv", "txt", "log", "md"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            if st.button("Process Upload", use_container_width=True, type="primary"):
                with st.spinner("Processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
                    result = _api_post("/api/upload", files=files)
                    if result:
                        status = result.get("status", "unknown")
                        if status == "success":
                            st.success(f"Uploaded! {result.get('chunks_created', 0)} chunks created.")
                        elif status == "partial":
                            st.warning("Document ingested but chunking/embedding had issues.")
                        elif status == "warning":
                            st.warning("No content could be extracted from the file.")
                        else:
                            st.info(f"Status: {status}")

        st.divider()

        # --- System stats ---
        st.markdown('<div class="sidebar-section">System Statistics</div>', unsafe_allow_html=True)
        stats = _load_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats["total_docs"]}</div>'
                f'<div class="stat-label">Documents</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats["total_chunks"]}</div>'
                f'<div class="stat-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats["queries_answered"]}</div>'
                f'<div class="stat-label">Queries</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats["feedback_count"]}</div>'
                f'<div class="stat-label">Feedback</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # --- Settings ---
        st.markdown('<div class="sidebar-section">Settings</div>', unsafe_allow_html=True)
        st.session_state["use_hybrid"] = st.toggle("Hybrid Search", value=st.session_state["use_hybrid"])
        st.session_state["top_k"] = st.slider("Top K results", min_value=1, max_value=20, value=st.session_state["top_k"])

        st.divider()

        # --- Health check ---
        if st.button("Check System Health", use_container_width=True):
            health = _api_get("/api/health")
            if health:
                status = health.get("status", "unknown")
                if status == "ok":
                    st.success("All systems operational")
                else:
                    st.warning(f"System status: {status}")
                for comp, comp_status in health.get("components", {}).items():
                    icon = "+" if comp_status == "ok" else "-"
                    label = comp.replace("_", " ").title()
                    st.text(f"  [{icon}] {label}: {comp_status}")


# ---------------------------------------------------------------------------
# Tab 1: Ask Questions
# ---------------------------------------------------------------------------

def _render_ask_tab():
    st.markdown("### Ask a Question")
    st.markdown("Enter your question below. The system will retrieve relevant documents and generate an answer.")

    question = st.text_input(
        "Your question",
        placeholder="e.g., What is the total revenue from the North region?",
        label_visibility="collapsed",
    )

    if st.button("Get Answer", type="primary", disabled=not question):
        with st.spinner("Searching and generating answer..."):
            payload = {
                "question": question,
                "use_hybrid": st.session_state["use_hybrid"],
                "include_sources": True,
                "top_k": st.session_state["top_k"],
            }
            result = _api_post("/api/ask", json_body=payload)

        if result:
            import uuid as _uuid
            query_id = _uuid.uuid4().hex[:12]
            entry = {
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "query_type": result.get("query_type", "unknown"),
                "chart_html": result.get("chart_html"),
                "query_id": query_id,
            }
            st.session_state["history"].insert(0, entry)
            st.session_state["last_query_id"] = query_id

    # Display history
    for entry in st.session_state["history"]:
        st.divider()
        _render_answer_card(entry)


def _render_answer_card(entry: Dict):
    """Render a single Q&A result card."""
    confidence = entry.get("confidence", 0.0)
    query_type = entry.get("query_type", "unknown")

    # Confidence badge
    if confidence >= 0.8:
        badge_class = "confidence-high"
    elif confidence >= 0.5:
        badge_class = "confidence-medium"
    else:
        badge_class = "confidence-low"

    st.markdown(f"**Q:** {entry['question']}")

    badge_html = (
        f'<span class="{badge_class}">Confidence: {confidence:.0%}</span>'
        f'<span class="query-badge">{query_type}</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown(entry.get("answer", ""))

    # Chart if available
    chart_html = entry.get("chart_html")
    if chart_html:
        st.components.v1.html(chart_html, height=420, scrolling=True)

    # Source documents
    sources = entry.get("sources", [])
    if sources:
        with st.expander(f"View {len(sources)} source document(s)"):
            for i, src in enumerate(sources):
                source_name = src.get("source", "unknown")
                score = src.get("score", 0.0)
                text = src.get("text", "")
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>Source {i+1}:</strong> {source_name} '
                    f'(score: {score:.4f})<br/>'
                    f'<small>{text[:300]}{"..." if len(text) > 300 else ""}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Tab 2: Analytics
# ---------------------------------------------------------------------------

def _render_analytics_tab():
    st.markdown("### Analytics Dashboard")
    st.markdown("Ask business questions and get SQL-powered analytics with auto-generated charts.")

    analytics_q = st.text_input(
        "Analytics question",
        placeholder="e.g., What is the monthly revenue trend?",
        label_visibility="collapsed",
        key="analytics_question",
    )

    if st.button("Run Analytics", type="primary", disabled=not analytics_q, key="run_analytics"):
        with st.spinner("Running analytics query..."):
            result = _api_get("/api/analytics", params={"question": analytics_q})

        if result:
            st.session_state["analytics_result"] = result

    result = st.session_state.get("analytics_result")
    if result:
        # SQL query
        sql = result.get("sql", "")
        if sql:
            st.markdown("**Generated SQL:**")
            st.code(sql, language="sql")

        # Results table
        rows = result.get("results", [])
        if rows:
            st.markdown(f"**Results** ({len(rows)} row{'s' if len(rows) != 1 else ''}):")
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Chart
        chart_html = result.get("chart_html", "")
        if chart_html:
            st.markdown("**Visualization:**")
            st.components.v1.html(chart_html, height=450, scrolling=True)

        # Insight
        insight = result.get("insight", "")
        if insight:
            st.markdown("**Insight:**")
            st.info(insight)


# ---------------------------------------------------------------------------
# Tab 3: Knowledge Graph
# ---------------------------------------------------------------------------

def _render_knowledge_graph_tab():
    st.markdown("### Knowledge Graph Explorer")
    st.markdown("Visualize relationships between entities in your documents.")

    # Entity search
    entity_query = st.text_input(
        "Search for an entity",
        placeholder="e.g., GlobalTech Supply",
        key="kg_entity_search",
    )

    # Try to load the knowledge graph visualization
    import os
    from pathlib import Path

    project_root = Path(__file__).resolve().parent
    kg_html_path = project_root / "static" / "knowledge_graph.html"

    if kg_html_path.exists():
        try:
            html_content = kg_html_path.read_text(encoding="utf-8")
            st.components.v1.html(html_content, height=600, scrolling=True)
        except Exception as exc:
            st.warning(f"Could not load knowledge graph visualization: {exc}")
    else:
        # Generate a basic graph from the data if possible
        _render_basic_knowledge_graph(entity_query)


def _render_basic_knowledge_graph(entity_filter: str = ""):
    """Generate and display a basic knowledge graph from available data."""
    try:
        import networkx as nx
        from pyvis.network import Network
        import tempfile
        import sqlite3
        from pathlib import Path
        import sys

        project_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(project_root))
        from config.settings import SQLITE_DB_PATH

        if not SQLITE_DB_PATH.exists():
            st.info(
                "No data available for knowledge graph. "
                "Upload documents or set up the SQL database first."
            )
            return

        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Build graph from orders data
        query = "SELECT customer_name, supplier, product, region, status FROM orders"
        if entity_filter:
            query += f" WHERE customer_name LIKE '%{entity_filter}%' OR supplier LIKE '%{entity_filter}%' OR product LIKE '%{entity_filter}%'"
        query += " LIMIT 200"

        rows = conn.execute(query).fetchall()
        conn.close()

        if not rows:
            st.info("No matching entities found.")
            return

        G = nx.Graph()

        for row in rows:
            customer = row["customer_name"]
            supplier = row["supplier"]
            product = row["product"]
            region = row["region"]

            G.add_node(customer, group="customer", title=f"Customer: {customer}")
            G.add_node(supplier, group="supplier", title=f"Supplier: {supplier}")
            G.add_node(product, group="product", title=f"Product: {product}")
            G.add_node(region, group="region", title=f"Region: {region}")

            G.add_edge(customer, product, title="ordered")
            G.add_edge(supplier, product, title="supplies")
            G.add_edge(customer, region, title="located_in")

        # Color map for groups
        color_map = {
            "customer": "#4a6cf7",
            "supplier": "#e74c3c",
            "product": "#2ecc71",
            "region": "#f39c12",
        }

        net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)

        for node, data in G.nodes(data=True):
            group = data.get("group", "other")
            net.add_node(
                node,
                label=node[:25],
                color=color_map.get(group, "#999999"),
                title=data.get("title", node),
                size=20 if group in ("customer", "supplier") else 15,
            )

        for u, v, data in G.edges(data=True):
            net.add_edge(u, v, title=data.get("title", ""))

        # Render to temporary HTML and display
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            net.save_graph(f.name)
            html_content = Path(f.name).read_text(encoding="utf-8")
            st.components.v1.html(html_content, height=600, scrolling=True)

        # Legend
        st.markdown(
            """
            **Legend:**
            <span style="color:#4a6cf7">&#9632;</span> Customer &nbsp;
            <span style="color:#e74c3c">&#9632;</span> Supplier &nbsp;
            <span style="color:#2ecc71">&#9632;</span> Product &nbsp;
            <span style="color:#f39c12">&#9632;</span> Region
            """,
            unsafe_allow_html=True,
        )

    except ImportError as exc:
        st.warning(
            f"Knowledge graph visualization requires networkx and pyvis. "
            f"Install with: pip install networkx pyvis\n\nDetails: {exc}"
        )
    except Exception as exc:
        st.error(f"Failed to generate knowledge graph: {exc}")


# ---------------------------------------------------------------------------
# Tab 4: Evaluation
# ---------------------------------------------------------------------------

def _render_evaluation_tab():
    st.markdown("### System Evaluation")
    st.markdown("Run evaluation checks to assess the RAG system's readiness and quality.")

    if st.button("Run Evaluation", type="primary", key="run_eval"):
        with st.spinner("Running evaluation..."):
            result = _api_get("/api/evaluation")

        if result:
            st.session_state["eval_result"] = result

    result = st.session_state.get("eval_result")
    if not result:
        st.info("Click 'Run Evaluation' to assess the system.")
        return

    # Summary
    summary = result.get("summary", "")
    if summary:
        st.markdown(f"**Summary:** {summary}")

    # Metrics in a dashboard layout
    metrics = result.get("metrics", [])
    if metrics:
        st.markdown("**Metrics:**")
        cols = st.columns(min(len(metrics), 4))
        for i, metric in enumerate(metrics):
            col = cols[i % len(cols)]
            name = metric.get("name", "").replace("_", " ").title()
            value = metric.get("value", 0.0)
            css_class = "metric-pass" if value >= 0.5 else "metric-fail"
            with col:
                st.markdown(
                    f'<div class="{css_class}">'
                    f'<div style="font-size:1.3rem;font-weight:700;">{value:.0%}</div>'
                    f'<div style="font-size:0.75rem;color:#555;">{name}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Per-category breakdown
    per_category = result.get("per_category", {})
    if per_category:
        st.markdown("**Per-Category Breakdown:**")
        for cat_name, cat_data in per_category.items():
            display_name = cat_name.replace("_", " ").title()
            score = cat_data.get("score", 0.0) if isinstance(cat_data, dict) else 0.0
            checks = cat_data.get("checks", []) if isinstance(cat_data, dict) else []

            with st.expander(f"{display_name} (Score: {score:.0%})"):
                if checks:
                    for check in checks:
                        check_label = check.replace("_", " ").title()
                        # Find the metric value
                        metric_val = next(
                            (m.get("value", 0) for m in metrics if m.get("name") == check),
                            None,
                        )
                        if metric_val is not None:
                            icon = "[PASS]" if metric_val >= 0.5 else "[FAIL]"
                            st.text(f"  {icon} {check_label}: {metric_val:.0%}")
                        else:
                            st.text(f"  [ -- ] {check_label}")
                else:
                    st.text(f"  Score: {score:.0%}")


# ---------------------------------------------------------------------------
# Tab 5: Feedback
# ---------------------------------------------------------------------------

def _render_feedback_tab():
    st.markdown("### Feedback")
    st.markdown("Rate previous answers to help improve the system.")

    # Rating form
    st.markdown("**Rate a Previous Answer:**")

    history = st.session_state.get("history", [])
    if not history:
        st.info("No queries to rate yet. Ask some questions first!")
        return

    # Build options for the selectbox
    options = [
        f"Q: {entry['question'][:80]}{'...' if len(entry['question']) > 80 else ''}"
        for entry in history
    ]
    selected_idx = st.selectbox("Select a query to rate", range(len(options)), format_func=lambda i: options[i])

    if selected_idx is not None:
        selected_entry = history[selected_idx]

        st.markdown(f"**Question:** {selected_entry['question']}")
        st.markdown(f"**Answer:** {selected_entry['answer'][:200]}{'...' if len(selected_entry['answer']) > 200 else ''}")

        rating = st.slider("Rating", min_value=1, max_value=5, value=3, key="feedback_rating")
        comment = st.text_area("Comment (optional)", key="feedback_comment", placeholder="Any additional thoughts...")

        if st.button("Submit Feedback", type="primary", key="submit_feedback"):
            query_id = selected_entry.get("query_id", "unknown")
            payload = {
                "query_id": query_id,
                "rating": rating,
                "comment": comment,
            }
            result = _api_post("/api/feedback", json_body=payload)
            if result and result.get("status") == "success":
                st.success("Feedback submitted. Thank you!")
            elif result:
                st.warning(f"Feedback status: {result.get('status', 'unknown')}")

    # Feedback statistics
    st.divider()
    st.markdown("**Feedback Statistics:**")
    stats = _load_stats()
    feedback_count = stats.get("feedback_count", 0)
    st.metric("Total Feedback Entries", feedback_count)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    _render_sidebar()

    # Header
    st.markdown('<div class="header-title">Enterprise RAG System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-subtitle">'
        "Ask questions, run analytics, explore knowledge graphs, and evaluate system quality."
        "</div>",
        unsafe_allow_html=True,
    )

    # Tabs
    tab_ask, tab_analytics, tab_kg, tab_eval, tab_feedback = st.tabs(
        ["Ask Questions", "Analytics", "Knowledge Graph", "Evaluation", "Feedback"]
    )

    with tab_ask:
        _render_ask_tab()

    with tab_analytics:
        _render_analytics_tab()

    with tab_kg:
        _render_knowledge_graph_tab()

    with tab_eval:
        _render_evaluation_tab()

    with tab_feedback:
        _render_feedback_tab()


if __name__ == "__main__":
    main()
