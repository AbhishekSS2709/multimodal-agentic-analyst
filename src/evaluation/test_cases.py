"""Pre-built test cases for the RAG evaluation pipeline.

Each test case specifies a query, expected keywords in the answer, and a
category.  Categories:

- **factual**: simple fact-lookup queries
- **reasoning**: multi-hop or causal-reasoning queries
- **sql**: queries that should trigger SQL generation
- **summary**: broad summarisation requests
- **comparison**: queries that compare entities or time periods
- **edge**: ambiguous, vague, or adversarial inputs
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Test-case catalogue
# ---------------------------------------------------------------------------

TEST_CASES: List[Dict[str, Any]] = [
    # ── factual ──────────────────────────────────────────────────────────
    {
        "query": "Which supplier has the most problems?",
        "expected_answer_contains": ["Apex Materials"],
        "category": "factual",
    },
    {
        "query": "What is the total number of orders in the database?",
        "expected_answer_contains": ["orders", "total"],
        "category": "factual",
    },
    {
        "query": "Who is the primary contact for Apex Materials?",
        "expected_answer_contains": ["Apex", "contact"],
        "category": "factual",
    },

    # ── reasoning ────────────────────────────────────────────────────────
    {
        "query": "Why are there dispatch delays?",
        "expected_answer_contains": ["Apex Materials", "supplier", "delay"],
        "category": "reasoning",
    },
    {
        "query": "What factors contribute to late deliveries in Q4?",
        "expected_answer_contains": ["delivery", "late"],
        "category": "reasoning",
    },
    {
        "query": "How does supplier reliability affect order fulfilment rates?",
        "expected_answer_contains": ["supplier", "reliability"],
        "category": "reasoning",
    },

    # ── sql ──────────────────────────────────────────────────────────────
    {
        "query": "Show monthly order trend",
        "expected_answer_contains": ["trend", "orders"],
        "category": "sql",
    },
    {
        "query": "What is the average order value by region?",
        "expected_answer_contains": ["average", "region"],
        "category": "sql",
    },
    {
        "query": "List the top 5 suppliers by order volume",
        "expected_answer_contains": ["supplier", "order"],
        "category": "sql",
    },

    # ── summary ──────────────────────────────────────────────────────────
    {
        "query": "Summarise the overall supply chain performance",
        "expected_answer_contains": ["supply", "chain", "performance"],
        "category": "summary",
    },
    {
        "query": "Give me an executive overview of logistics operations",
        "expected_answer_contains": ["logistics", "operations"],
        "category": "summary",
    },
    {
        "query": "What are the key takeaways from the last quarter?",
        "expected_answer_contains": ["quarter"],
        "category": "summary",
    },

    # ── comparison ───────────────────────────────────────────────────────
    {
        "query": "Compare supplier performance between Q3 and Q4",
        "expected_answer_contains": ["Q3", "Q4"],
        "category": "comparison",
    },
    {
        "query": "Which region has higher fulfilment rates, East or West?",
        "expected_answer_contains": ["region"],
        "category": "comparison",
    },
    {
        "query": "How do on-time delivery rates differ across suppliers?",
        "expected_answer_contains": ["delivery", "supplier"],
        "category": "comparison",
    },

    # ── edge cases ───────────────────────────────────────────────────────
    {
        "query": "Tell me everything",
        "expected_answer_contains": [],
        "category": "edge",
        "notes": "Intentionally vague — tests graceful handling of ambiguity.",
    },
    {
        "query": "What about the thing with the stuff?",
        "expected_answer_contains": [],
        "category": "edge",
        "notes": "Completely ambiguous query.",
    },
    {
        "query": (
            "What was the dispatch delay for Apex Materials in October, "
            "and also how does that compare with November, "
            "and what recommendations would you make?"
        ),
        "expected_answer_contains": ["Apex", "delay"],
        "category": "edge",
        "notes": "Multi-part question that tests handling of compound queries.",
    },
    {
        "query": "",
        "expected_answer_contains": [],
        "category": "edge",
        "notes": "Empty query — should be handled gracefully.",
    },
    {
        "query": "SELECT * FROM orders; DROP TABLE orders;--",
        "expected_answer_contains": [],
        "category": "edge",
        "notes": "SQL injection attempt — must not be executed literally.",
    },

    # Multimodal test cases
    {
        "query": "What does the chart in the quarterly report show?",
        "expected_answer": "The chart shows revenue trends across quarters.",
        "expected_modality": "visual",
        "expected_sources": [],
        "category": "visual_reasoning",
    },
    {
        "query": "Describe the architecture diagram",
        "expected_answer": "The architecture diagram shows the system components.",
        "expected_modality": "visual",
        "expected_sources": [],
        "category": "visual_reasoning",
    },
    {
        "query": "What text appears in the scanned document image?",
        "expected_answer": "The scanned document contains...",
        "expected_modality": "visual",
        "expected_sources": [],
        "category": "ocr_extraction",
    },
    {
        "query": "Summarize the content from the PowerPoint presentation",
        "expected_answer": "The presentation covers...",
        "expected_modality": "text",
        "expected_sources": [],
        "category": "multimodal_document",
    },
    {
        "query": "What data is in the Excel spreadsheet?",
        "expected_answer": "The spreadsheet contains...",
        "expected_modality": "text",
        "expected_sources": [],
        "category": "multimodal_document",
    },
]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """Return only test cases matching *category*."""
    return [tc for tc in TEST_CASES if tc.get("category") == category]


def get_all_categories() -> List[str]:
    """Return sorted list of unique categories present in TEST_CASES."""
    return sorted({tc["category"] for tc in TEST_CASES if "category" in tc})
