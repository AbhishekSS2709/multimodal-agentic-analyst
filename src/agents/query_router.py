"""Query classification and routing for the RAG pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import *

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryCategory:
    """Metadata for a single query category."""

    name: str
    keywords: Tuple[str, ...]
    handler: str
    description: str


# The order matters: categories listed earlier win on ties during
# keyword-based classification.
CATEGORIES: List[QueryCategory] = [
    QueryCategory(
        name="analytics",
        keywords=(
            "trend", "pattern", "statistics", "monthly", "quarterly",
            "yearly", "annual", "growth", "decline", "distribution",
            "correlation", "metric", "metrics", "kpi",
        ),
        handler="sql_pipeline",
        description="Analytical queries requiring trend/pattern analysis.",
    ),
    QueryCategory(
        name="sql",
        keywords=(
            "how many", "count", "total", "average", "sum",
            "list all", "group by", "maximum", "minimum",
            "number of", "aggregate", "median",
        ),
        handler="sql_pipeline",
        description="Quantitative queries that map naturally to SQL.",
    ),
    QueryCategory(
        name="reasoning",
        keywords=(
            "why", "cause", "reason", "explain", "root cause",
            "because", "impact", "consequence", "effect",
            "lead to", "result in", "due to",
        ),
        handler="multi_hop_retrieval",
        description="Queries requiring causal or multi-step reasoning.",
    ),
    QueryCategory(
        name="comparison",
        keywords=(
            "compare", "comparison", "difference", "differences",
            "versus", "vs", "better", "worse", "pros and cons",
            "advantages", "disadvantages", "similarities",
        ),
        handler="multi_document_retrieval",
        description="Queries comparing two or more entities or concepts.",
    ),
    QueryCategory(
        name="summary",
        keywords=(
            "summarize", "summary", "overview", "brief",
            "key points", "highlights", "recap", "outline",
            "in short", "tldr",
        ),
        handler="standard_rag",
        description="Requests for summarisation or high-level overviews.",
    ),
    QueryCategory(
        name="factual",
        keywords=(
            "what", "when", "where", "who", "which",
            "define", "definition", "meaning", "is it true",
        ),
        handler="standard_rag",
        description="Straightforward factual look-up queries.",
    ),
]

# Build a fast lookup dict keyed by category name.
_CATEGORY_MAP: Dict[str, QueryCategory] = {c.name: c for c in CATEGORIES}


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Output of :meth:`QueryRouter.classify`."""

    category: str
    confidence: float
    suggested_handler: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "category": self.category,
            "confidence": self.confidence,
            "suggested_handler": self.suggested_handler,
        }


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

# Maps handler *names* (strings returned by ``route``) to actual callables.
# External code registers real implementations at startup via
# ``QueryRouter.register_handler``.
_HANDLER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def _default_handler(query: str, **kwargs: Any) -> Dict[str, Any]:
    """Fallback handler when no specialised handler is registered."""
    return {
        "answer": "No specialised handler available. Please refine your query.",
        "query": query,
        "handler": "default",
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class QueryRouter:
    """Classify natural-language queries and route them to handlers.

    The router uses a fast keyword-based classifier by default.  An
    optional LLM-based classifier can be enabled for ambiguous queries
    whose keyword confidence falls below *llm_threshold*.

    Parameters
    ----------
    llm_classify_fn : callable | None
        An async or sync function ``(query: str) -> str`` that returns a
        category name.  Used when keyword confidence is below
        *llm_threshold*.
    llm_threshold : float
        Minimum keyword-confidence required to skip LLM classification.
    """

    def __init__(
        self,
        llm_classify_fn: Optional[Callable[[str], str]] = None,
        llm_threshold: float = 0.3,
    ) -> None:
        self._llm_classify_fn = llm_classify_fn
        self._llm_threshold = llm_threshold
        self._handlers: Dict[str, Callable[..., Any]] = dict(_HANDLER_REGISTRY)

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(
        self, name: str, handler: Callable[..., Any]
    ) -> None:
        """Register a callable *handler* under *name*.

        Once registered, :meth:`route` will return *handler* whenever the
        classified category maps to *name*.
        """
        self._handlers[name] = handler
        logger.debug("Registered handler '%s'.", name)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, query: str) -> Dict[str, Any]:
        """Classify *query* into a category with confidence.

        Returns a dict with keys ``category``, ``confidence``, and
        ``suggested_handler``.
        """
        result = self._keyword_classify(query)

        # If confidence is too low and an LLM classifier is available,
        # try the LLM path.
        if (
            result.confidence < self._llm_threshold
            and self._llm_classify_fn is not None
        ):
            logger.info(
                "Keyword confidence %.2f below threshold %.2f; "
                "falling back to LLM classification.",
                result.confidence,
                self._llm_threshold,
            )
            result = self._llm_classify(query, keyword_result=result)

        logger.info(
            "Query classified as '%s' (confidence=%.2f, handler='%s').",
            result.category,
            result.confidence,
            result.suggested_handler,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, query: str) -> Callable[..., Any]:
        """Classify *query* and return the appropriate handler function.

        If no handler is registered for the suggested handler name the
        default fallback handler is returned.
        """
        classification = self.classify(query)
        handler_name: str = classification["suggested_handler"]
        handler = self._handlers.get(handler_name, _default_handler)
        logger.info(
            "Routing query to handler '%s'.",
            handler_name if handler is not _default_handler else "default",
        )
        return handler

    # ------------------------------------------------------------------
    # Internal: keyword-based classification
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_classify(query: str) -> ClassificationResult:
        """Score each category by counting keyword hits in *query*.

        The confidence is the fraction of a category's keywords that
        appear in the query, weighted by the inverse of total keywords
        (so a hit on a rarer keyword set carries more weight).
        """
        query_lower = query.lower()

        best_category: Optional[QueryCategory] = None
        best_score: float = 0.0

        for cat in CATEGORIES:
            hits = sum(1 for kw in cat.keywords if kw in query_lower)
            if hits == 0:
                continue
            # Normalise by the keyword-list length so categories with
            # fewer keywords are not penalised.
            score = hits / len(cat.keywords)
            if score > best_score:
                best_score = score
                best_category = cat

        if best_category is None:
            # Nothing matched — default to factual with low confidence.
            fallback = _CATEGORY_MAP["factual"]
            return ClassificationResult(
                category=fallback.name,
                confidence=0.0,
                suggested_handler=fallback.handler,
            )

        return ClassificationResult(
            category=best_category.name,
            confidence=round(best_score, 4),
            suggested_handler=best_category.handler,
        )

    # ------------------------------------------------------------------
    # Internal: LLM-based classification (optional)
    # ------------------------------------------------------------------

    def _llm_classify(
        self,
        query: str,
        keyword_result: ClassificationResult,
    ) -> ClassificationResult:
        """Use the configured LLM function to classify *query*.

        Falls back to *keyword_result* if the LLM returns an
        unrecognised category or raises an exception.
        """
        assert self._llm_classify_fn is not None

        try:
            predicted = self._llm_classify_fn(query)
            predicted = predicted.strip().lower()
        except Exception:
            logger.exception("LLM classification failed; using keyword result.")
            return keyword_result

        cat = _CATEGORY_MAP.get(predicted)
        if cat is None:
            logger.warning(
                "LLM returned unknown category '%s'; using keyword result.",
                predicted,
            )
            return keyword_result

        return ClassificationResult(
            category=cat.name,
            confidence=0.85,  # LLM classifications receive a fixed high confidence
            suggested_handler=cat.handler,
        )
