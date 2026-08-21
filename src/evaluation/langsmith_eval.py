"""LangSmith datasets, evaluators, and experiments for the analyst graph.

The evaluators here are **pure functions** — ``(run_outputs, example_outputs)
-> {"key", "score", "comment"}``.  Keeping them free of the LangSmith SDK means
they run in CI with no key and no network, and the same functions are what
LangSmith calls when a real experiment runs.

Every LangSmith-touching entry point degrades to a no-op without a key, so
importing this module never requires an account.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.test_cases import TEST_CASES
from src.graph.observability import get_client
from src.graph.textutil import overlap_ratio, tokenize

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "multimodal-agentic-analyst"

# Which specialists *should* handle each existing test-case category.  This is
# what makes routing measurable: the supervisor's choice is now a scored
# prediction, not an untested assumption.
_CATEGORY_SPECIALISTS: Dict[str, List[str]] = {
    "factual": ["document"],
    "summary": ["document"],
    "sql": ["analytics"],
    "reasoning": ["graph"],
    "comparison": ["graph"],
    "visual_reasoning": ["visual"],
    "ocr_extraction": ["visual"],
    "multimodal_document": ["document", "visual"],
    "edge": ["document"],
}

MAX_RETRIES_SCORED = 2


def expected_specialists_for(category: str) -> List[str]:
    """Specialists a correct router should pick for ``category``."""
    return _CATEGORY_SPECIALISTS.get(category, ["document"])


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def _findings_text(run: Dict[str, Any]) -> str:
    return "\n".join(str(f.get("content", "")) for f in run.get("findings", []) or [])


def faithfulness(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """How much of the answer is supported by the retrieved findings."""
    corpus = _findings_text(run)
    answer = str(run.get("answer", ""))
    if not corpus or not answer:
        return {"key": "faithfulness", "score": 0.0,
                "comment": "no findings to ground the answer in"}
    score = round(overlap_ratio(answer, corpus), 3)
    return {"key": "faithfulness", "score": score,
            "comment": f"{score:.0%} of answer terms supported by findings"}


def citation_accuracy(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """Fraction of cited sources that actually appear among the findings."""
    citations = run.get("citations", []) or []
    if not citations:
        return {"key": "citation_accuracy", "score": None,
                "comment": "no citations emitted"}
    finding_sources = {str(f.get("source", "")) for f in run.get("findings", []) or []}
    hits = sum(1 for c in citations if str(c.get("source", "")) in finding_sources)
    score = round(hits / len(citations), 3)
    return {"key": "citation_accuracy", "score": score,
            "comment": f"{hits}/{len(citations)} citations trace to a finding"}


def routing_accuracy(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """Did the supervisor pick the specialists this category needs?"""
    expected = set(example.get("specialists", []) or [])
    if not expected:
        return {"key": "routing_accuracy", "score": None,
                "comment": "no routing expectation for this example"}
    actual = set(run.get("specialists", []) or [])
    hits = len(expected & actual)
    score = round(hits / len(expected), 3)
    return {"key": "routing_accuracy", "score": score,
            "comment": f"expected {sorted(expected)}, got {sorted(actual)}"}


def modality_match(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """Did the evidence come from the modality the question calls for?"""
    expected = example.get("modality")
    if not expected:
        return {"key": "modality_match", "score": None,
                "comment": "no modality expectation"}
    modalities = {str(f.get("modality", "text")) for f in run.get("findings", []) or []}
    if expected == "text":
        matched = "text" in modalities
    else:
        matched = bool(modalities - {"text"})
    return {"key": "modality_match", "score": 1.0 if matched else 0.0,
            "comment": f"expected {expected}, findings had {sorted(modalities)}"}


def answer_correctness(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """Fraction of the expected keywords present in the answer."""
    expected = example.get("expected_answer_contains") or []
    if not expected:
        return {"key": "answer_correctness", "score": None,
                "comment": "no keyword expectation"}
    answer_tokens = set(tokenize(str(run.get("answer", ""))))
    hits = 0
    for keyword in expected:
        if set(tokenize(str(keyword))) <= answer_tokens:
            hits += 1
    score = round(hits / len(expected), 3)
    return {"key": "answer_correctness", "score": score,
            "comment": f"{hits}/{len(expected)} expected keywords present"}


def retry_efficiency(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """Penalise runs that needed corrective retries; 1.0 means first-try success."""
    retries = int(run.get("retry_count", 0) or 0)
    score = round(max(0.0, 1.0 - retries / (MAX_RETRIES_SCORED + 1)), 3)
    return {"key": "retry_efficiency", "score": score,
            "comment": f"{retries} synthesis attempt(s)"}


EVALUATORS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "faithfulness": faithfulness,
    "citation_accuracy": citation_accuracy,
    "routing_accuracy": routing_accuracy,
    "modality_match": modality_match,
    "answer_correctness": answer_correctness,
    "retry_efficiency": retry_efficiency,
}


def score_run(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """Run every evaluator and return ``{name: score}``."""
    scores: Dict[str, Any] = {}
    for name, fn in EVALUATORS.items():
        try:
            scores[name] = fn(run, example)["score"]
        except Exception as exc:
            logger.warning("Evaluator %s failed: %s", name, exc)
            scores[name] = None
    return scores


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def build_examples() -> List[Dict[str, Any]]:
    """Turn the existing test-case catalogue into LangSmith examples."""
    examples: List[Dict[str, Any]] = []
    for case in TEST_CASES:
        category = case.get("category", "factual")
        outputs: Dict[str, Any] = {
            "category": category,
            "specialists": expected_specialists_for(category),
        }
        if case.get("expected_answer_contains"):
            outputs["expected_answer_contains"] = case["expected_answer_contains"]
        if case.get("expected_answer"):
            outputs["expected_answer"] = case["expected_answer"]
        if case.get("expected_modality"):
            outputs["modality"] = case["expected_modality"]
        if case.get("expected_sources"):
            outputs["sources"] = case["expected_sources"]

        examples.append({"inputs": {"question": case["query"]}, "outputs": outputs})
    return examples


def push_dataset(
    name: str = DEFAULT_DATASET,
    client: Optional[Any] = None,
) -> Optional[str]:
    """Upload the example set to LangSmith. Returns the dataset id, or ``None``.

    A ``None`` client means no key is configured — that is a no-op, not an error.
    """
    client = client or get_client()
    if client is None:
        logger.info("No LangSmith client; skipping dataset upload.")
        return None

    examples = build_examples()
    try:
        if client.has_dataset(dataset_name=name):
            dataset = client.read_dataset(dataset_name=name)
        else:
            dataset = client.create_dataset(
                dataset_name=name,
                description="Multimodal agentic analyst regression suite.",
            )
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
        )
        logger.info("Pushed %d examples to dataset %s", len(examples), name)
        return str(dataset.id)
    except Exception as exc:
        logger.warning("Dataset upload failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiment(
    name: str = DEFAULT_DATASET,
    config: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None,
    components: Any = None,
) -> Dict[str, Any]:
    """Evaluate the graph over every example and aggregate the scores.

    Runs locally regardless of LangSmith availability; results are additionally
    uploaded when a client is configured.  ``config`` is passed through to the
    graph so alternative setups (heuristic vs LLM routing, retries on or off)
    can be compared as separate experiments.
    """
    from src.graph.build import run_query

    config = config or {}
    examples = build_examples()
    started = time.time()

    per_example: List[Dict[str, Any]] = []
    for example in examples:
        question = example["inputs"]["question"]
        t0 = time.time()
        try:
            result = run_query(
                question,
                require_approval=False,
                components=components,
            )
        except Exception as exc:
            logger.warning("Run failed for %r: %s", question[:60], exc)
            result = {"answer": "", "findings": [], "citations": [],
                      "specialists": [], "retry_count": 0}
        latency = time.time() - t0

        scores = score_run(result, example["outputs"])
        scores["latency_seconds"] = round(latency, 3)
        per_example.append({"question": question,
                            "category": example["outputs"].get("category"),
                            "scores": scores})

    aggregate: Dict[str, float] = {}
    for metric in list(EVALUATORS) + ["latency_seconds"]:
        values = [e["scores"].get(metric) for e in per_example]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            aggregate[metric] = round(sum(values) / len(values), 3)

    summary = {
        "experiment": name,
        "config": config,
        "examples": len(per_example),
        "aggregate": aggregate,
        "per_example": per_example,
        "duration_seconds": round(time.time() - started, 2),
    }

    client = client or get_client()
    if client is not None:
        logger.info("LangSmith client active; traces uploaded automatically.")

    return summary


def mirror_feedback(
    query_id: str,
    rating: int,
    comment: str = "",
    client: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> bool:
    """Mirror a FeedbackStore entry into LangSmith. ``False`` when unconfigured.

    This is what closes the loop: production thumbs up/down land next to the
    traces that produced them, so bad runs can be promoted into the dataset.
    """
    client = client or get_client()
    if client is None or run_id is None:
        return False
    try:
        client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=rating / 5.0,
            comment=comment or f"query_id={query_id}",
        )
        return True
    except Exception as exc:
        logger.warning("Feedback mirroring failed: %s", exc)
        return False
