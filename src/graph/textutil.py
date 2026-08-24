"""Token-overlap scoring shared by the heuristic paths.

Three nodes need the same "how much of A appears in B" measure: the document
grader, the synthesizer's ranking, and the verifier's groundedness check.
It reuses the stop-word tokenizer already in ``src/evaluation/eval_pipeline.py``
so heuristic scoring here and evaluation scoring there stay consistent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from src.evaluation.eval_pipeline import _tokenize as _eval_tokenize
except Exception:  # pragma: no cover - only if eval_pipeline is unavailable
    _eval_tokenize = None

_FALLBACK_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "and", "or",
    "what", "which", "who", "how", "why", "when", "where", "this", "that",
    "it", "its", "do", "does", "did", "can", "could", "will", "would",
}


def _stem(token: str) -> str:
    """Crude suffix stripping so ``refund`` matches ``refunds``.

    Deliberately not a real stemmer — it only has to be *consistent* on both
    sides of a comparison. Without it, simple plural mismatches sink the
    overlap score and the grader rejects genuinely relevant documents.
    """
    if len(token) > 5:
        for suffix in ("ing", "ed"):
            if token.endswith(suffix):
                return token[: -len(suffix)]
    # -ies -> -y, before the plain plural rule. Without this "deliveries"
    # stemmed to "deliverie" while "delivery" stayed "delivery", so a question
    # about late deliveries could not match a document about a late delivery --
    # the grader scored 0.20 against a 0.30 threshold and the graph abstained.
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stop-words, normalise suffixes."""
    if _eval_tokenize is not None:
        raw = _eval_tokenize(text or "")
    else:
        raw = [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
               if t not in _FALLBACK_STOP_WORDS]
    return [_stem(t) for t in raw]


def overlap_ratio(needle: str, haystack: str) -> float:
    """Fraction of ``needle``'s content words that appear in ``haystack``.

    Asymmetric on purpose: for groundedness we ask "how much of the *answer*
    is supported by the findings", not the reverse.
    """
    needle_tokens = set(tokenize(needle))
    if not needle_tokens:
        return 0.0
    haystack_tokens = set(tokenize(haystack))
    if not haystack_tokens:
        return 0.0
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)
