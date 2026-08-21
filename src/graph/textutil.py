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


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stop-words."""
    if _eval_tokenize is not None:
        return _eval_tokenize(text or "")
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in _FALLBACK_STOP_WORDS]


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
