"""BM25 keyword search built on the rank_bm25 library."""

from __future__ import annotations

import logging
import pickle
import re
import string
from pathlib import Path
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Pre-compile a regex that strips punctuation — faster than str.translate in
# tight loops and avoids rebuilding the table on every call.
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _tokenize(text: str) -> List[str]:
    """Tokenize *text* into lowercase words with punctuation removed.

    This deliberately uses a lightweight approach (split on whitespace,
    strip punctuation, lowercase) so that indexing stays fast and
    dependency-free.
    """
    cleaned = _PUNCT_RE.sub("", text.lower())
    return [tok for tok in cleaned.split() if tok]


class BM25Search:
    """BM25Okapi-based keyword search over a corpus of text chunks.

    Parameters
    ----------
    corpus : list[str] | None
        Initial corpus of chunk texts.  If *None*, an empty index is
        created and documents can be added later via :meth:`add_documents`.
    """

    def __init__(self, corpus: Optional[List[str]] = None) -> None:
        self._corpus: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._index: Optional[BM25Okapi] = None

        if corpus:
            self.add_documents(corpus)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Return the number of documents currently in the index."""
        return len(self._corpus)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, texts: List[str]) -> None:
        """Tokenize *texts* and append them to the index.

        The BM25 index is rebuilt from scratch each time because
        ``rank_bm25.BM25Okapi`` does not support incremental updates.
        For corpora of the size typical in enterprise RAG pipelines
        (tens of thousands of chunks) this is fast enough.
        """
        if not texts:
            logger.warning("add_documents called with an empty list; skipping.")
            return

        new_tokenized = [_tokenize(t) for t in texts]
        self._corpus.extend(texts)
        self._tokenized_corpus.extend(new_tokenized)
        self._rebuild_index()
        logger.info(
            "Added %d document(s); corpus now contains %d document(s).",
            len(texts),
            self.corpus_size,
        )

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from the current tokenized corpus."""
        if not self._tokenized_corpus:
            self._index = None
            return
        self._index = BM25Okapi(self._tokenized_corpus)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return the *top_k* most relevant (chunk_index, score) pairs.

        Parameters
        ----------
        query : str
            The user query in natural language.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[tuple[int, float]]
            Each element is ``(chunk_index, bm25_score)`` sorted by
            descending score.
        """
        if self._index is None:
            logger.warning("search() called on an empty index; returning [].")
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            logger.warning("Query tokenized to empty list; returning [].")
            return []

        scores = self._index.get_scores(tokenized_query)

        # Build (index, score) pairs, filter zero-score docs, sort desc.
        scored = [
            (idx, float(score))
            for idx, score in enumerate(scores)
            if score > 0.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the index and corpus to *path* using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "corpus": self._corpus,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved to %s.", path)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Search":
        """Load a previously saved index from *path*.

        Returns
        -------
        BM25Search
            A fully initialised instance with the restored corpus.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as fh:
            state = pickle.load(fh)  # noqa: S301

        instance = cls()
        instance._corpus = state["corpus"]
        instance._tokenized_corpus = state["tokenized_corpus"]
        instance._rebuild_index()
        logger.info(
            "BM25 index loaded from %s (%d documents).",
            path,
            instance.corpus_size,
        )
        return instance
