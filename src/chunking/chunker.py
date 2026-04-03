"""Adaptive semantic chunking engine.

Splits documents into semantically coherent chunks using sentence boundaries,
with adaptive sizing that produces shorter chunks for dense technical content
and longer chunks for narrative prose.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE

if TYPE_CHECKING:
    from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk produced by the chunking engine."""

    chunk_id: str
    text: str
    metadata: dict
    doc_id: str
    token_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tokenizer():
    """Return a tiktoken encoding (cl100k_base, used by most modern models)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.warning(
            "tiktoken not installed — falling back to whitespace tokenisation."
        )
        return None


def _count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in *text*."""
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Content-density heuristics
# ---------------------------------------------------------------------------

# Patterns that indicate dense / technical content
_CODE_PATTERN = re.compile(r"[{}\[\]();=<>]")
_URL_PATTERN = re.compile(r"https?://")
_NUMERIC_HEAVY = re.compile(r"\d+[.,:;]\d+")


def _technical_density(text: str) -> float:
    """Return a 0-1 score estimating how 'technical' a passage is.

    Higher values mean denser / more technical content that benefits from
    shorter chunks so that retrieval precision stays high.
    """
    if not text:
        return 0.0

    words = text.split()
    n_words = max(len(words), 1)

    code_chars = len(_CODE_PATTERN.findall(text))
    url_count = len(_URL_PATTERN.findall(text))
    numeric_count = len(_NUMERIC_HEAVY.findall(text))

    # Average word length — technical prose tends to have longer words
    avg_word_len = sum(len(w) for w in words) / n_words

    # Normalised features (capped at 1.0)
    code_score = min(code_chars / n_words, 1.0)
    url_score = min(url_count / max(n_words / 20, 1), 1.0)
    numeric_score = min(numeric_count / max(n_words / 10, 1), 1.0)
    length_score = min(max(avg_word_len - 5, 0) / 5, 1.0)

    density = 0.35 * code_score + 0.20 * url_score + 0.25 * numeric_score + 0.20 * length_score
    return min(density, 1.0)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"  # standard sentence end
    r"|(?<=\n)\s*(?=\n)"       # blank-line paragraph boundary
)


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences using regex heuristics."""
    parts = _SENTENCE_BOUNDARY.split(text)
    # Collapse any empty strings and strip whitespace
    return [s.strip() for s in parts if s and s.strip()]


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """Adaptive semantic chunker.

    Parameters
    ----------
    max_tokens : int
        Maximum chunk size in tokens (default from settings).
    overlap_tokens : int
        Number of overlapping tokens between consecutive chunks.
    min_chunk_tokens : int
        Minimum chunk size; shorter remainders are merged into the previous
        chunk rather than emitted separately.
    """

    def __init__(
        self,
        max_tokens: int = CHUNK_SIZE,
        overlap_tokens: int = CHUNK_OVERLAP,
        min_chunk_tokens: int = MIN_CHUNK_SIZE,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self._tokenizer = _get_tokenizer()
        logger.info(
            "SemanticChunker initialised (max=%d, overlap=%d, min=%d)",
            max_tokens,
            overlap_tokens,
            min_chunk_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, document: "Document") -> List[Chunk]:
        """Split a *Document* into a list of :class:`Chunk` objects.

        Parameters
        ----------
        document:
            A Document dataclass instance produced by the ingestion pipeline.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks with inherited metadata.
        """
        text = document.text
        if not text or not text.strip():
            logger.warning("Document %s has no text — skipping.", document.doc_id)
            return []

        sentences = _split_sentences(text)
        if not sentences:
            logger.warning(
                "Document %s yielded no sentences after splitting.", document.doc_id
            )
            return []

        # Determine adaptive target size based on content density
        density = _technical_density(text)
        adaptive_max = self._adaptive_max(density)
        logger.debug(
            "Doc %s: density=%.2f, adaptive_max=%d tokens",
            document.doc_id,
            density,
            adaptive_max,
        )

        raw_chunks = self._merge_sentences(sentences, adaptive_max)

        # Build Chunk objects
        chunks: List[Chunk] = []
        total = len(raw_chunks)
        for idx, chunk_text in enumerate(raw_chunks):
            token_count = _count_tokens(chunk_text, self._tokenizer)
            metadata = {
                **document.metadata,
                "chunk_index": idx,
                "chunk_total": total,
                "token_count": token_count,
                "doc_id": document.doc_id,
                "source": document.source,
                "technical_density": round(density, 3),
            }
            chunks.append(
                Chunk(
                    chunk_id=uuid.uuid4().hex,
                    text=chunk_text,
                    metadata=metadata,
                    doc_id=document.doc_id,
                    token_count=token_count,
                )
            )

        logger.info(
            "Document %s → %d chunk(s) (density=%.2f, target=%d tokens)",
            document.doc_id,
            len(chunks),
            density,
            adaptive_max,
        )
        return chunks

    def chunk_documents(self, documents: "List[Document]") -> List[Chunk]:
        """Convenience method — chunk multiple documents at once."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_max(self, density: float) -> int:
        """Compute the target chunk size (in tokens) given a density score.

        Dense technical content → shorter chunks (down to 50 % of max).
        Narrative content → full max_tokens.
        """
        scale = 1.0 - 0.5 * density  # range [0.5, 1.0]
        target = int(self.max_tokens * scale)
        return max(target, self.min_chunk_tokens)

    def _merge_sentences(self, sentences: List[str], target_tokens: int) -> List[str]:
        """Greedily merge sentences into chunks up to *target_tokens*.

        Implements token-level overlap between consecutive chunks by
        re-including trailing sentences from the previous chunk.
        """
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = _count_tokens(sentence, self._tokenizer)

            # If a single sentence exceeds the target, emit it as its own chunk
            if sent_tokens > target_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                chunks.append(sentence)
                continue

            if current_tokens + sent_tokens > target_tokens and current_sentences:
                # Emit the current chunk
                chunks.append(" ".join(current_sentences))

                # Compute overlap: walk backwards to collect ~overlap_tokens
                overlap_sents: List[str] = []
                overlap_tok = 0
                for s in reversed(current_sentences):
                    s_tok = _count_tokens(s, self._tokenizer)
                    if overlap_tok + s_tok > self.overlap_tokens:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tok += s_tok

                current_sentences = overlap_sents
                current_tokens = overlap_tok

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            tok = _count_tokens(chunk_text, self._tokenizer)
            if chunks and tok < self.min_chunk_tokens:
                # Merge small trailing fragment into previous chunk
                chunks[-1] = chunks[-1] + " " + chunk_text
            else:
                chunks.append(chunk_text)

        return chunks
