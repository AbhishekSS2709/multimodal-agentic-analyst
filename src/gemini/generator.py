"""Answer generator using Gemini for structured RAG responses with citation verification."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.gemini.client import GeminiClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on provided context. "
    "Rules: "
    "1. ONLY use info from sources. "
    "2. Cite using [Source N]. "
    "3. Say 'I don't know' if insufficient. "
    "4. Be concise. "
    "5. Respond in JSON."
)


class AnswerGenerator:
    """Generate structured answers from retrieved chunks using the Gemini API.

    Parameters
    ----------
    api_key:
        Gemini API key. Must be non-empty.
    max_images:
        Maximum number of PIL images to pass to the model alongside text (default 3).
    model_name:
        Model identifier passed to GeminiClient (default ``"gemini-2.5-flash"``).
    """

    def __init__(
        self,
        api_key: str,
        max_images: int = 3,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self._client = GeminiClient(api_key=api_key, model_name=model_name)
        self.max_images = max_images

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        asset_loader=None,
    ) -> dict:
        """Generate a structured answer from retrieved chunks.

        Parameters
        ----------
        query:
            The user's question.
        retrieved_chunks:
            List of chunk dicts with at least ``"text"``, ``"source"``, and ``"type"`` keys.
        asset_loader:
            Optional asset loader providing a ``load_image(chunk)`` method to
            resolve images for image-type chunks.

        Returns
        -------
        dict
            Enriched answer dict including ``citation_verified``,
            ``faithfulness_score``, and optionally ``warning``.
        """
        context_str, images = self._build_context_with_images(retrieved_chunks, asset_loader)

        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Respond as JSON with keys: answer (str), source_ids (list[int]), confidence (float)."
        )

        raw_answer = self._client.generate_json(
            prompt,
            system=_SYSTEM_PROMPT,
            images=images if images else None,
        )

        return self._post_process(raw_answer, retrieved_chunks)

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks as numbered source entries.

        Each entry has the form::

            [Source N] (source: X, type: Y)
            text content

        Parameters
        ----------
        chunks:
            List of chunk dicts.

        Returns
        -------
        str
            Formatted context string.
        """
        parts: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            chunk_type = chunk.get("type", "text")
            text = chunk.get("text", "")
            header = f"[Source {idx}] (source: {source}, type: {chunk_type})"
            parts.append(f"{header}\n{text}")
        return "\n\n".join(parts)

    def _build_context_with_images(
        self,
        chunks: List[Dict[str, Any]],
        asset_loader=None,
    ) -> Tuple[str, List[Any]]:
        """Format chunks as numbered source entries and collect up to max_images PIL images.

        Images are collected from chunks in order, first trying
        ``asset_loader.load_image(chunk)`` (if an asset_loader is provided),
        then falling back to the ``"image"`` key in the chunk dict.

        Parameters
        ----------
        chunks:
            List of chunk dicts.
        asset_loader:
            Optional object with a ``load_image(chunk)`` method.

        Returns
        -------
        Tuple[str, List]
            A ``(context_string, images_list)`` tuple.  ``images_list`` contains
            at most ``self.max_images`` PIL Image objects.
        """
        parts: List[str] = []
        images: List[Any] = []

        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            chunk_type = chunk.get("type", "text")
            text = chunk.get("text", "")
            header = f"[Source {idx}] (source: {source}, type: {chunk_type})"
            parts.append(f"{header}\n{text}")

            # Collect image if we haven't reached the cap yet
            if len(images) < self.max_images:
                img = None
                if asset_loader is not None:
                    try:
                        img = asset_loader.load_image(chunk)
                    except Exception:  # noqa: BLE001
                        pass
                if img is None:
                    img = chunk.get("image")
                if img is not None:
                    images.append(img)

        return "\n\n".join(parts), images

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self, answer: dict, sources: List[Dict[str, Any]]) -> dict:
        """Enrich the raw model answer with verification metadata.

        Adds the following keys to the returned dict:

        ``citation_verified``
            ``True`` if every ``source_id`` in ``answer["source_ids"]`` is within
            the valid range ``[1, len(sources)]``.
        ``faithfulness_score``
            Fraction of words in the answer text that also appear in the
            concatenated source texts (simple word-overlap metric).
        ``warning``
            Present (and truthy) when citations are invalid **or** when
            ``confidence < 0.5``.

        Parameters
        ----------
        answer:
            Dict produced by ``GeminiClient.generate_json``.  Expected keys:
            ``answer`` (str), ``source_ids`` (list[int]),
            ``confidence`` (float, optional).
        sources:
            The original list of retrieved chunks used to build the context.

        Returns
        -------
        dict
            A copy of *answer* with the extra verification keys added.
        """
        result = dict(answer)

        n_sources = len(sources)
        source_ids: List[int] = result.get("source_ids") or []

        # Citation verification: all ids must be in [1, n_sources]
        citation_verified = all(
            isinstance(sid, int) and 1 <= sid <= n_sources
            for sid in source_ids
        )
        result["citation_verified"] = citation_verified

        # Faithfulness score: word overlap between answer text and all source texts
        answer_text: str = result.get("answer", "") or ""
        answer_words = set(answer_text.lower().split())

        all_source_text = " ".join(
            chunk.get("text", "") for chunk in sources
        )
        source_words = set(all_source_text.lower().split())

        if answer_words:
            overlap = answer_words & source_words
            faithfulness_score = len(overlap) / len(answer_words)
        else:
            faithfulness_score = 0.0

        result["faithfulness_score"] = faithfulness_score

        # Warning conditions
        confidence: float = result.get("confidence", 1.0) or 1.0
        warnings: List[str] = []

        if not citation_verified:
            warnings.append("One or more cited source IDs are out of range.")
        if confidence < 0.5:
            warnings.append(f"Low confidence score: {confidence:.2f}.")

        if warnings:
            result["warning"] = " ".join(warnings)

        return result
