"""End-to-end Retrieval-Augmented Generation (RAG) QA pipeline.

Pipeline flow:
    question -> retrieve top-K chunks -> build prompt -> LLM answer

Supports the OpenAI API when ``OPENAI_API_KEY`` is set; otherwise falls back
to a template-based response that surfaces the retrieved context and sources.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    TOP_K,
)
from src.llm_provider import call_llm, get_active_provider

from src.chunking.chunker import Chunk
from src.embedding.embed_chunks import EmbeddingEngine
from src.embedding.retrieve import Retriever
from src.embedding.store_vector_db import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided "
    "context. If the context does not contain enough information to answer the "
    "question, say so honestly. Always cite the source documents where possible."
)

_CONTEXT_TEMPLATE = """Use the following context to answer the question. Each context passage includes its source.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Provide a clear, concise answer with source citations."""


def _format_context(chunks_with_scores: List[Tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts: List[str] = []
    for i, (chunk, score) in enumerate(chunks_with_scores, 1):
        source = chunk.metadata.get("source", chunk.doc_id)
        parts.append(
            f"[{i}] (source: {source}, relevance: {score:.4f})\n{chunk.text}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Complete RAG question-answering pipeline.

    Parameters
    ----------
    retriever : Retriever | None
        Pre-built retriever; one is created automatically if omitted.
    top_k : int
        Number of chunks to retrieve per question.
    llm_model : str
        OpenAI model name.
    temperature : float
        Sampling temperature for the LLM.
    api_key : str
        OpenAI API key; read from settings/env if not supplied.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        top_k: int = TOP_K,
        llm_model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        api_key: str = OPENAI_API_KEY,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.top_k = top_k
        self.llm_model = llm_model
        self.temperature = temperature
        self._api_key = api_key
        self._openai_client = None

        if self._api_key:
            self._init_openai()

        self._llm_provider = get_active_provider()
        logger.info(
            "RAGPipeline ready (llm=%s, top_k=%d)",
            self._llm_provider,
            top_k,
        )

    # ------------------------------------------------------------------
    # OpenAI client
    # ------------------------------------------------------------------

    def _init_openai(self) -> None:
        """Try to initialise the OpenAI client."""
        try:
            import openai

            self._openai_client = openai.OpenAI(api_key=self._api_key)
            logger.info("OpenAI client initialised (model=%s).", self.llm_model)
        except Exception as exc:
            logger.warning("Could not initialise OpenAI client: %s", exc)
            self._openai_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a question using the full RAG pipeline.

        Parameters
        ----------
        question : str
            Natural-language question.

        Returns
        -------
        dict
            ``answer`` — the generated (or template) answer text.
            ``sources`` — list of source dicts with chunk_id, source, score.
            ``confidence`` — estimated confidence (0-1) based on retrieval scores.
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a question.",
                "sources": [],
                "confidence": 0.0,
            }

        # Step 1: Retrieve relevant chunks
        retrieved: List[Tuple[Chunk, float]] = self.retriever.retrieve(
            question, top_k=self.top_k
        )

        if not retrieved:
            return {
                "answer": "No relevant documents were found for your question.",
                "sources": [],
                "confidence": 0.0,
            }

        # Step 2: Build the prompt
        context_str = _format_context(retrieved)
        user_prompt = _CONTEXT_TEMPLATE.format(
            context=context_str, question=question
        )

        # Step 3: Generate the answer via unified LLM provider
        answer_text = call_llm(user_prompt, _SYSTEM_PROMPT)

        # Step 4: Build source metadata and confidence
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.metadata.get("source", chunk.doc_id),
                "score": round(score, 4),
                "text_preview": chunk.text[:200],
            }
            for chunk, score in retrieved
        ]

        confidence = self._estimate_confidence(retrieved)

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # LLM backends
    # ------------------------------------------------------------------

    def _call_openai(self, user_prompt: str) -> str:
        """Call the OpenAI Chat Completions API."""
        try:
            response = self._openai_client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            return f"[LLM Error] Could not generate answer: {exc}"

    @staticmethod
    def _fallback_answer(
        question: str, retrieved: List[Tuple[Chunk, float]]
    ) -> str:
        """Template-based answer when no LLM API key is available.

        Surfaces the retrieved context directly so the user still gets value.
        """
        lines = [
            f"Question: {question}",
            "",
            "Based on the retrieved documents, here are the most relevant passages:",
            "",
        ]
        for i, (chunk, score) in enumerate(retrieved, 1):
            source = chunk.metadata.get("source", chunk.doc_id)
            lines.append(f"--- Passage {i} (source: {source}, score: {score:.4f}) ---")
            lines.append(chunk.text[:500])
            lines.append("")

        lines.append(
            "Note: No OpenAI API key is configured. Set the OPENAI_API_KEY "
            "environment variable to enable LLM-generated answers."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_confidence(retrieved: List[Tuple[Chunk, float]]) -> float:
        """Heuristic confidence based on retrieval similarity scores.

        Uses the mean of the top scores, weighted toward the top result.
        Inner-product scores on normalised vectors are in [0, 1].
        """
        if not retrieved:
            return 0.0

        scores = [score for _, score in retrieved]
        top_score = scores[0]
        avg_score = sum(scores) / len(scores)

        # Weighted blend: 60 % top score, 40 % average
        confidence = 0.6 * top_score + 0.4 * avg_score
        return max(0.0, min(confidence, 1.0))


# ---------------------------------------------------------------------------
# Demo / CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 72)
    print("  RAG Pipeline — Demo")
    print("=" * 72)

    # ---- Build a small demo index from synthetic documents ----
    from dataclasses import dataclass

    @dataclass
    class _DemoDocument:
        doc_id: str
        text: str
        metadata: dict
        source: str

    demo_docs = [
        _DemoDocument(
            doc_id="demo-001",
            text=(
                "Retrieval-Augmented Generation (RAG) is a technique that combines "
                "information retrieval with language model generation. The retriever "
                "fetches relevant passages from a knowledge base, and the generator "
                "produces an answer conditioned on both the query and the retrieved "
                "context. RAG was introduced by Lewis et al. in 2020 and has since "
                "become a standard approach for grounded question answering."
            ),
            metadata={"topic": "RAG", "file_type": "txt"},
            source="rag_overview.txt",
        ),
        _DemoDocument(
            doc_id="demo-002",
            text=(
                "FAISS (Facebook AI Similarity Search) is a library for efficient "
                "similarity search and clustering of dense vectors. It supports "
                "several index types including flat (brute force), IVF (inverted "
                "file), and HNSW (hierarchical navigable small world). For normalised "
                "vectors, IndexFlatIP (inner product) is equivalent to cosine "
                "similarity search."
            ),
            metadata={"topic": "vector search", "file_type": "txt"},
            source="faiss_docs.txt",
        ),
        _DemoDocument(
            doc_id="demo-003",
            text=(
                "Sentence transformers are neural network models fine-tuned to "
                "produce semantically meaningful sentence embeddings. The BAAI/bge "
                "family of models is widely used for retrieval tasks. These models "
                "map variable-length text into fixed-dimension vectors in a shared "
                "embedding space where semantic similarity correlates with vector "
                "proximity."
            ),
            metadata={"topic": "embeddings", "file_type": "txt"},
            source="embedding_models.txt",
        ),
    ]

    print("\n[1/4] Chunking demo documents...")
    from src.chunking.chunker import SemanticChunker

    chunker = SemanticChunker()
    all_chunks = chunker.chunk_documents(demo_docs)
    print(f"  -> {len(all_chunks)} chunk(s) created")

    print("\n[2/4] Embedding chunks...")
    engine = EmbeddingEngine()
    embedded = engine.embed_chunks(all_chunks)
    chunks_list = [c for c, _ in embedded]
    embeddings_list = [e for _, e in embedded]
    print(f"  -> {len(embeddings_list)} embedding(s) computed")

    print("\n[3/4] Storing in FAISS index...")
    store = VectorStore()
    store.store_embeddings(chunks_list, embeddings_list)
    store.save()
    engine.save_cache()
    print(f"  -> {store.total_vectors} vector(s) in index")

    print("\n[4/4] Running demo queries...\n")
    retriever = Retriever(embedding_engine=engine, vector_store=store)
    pipeline = RAGPipeline(retriever=retriever)

    demo_questions = [
        "What is RAG and who introduced it?",
        "How does FAISS perform similarity search?",
        "What are sentence transformers used for?",
    ]

    for q in demo_questions:
        print("-" * 72)
        print(f"Q: {q}")
        result = pipeline.answer(q)
        print(f"\nA: {result['answer'][:500]}")
        print(f"\nConfidence: {result['confidence']}")
        print(f"Sources: {[s['source'] for s in result['sources']]}")
        print()

    print("=" * 72)
    print("Demo complete.")
