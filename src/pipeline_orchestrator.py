"""
Enterprise RAG Pipeline Orchestrator
Wires together all components: ingestion, chunking, embedding, retrieval,
query routing, SQL analytics, knowledge graph, evaluation, and feedback.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on the path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_DIR,
    EVALUATION_DIR,
    SQLITE_DB_PATH,
    TOP_K,
    VECTOR_DB_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter: make Retriever compatible with MultiHopRetriever's BaseRetriever
# protocol, which expects retrieve(query, top_k) -> List[Dict[str, Any]]
# ---------------------------------------------------------------------------

class _RetrieverAdapter:
    """Wraps the Retriever so that its output matches the dict-based protocol
    expected by MultiHopRetriever."""

    def __init__(self, retriever) -> None:
        self._retriever = retriever

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        results = self._retriever.retrieve(query, top_k=top_k)
        adapted: List[Dict[str, Any]] = []
        for chunk, score in results:
            adapted.append({
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": score,
                "metadata": chunk.metadata,
                "source": chunk.metadata.get("source", chunk.doc_id),
            })
        return adapted


# ---------------------------------------------------------------------------
# Adapter: make VectorStore + EmbeddingEngine behave like the
# HybridRetriever's VectorRetriever protocol which expects
# search(query, top_k) -> List[Tuple[int, float]]
# ---------------------------------------------------------------------------

class _VectorSearchAdapter:
    """Adapts EmbeddingEngine + VectorStore into the VectorRetriever protocol
    that HybridRetriever expects: search(query, top_k) -> [(idx, score)].

    The adapter maps each FAISS result back to its position in the chunk
    list by matching chunk_id.  If the chunk list is not available (or
    the chunk_id is not found), it falls back to using the result's
    ordinal position.
    """

    def __init__(self, embedding_engine, vector_store, chunk_id_to_index: Optional[Dict[str, int]] = None) -> None:
        self._ee = embedding_engine
        self._vs = vector_store
        self._chunk_id_to_index = chunk_id_to_index or {}

    def search(self, query: str, top_k: int = 5) -> List:
        query_vec = self._ee.embed_query(query)
        raw = self._vs.search(query_vec, top_k=top_k)
        results = []
        for hit in raw:
            score = hit.get("score", 0.0)
            chunk_id = hit.get("chunk_id", "")
            # Map chunk_id back to the chunk-list index
            idx = self._chunk_id_to_index.get(chunk_id, len(results))
            results.append((idx, score))
        return results


# ---------------------------------------------------------------------------
# EnterpriseRAGOrchestrator
# ---------------------------------------------------------------------------

class EnterpriseRAGOrchestrator:
    """Master orchestrator that wires all RAG components together.

    Components are initialised lazily so that the orchestrator can be
    created quickly and only pays the cost of loading heavy models when
    they are actually needed.
    """

    def __init__(self) -> None:
        # Component slots -- all lazily populated
        self._ingest_pipeline = None
        self._chunker = None
        self._embedding_engine = None
        self._vector_store = None
        self._retriever = None
        self._bm25_search = None
        self._hybrid_retriever = None
        self._rag_pipeline = None
        self._sql_pipeline = None
        self._query_router = None
        self._knowledge_graph_builder = None
        self._graph_retriever = None
        self._multi_hop_retriever = None
        self._source_attributor = None
        self._feedback_store = None
        self._evaluator = None
        self._clip_engine = None
        self._visual_store = None
        self._asset_store = None
        self._captioner = None
        self._multimodal_retriever = None
        self._answer_generator = None

        # State
        self._documents: List[Any] = []
        self._chunks: List[Any] = []
        self._is_setup = False
        self._stats = {
            "documents_ingested": 0,
            "chunks_created": 0,
            "queries_answered": 0,
            "setup_time_seconds": 0.0,
        }

        logger.info("EnterpriseRAGOrchestrator created.")

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

    def _get_ingest_pipeline(self):
        if self._ingest_pipeline is None:
            from src.ingestion.pipeline import IngestPipeline
            self._ingest_pipeline = IngestPipeline()
        return self._ingest_pipeline

    def _get_chunker(self):
        if self._chunker is None:
            from src.chunking.chunker import SemanticChunker
            self._chunker = SemanticChunker()
        return self._chunker

    def _get_embedding_engine(self):
        if self._embedding_engine is None:
            from src.embedding.embed_chunks import EmbeddingEngine
            self._embedding_engine = EmbeddingEngine()
        return self._embedding_engine

    def _get_vector_store(self):
        if self._vector_store is None:
            from src.embedding.store_vector_db import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store

    def _get_retriever(self):
        if self._retriever is None:
            from src.embedding.retrieve import Retriever
            self._retriever = Retriever(
                embedding_engine=self._get_embedding_engine(),
                vector_store=self._get_vector_store(),
            )
        return self._retriever

    def _get_bm25_search(self):
        if self._bm25_search is None:
            from src.retrieval.bm25_search import BM25Search
            self._bm25_search = BM25Search()
        return self._bm25_search

    def _get_hybrid_retriever(self):
        if self._hybrid_retriever is None:
            from src.retrieval.hybrid_retriever import HybridRetriever
            # The HybridRetriever needs a VectorRetriever and BM25Searcher
            # that both expose search(query, top_k) -> [(idx, score)].
            # BM25Search already matches the BM25Searcher protocol.
            # For the vector side, we wrap with an adapter.
            # Build a chunk_id -> index mapping so FAISS results can be
            # aligned with the BM25 corpus indices.
            chunk_id_map = {
                c.chunk_id: i for i, c in enumerate(self._chunks)
            } if self._chunks else {}

            vector_adapter = _VectorSearchAdapter(
                self._get_embedding_engine(),
                self._get_vector_store(),
                chunk_id_to_index=chunk_id_map,
            )
            self._hybrid_retriever = HybridRetriever(
                vector_retriever=vector_adapter,
                bm25_searcher=self._get_bm25_search(),
                chunks=self._chunks if self._chunks else None,
            )
        return self._hybrid_retriever

    def _get_rag_pipeline(self):
        if self._rag_pipeline is None:
            from src.qa_pipeline import RAGPipeline
            self._rag_pipeline = RAGPipeline(retriever=self._get_retriever())
        return self._rag_pipeline

    def _get_sql_pipeline(self):
        if self._sql_pipeline is None:
            try:
                from src.sql_tool.sql_pipeline import SQLAnalyticsPipeline
                self._sql_pipeline = SQLAnalyticsPipeline()
            except Exception as exc:
                logger.warning("SQLAnalyticsPipeline unavailable: %s", exc)
        return self._sql_pipeline

    def _get_query_router(self):
        if self._query_router is None:
            from src.agents.query_router import QueryRouter
            self._query_router = QueryRouter()
            self._register_handlers()
        return self._query_router

    def _get_knowledge_graph_builder(self):
        if self._knowledge_graph_builder is None:
            from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
            self._knowledge_graph_builder = KnowledgeGraphBuilder()
        return self._knowledge_graph_builder

    def _get_graph_retriever(self):
        if self._graph_retriever is None:
            builder = self._get_knowledge_graph_builder()
            if builder.graph.number_of_nodes() > 0:
                from src.knowledge_graph.graph_retriever import GraphRetriever
                self._graph_retriever = GraphRetriever(builder.graph)
        return self._graph_retriever

    def _get_multi_hop_retriever(self):
        if self._multi_hop_retriever is None:
            from src.retrieval.multi_hop import MultiHopRetriever
            adapter = _RetrieverAdapter(self._get_retriever())
            self._multi_hop_retriever = MultiHopRetriever(
                base_retriever=adapter,
                top_k=TOP_K,
            )
        return self._multi_hop_retriever

    def _get_source_attributor(self):
        if self._source_attributor is None:
            from src.retrieval.source_attribution import SourceAttributor
            self._source_attributor = SourceAttributor()
        return self._source_attributor

    def _get_feedback_store(self):
        if self._feedback_store is None:
            try:
                from src.feedback.feedback_store import FeedbackStore
                self._feedback_store = FeedbackStore()
            except Exception as exc:
                logger.warning("FeedbackStore unavailable: %s", exc)
        return self._feedback_store

    def _get_evaluator(self):
        if self._evaluator is None:
            from src.evaluation.eval_pipeline import RAGEvaluator
            self._evaluator = RAGEvaluator(
                embedding_engine=self._get_embedding_engine()
            )
        return self._evaluator

    def _get_clip_engine(self):
        if self._clip_engine is None:
            from src.embedding.clip_engine import CLIPEngine
            self._clip_engine = CLIPEngine()
        return self._clip_engine

    def _get_visual_store(self):
        if self._visual_store is None:
            from src.embedding.visual_store import VisualVectorStore
            self._visual_store = VisualVectorStore()
        return self._visual_store

    def _get_asset_store(self):
        if self._asset_store is None:
            from src.ingestion.asset_store import AssetStore
            self._asset_store = AssetStore()
        return self._asset_store

    def _get_captioner(self):
        if self._captioner is None:
            from config.settings import GEMINI_API_KEY
            if GEMINI_API_KEY:
                from src.gemini.captioner import ImageCaptioner
                self._captioner = ImageCaptioner(api_key=GEMINI_API_KEY)
        return self._captioner

    def _get_answer_generator(self):
        if self._answer_generator is None:
            from config.settings import GEMINI_API_KEY, GEMINI_MODEL
            if GEMINI_API_KEY:
                from src.gemini.generator import AnswerGenerator
                self._answer_generator = AnswerGenerator(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
        return self._answer_generator

    def _get_multimodal_retriever(self):
        if self._multimodal_retriever is None:
            from src.retrieval.multimodal_retriever import MultimodalRetriever
            from src.retrieval.query_analyzer import QueryAnalyzer

            def text_search(query, top_k=5):
                retriever = self._get_retriever()
                results = retriever.retrieve(query, top_k=top_k)
                return [{"text": chunk.text, "doc_id": chunk.doc_id, "score": score,
                         "modality": chunk.metadata.get("modality", "text"),
                         "source": chunk.metadata.get("source", chunk.doc_id),
                         **chunk.metadata} for chunk, score in results]

            def visual_search(query, top_k=5):
                try:
                    clip = self._get_clip_engine()
                    vs = self._get_visual_store()
                    query_vec = clip.embed_text(query)
                    return vs.search(query_vec, top_k=top_k)
                except Exception:
                    return []

            self._multimodal_retriever = MultimodalRetriever(
                text_search_fn=text_search, visual_search_fn=visual_search, query_analyzer=QueryAnalyzer())
        return self._multimodal_retriever

    # ------------------------------------------------------------------
    # Handler registration for QueryRouter
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """Register concrete handler functions on the query router."""
        router = self._query_router

        router.register_handler("sql_pipeline", self._handle_sql)
        router.register_handler("multi_hop_retrieval", self._handle_reasoning)
        router.register_handler("multi_document_retrieval", self._handle_standard)
        router.register_handler("standard_rag", self._handle_standard)

    # ------------------------------------------------------------------
    # Handler implementations
    # ------------------------------------------------------------------

    def _handle_sql(self, query: str, **kwargs) -> Dict[str, Any]:
        """Route to SQL analytics pipeline."""
        pipeline = self._get_sql_pipeline()
        if pipeline is None:
            return {
                "answer": "SQL analytics pipeline is not available.",
                "query_type": "sql",
                "sources": [],
            }
        try:
            result = pipeline.analyze(query)
            return {
                "answer": result.get("insight", "No insight generated."),
                "query_type": "sql",
                "sql": result.get("sql", ""),
                "results": result.get("results", []),
                "chart_html": result.get("chart_html", ""),
                "sources": [],
            }
        except Exception as exc:
            logger.error("SQL pipeline failed: %s", exc)
            return {
                "answer": f"SQL analytics failed: {exc}",
                "query_type": "sql",
                "sources": [],
            }

    def _handle_reasoning(self, query: str, **kwargs) -> Dict[str, Any]:
        """Route to multi-hop retriever for reasoning queries."""
        multi_hop = self._get_multi_hop_retriever()
        result = multi_hop.retrieve(query, max_hops=3)

        all_chunks = result.get("all_chunks", [])
        reasoning_chain = result.get("reasoning_chain", [])

        # Generate answer using RAG pipeline's fallback (context-based)
        if all_chunks:
            context = "\n\n".join(
                c.get("text", "") for c in all_chunks[:5]
            )
            answer_text = (
                f"Based on multi-hop reasoning across {len(all_chunks)} passages:\n\n"
                f"{context[:2000]}"
            )
        else:
            answer_text = "No relevant information found through multi-hop retrieval."

        return {
            "answer": answer_text,
            "query_type": "reasoning",
            "reasoning_chain": reasoning_chain,
            "sources": [
                {
                    "text": c.get("text", "")[:200],
                    "source": c.get("metadata", {}).get("source", c.get("doc_id", "unknown")),
                    "score": c.get("score", 0.0),
                }
                for c in all_chunks[:5]
            ],
        }

    def _handle_standard(self, query: str, **kwargs) -> Dict[str, Any]:
        """Standard RAG retrieval and answer generation.

        Tries the multimodal path (AnswerGenerator + MultimodalRetriever) first;
        falls back to the text-only RAGPipeline if the generator is unavailable
        or raises an exception.
        """
        try:
            generator = self._get_answer_generator()
            if generator is not None:
                retriever = self._get_multimodal_retriever()
                chunks = retriever.retrieve(query)
                result = generator.generate(query, chunks)
                return {
                    "answer": result.get("answer", ""),
                    "query_type": "standard",
                    "confidence": result.get("confidence", 0.0),
                    "sources": result.get("sources", []),
                    "faithfulness": result.get("faithfulness", None),
                    "citation_verified": result.get("citation_verified", False),
                    "warning": result.get("warning", None),
                }
        except Exception as exc:
            logger.warning("Multimodal path failed, falling back to text-only RAG: %s", exc)

        # Fallback: text-only RAG pipeline
        rag = self._get_rag_pipeline()
        result = rag.answer(query)
        return {
            "answer": result.get("answer", ""),
            "query_type": "standard",
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("sources", []),
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run the full setup pipeline.

        Steps:
            1. Ingest all documents from the data/ directory
            2. Chunk all documents
            3. Embed chunks and store in FAISS
            4. Build BM25 index
            5. Set up SQLite database from orders.csv
            6. Build knowledge graph
            7. Register query router handlers

        Args:
            data_dir: Path to the data directory.  Defaults to config DATA_DIR.

        Returns:
            A dict summarising setup results.
        """
        t_start = time.time()
        data_path = Path(data_dir) if data_dir else DATA_DIR
        setup_report: Dict[str, Any] = {}

        # -- Step 1: Ingest documents ----------------------------------------
        logger.info("Step 1/7: Ingesting documents from %s ...", data_path)
        try:
            pipeline = self._get_ingest_pipeline()
            self._documents = pipeline.ingest_directory(str(data_path))
            setup_report["ingestion"] = {
                "documents": len(self._documents),
                "stats": pipeline.stats,
            }
            self._stats["documents_ingested"] = len(self._documents)
            logger.info("  Ingested %d document(s).", len(self._documents))
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            setup_report["ingestion"] = {"error": str(exc)}

        # -- Step 2: Chunk documents -----------------------------------------
        logger.info("Step 2/7: Chunking documents ...")
        try:
            chunker = self._get_chunker()
            self._chunks = chunker.chunk_documents(self._documents)
            setup_report["chunking"] = {"chunks": len(self._chunks)}
            self._stats["chunks_created"] = len(self._chunks)
            logger.info("  Created %d chunk(s).", len(self._chunks))
        except Exception as exc:
            logger.error("Chunking failed: %s", exc)
            setup_report["chunking"] = {"error": str(exc)}

        # -- Step 3: Embed and store in FAISS --------------------------------
        logger.info("Step 3/7: Embedding and storing in FAISS ...")
        try:
            ee = self._get_embedding_engine()
            embedded = ee.embed_chunks(self._chunks)
            chunks_only = [c for c, _ in embedded]
            embeddings_only = [e for _, e in embedded]

            vs = self._get_vector_store()
            vs.store_embeddings(chunks_only, embeddings_only)
            vs.save()
            ee.save_cache()

            setup_report["embedding"] = {
                "vectors_stored": vs.total_vectors,
            }
            logger.info("  Stored %d vectors in FAISS.", vs.total_vectors)
        except Exception as exc:
            logger.error("Embedding/storage failed: %s", exc)
            setup_report["embedding"] = {"error": str(exc)}

        # -- Step 4: Build BM25 index ----------------------------------------
        logger.info("Step 4/7: Building BM25 index ...")
        try:
            bm25 = self._get_bm25_search()
            chunk_texts = [c.text for c in self._chunks]
            if chunk_texts:
                bm25.add_documents(chunk_texts)
                bm25_path = VECTOR_DB_DIR / "bm25_index.pkl"
                bm25.save(bm25_path)
            setup_report["bm25"] = {"corpus_size": bm25.corpus_size}
            logger.info("  BM25 index built with %d documents.", bm25.corpus_size)
        except Exception as exc:
            logger.error("BM25 index build failed: %s", exc)
            setup_report["bm25"] = {"error": str(exc)}

        # -- Step 5: Set up SQLite from orders.csv ---------------------------
        logger.info("Step 5/7: Setting up SQLite database ...")
        csv_path = data_path / "orders.csv"
        if csv_path.exists():
            try:
                from src.sql_tool.db_setup import setup_database
                db_path = setup_database(str(csv_path))
                setup_report["sql_database"] = {"path": str(db_path)}
                logger.info("  SQLite database ready at %s.", db_path)
            except Exception as exc:
                logger.error("SQLite setup failed: %s", exc)
                setup_report["sql_database"] = {"error": str(exc)}
        else:
            logger.warning("  No orders.csv found at %s -- skipping SQL setup.", csv_path)
            setup_report["sql_database"] = {"skipped": "orders.csv not found"}

        # -- Step 6: Build knowledge graph -----------------------------------
        logger.info("Step 6/7: Building knowledge graph ...")
        try:
            kg_builder = self._get_knowledge_graph_builder()
            # Convert Document objects to dicts for the graph builder
            doc_dicts = [
                {
                    "text": doc.text,
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "metadata": doc.metadata,
                }
                for doc in self._documents
            ]
            graph = kg_builder.build_graph(doc_dicts)
            setup_report["knowledge_graph"] = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "triples": len(kg_builder.triples),
            }
            # Save graph to disk
            kg_path = VECTOR_DB_DIR / "knowledge_graph.pkl"
            kg_builder.save_graph(kg_path)
            logger.info(
                "  Knowledge graph: %d nodes, %d edges.",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
        except Exception as exc:
            logger.error("Knowledge graph build failed: %s", exc)
            setup_report["knowledge_graph"] = {"error": str(exc)}

        # -- Step 7: Register query router handlers --------------------------
        logger.info("Step 7/7: Registering query router handlers ...")
        try:
            self._get_query_router()
            setup_report["query_router"] = {"status": "ok"}
            logger.info("  Query router handlers registered.")
        except Exception as exc:
            logger.error("Query router setup failed: %s", exc)
            setup_report["query_router"] = {"error": str(exc)}

        elapsed = time.time() - t_start
        self._stats["setup_time_seconds"] = round(elapsed, 2)
        self._is_setup = True

        logger.info(
            "Setup complete in %.1f seconds: %d docs, %d chunks.",
            elapsed,
            self._stats["documents_ingested"],
            self._stats["chunks_created"],
        )

        setup_report["elapsed_seconds"] = round(elapsed, 2)
        return setup_report

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """Answer a question using the full pipeline.

        Steps:
            1. Classify the query via QueryRouter
            2. Route to the appropriate handler
            3. Apply source attribution
            4. Record feedback opportunity

        Args:
            question: Natural-language question.

        Returns:
            Comprehensive response dict with answer, sources, confidence,
            query_type, attribution, and metadata.
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a question.",
                "query_type": "empty",
                "sources": [],
                "confidence": 0.0,
            }

        t_start = time.time()

        # Step 1: Classify
        router = self._get_query_router()
        classification = router.classify(question)
        query_type = classification.get("category", "factual")
        suggested_handler = classification.get("suggested_handler", "standard_rag")
        classification_confidence = classification.get("confidence", 0.0)

        logger.info(
            "Query classified as '%s' (handler='%s', confidence=%.2f): %s",
            query_type,
            suggested_handler,
            classification_confidence,
            question[:80],
        )

        # Step 2: Route to handler
        handler = router.route(question)
        try:
            handler_result = handler(question)
        except Exception as exc:
            logger.error("Handler '%s' failed: %s", suggested_handler, exc)
            # Fall back to standard RAG
            handler_result = self._handle_standard(question)

        # Step 3: Apply source attribution
        answer = handler_result.get("answer", "")
        sources = handler_result.get("sources", [])

        attribution = {}
        try:
            attributor = self._get_source_attributor()
            # Convert sources to the format SourceAttributor expects
            attribution_chunks = []
            for src in sources:
                if isinstance(src, dict):
                    attribution_chunks.append({
                        "text": src.get("text", src.get("text_preview", "")),
                        "doc_id": src.get("doc_id", ""),
                        "source": src.get("source", "unknown"),
                        "score": src.get("score", 0.0),
                    })
            if attribution_chunks:
                attribution = attributor.attribute(answer, question, attribution_chunks)
        except Exception as exc:
            logger.warning("Source attribution failed: %s", exc)

        # Step 4: Record feedback opportunity
        query_id = f"q_{int(time.time() * 1000)}"
        try:
            store = self._get_feedback_store()
            if store is not None:
                # Log retrieval events
                for rank, src in enumerate(sources[:5], start=1):
                    if isinstance(src, dict):
                        store.record_retrieval(
                            query=question,
                            chunk_id=src.get("chunk_id", src.get("doc_id", f"chunk_{rank}")),
                            rank=rank,
                            score=float(src.get("score", 0.0)),
                        )
        except Exception as exc:
            logger.warning("Feedback logging failed: %s", exc)

        elapsed = time.time() - t_start
        self._stats["queries_answered"] += 1

        # Build response
        response = {
            "answer": answer,
            "query_type": query_type,
            "handler": suggested_handler,
            "classification_confidence": classification_confidence,
            "sources": sources,
            "confidence": handler_result.get(
                "confidence",
                attribution.get("confidence", 0.0),
            ),
            "attribution": {
                "faithfulness": attribution.get("faithfulness", None),
                "coverage": attribution.get("coverage", None),
                "warning": attribution.get("warning", None),
            },
            "query_id": query_id,
            "latency_ms": round(elapsed * 1000, 1),
        }

        # Include extra fields from specific handlers
        if "sql" in handler_result:
            response["sql"] = handler_result["sql"]
        if "results" in handler_result:
            response["sql_results"] = handler_result["results"]
        if "chart_html" in handler_result:
            response["chart_html"] = handler_result["chart_html"]
        if "reasoning_chain" in handler_result:
            response["reasoning_chain"] = handler_result["reasoning_chain"]

        logger.info(
            "Query answered in %.0f ms (type=%s, handler=%s).",
            elapsed * 1000,
            query_type,
            suggested_handler,
        )
        return response

    # ------------------------------------------------------------------
    # Document upload
    # ------------------------------------------------------------------

    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest, chunk, embed, and store a single document.

        Args:
            file_path: Path to the document file.

        Returns:
            Dict with doc_id, chunks_created, and status.
        """
        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "detail": f"File not found: {file_path}"}

        try:
            # Ingest
            pipeline = self._get_ingest_pipeline()
            docs = pipeline.ingest(str(path))
            if not docs:
                return {"status": "warning", "detail": "No documents created from file."}

            self._documents.extend(docs)
            doc_id = docs[0].doc_id

            # Chunk
            chunker = self._get_chunker()
            new_chunks = chunker.chunk_documents(docs)
            self._chunks.extend(new_chunks)

            # Embed and store
            if new_chunks:
                ee = self._get_embedding_engine()
                embedded = ee.embed_chunks(new_chunks)
                chunks_only = [c for c, _ in embedded]
                embeddings_only = [e for _, e in embedded]

                vs = self._get_vector_store()
                vs.store_embeddings(chunks_only, embeddings_only)
                vs.save()
                ee.save_cache()

                # Update BM25
                bm25 = self._get_bm25_search()
                bm25.add_documents([c.text for c in new_chunks])

                # Update knowledge graph
                try:
                    kg = self._get_knowledge_graph_builder()
                    doc_dicts = [
                        {"text": d.text, "doc_id": d.doc_id, "source": d.source, "metadata": d.metadata}
                        for d in docs
                    ]
                    kg.build_graph(doc_dicts)
                except Exception as exc:
                    logger.warning("KG update failed for upload: %s", exc)

            self._stats["documents_ingested"] += len(docs)
            self._stats["chunks_created"] += len(new_chunks)

            return {
                "status": "success",
                "doc_id": doc_id,
                "documents_created": len(docs),
                "chunks_created": len(new_chunks),
            }

        except Exception as exc:
            logger.error("Upload failed for %s: %s", file_path, exc)
            return {"status": "error", "detail": str(exc)}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive system statistics."""
        stats = dict(self._stats)
        stats["is_setup"] = self._is_setup

        try:
            vs = self._get_vector_store()
            stats["total_vectors"] = vs.total_vectors
        except Exception:
            stats["total_vectors"] = 0

        try:
            bm25 = self._get_bm25_search()
            stats["bm25_corpus_size"] = bm25.corpus_size
        except Exception:
            stats["bm25_corpus_size"] = 0

        try:
            kg = self._get_knowledge_graph_builder()
            stats["knowledge_graph_nodes"] = kg.graph.number_of_nodes()
            stats["knowledge_graph_edges"] = kg.graph.number_of_edges()
        except Exception:
            stats["knowledge_graph_nodes"] = 0
            stats["knowledge_graph_edges"] = 0

        stats["sql_database_exists"] = SQLITE_DB_PATH.exists()

        try:
            store = self._get_feedback_store()
            if store is not None:
                fb_stats = store.get_feedback_stats()
                stats["feedback_total"] = fb_stats.get("total_feedback", 0)
                stats["feedback_avg_rating"] = fb_stats.get("avg_rating", None)
        except Exception:
            stats["feedback_total"] = 0

        return stats

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the evaluation pipeline on test cases.

        Args:
            test_cases: Custom test cases.  If None, uses the built-in
                TEST_CASES from src.evaluation.test_cases.

        Returns:
            Evaluation results dict with per_case, aggregated, and
            per_category metrics.
        """
        if test_cases is None:
            from src.evaluation.test_cases import TEST_CASES
            test_cases = TEST_CASES

        evaluator = self._get_evaluator()

        def _run_query(query: str) -> Dict[str, Any]:
            """Run a query through the orchestrator and format for evaluation."""
            if not query.strip():
                return {"answer": "", "retrieved_chunks": []}

            result = self.query(query)
            # Extract chunk texts for evaluation
            retrieved_chunks = []
            for src in result.get("sources", []):
                if isinstance(src, dict):
                    text = src.get("text", src.get("text_preview", ""))
                    if text:
                        retrieved_chunks.append(text)
            return {
                "answer": result.get("answer", ""),
                "retrieved_chunks": retrieved_chunks,
            }

        results = evaluator.evaluate_batch(
            test_cases,
            run_pipeline_fn=_run_query,
        )

        # Save report
        try:
            report_path = evaluator.generate_report(results, EVALUATION_DIR)
            logger.info("Evaluation report saved to %s.", report_path)
            results["report_path"] = str(report_path)
        except Exception as exc:
            logger.warning("Could not save evaluation report: %s", exc)

        return results


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _print_result(result: Dict[str, Any], indent: int = 0) -> None:
    """Print a query result nicely."""
    prefix = " " * indent
    print(f"{prefix}Answer: {result.get('answer', '')[:500]}")
    print(f"{prefix}Query Type: {result.get('query_type', 'unknown')}")
    print(f"{prefix}Handler: {result.get('handler', 'unknown')}")
    print(f"{prefix}Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"{prefix}Latency: {result.get('latency_ms', 0):.0f} ms")

    sources = result.get("sources", [])
    if sources:
        print(f"{prefix}Sources ({len(sources)}):")
        for s in sources[:3]:
            if isinstance(s, dict):
                print(f"{prefix}  - {s.get('source', 'unknown')} "
                      f"(score: {s.get('score', 0):.3f})")

    attribution = result.get("attribution", {})
    if attribution.get("faithfulness") is not None:
        print(f"{prefix}Faithfulness: {attribution['faithfulness']:.2f}")
    if attribution.get("warning"):
        print(f"{prefix}Warning: {attribution['warning']}")

    if "sql" in result:
        print(f"{prefix}SQL: {result['sql']}")
    if "reasoning_chain" in result:
        print(f"{prefix}Reasoning chain:")
        for step in result["reasoning_chain"][:5]:
            print(f"{prefix}  - {step}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 72)
    print("  Enterprise RAG Pipeline - Full Demo")
    print("=" * 72)

    # Create orchestrator and run setup
    orchestrator = EnterpriseRAGOrchestrator()

    print("\n[1/3] Running setup ...")
    setup_report = orchestrator.setup()
    print(f"\nSetup Report:")
    print(json.dumps(setup_report, indent=2, default=str))

    # Demo queries showing different routing paths
    print("\n[2/3] Running demo queries ...\n")

    demo_queries = [
        # SQL/analytics query
        "How many orders are there by region?",
        # Reasoning query (multi-hop)
        "Why are there delays in the supply chain?",
        # Standard factual query
        "What products are available in the system?",
        # Summary query
        "Summarize the overall order status distribution",
        # Comparison query
        "Compare the performance of different suppliers",
    ]

    for i, q in enumerate(demo_queries, 1):
        print(f"\n{'─' * 72}")
        print(f"  Query {i}: {q}")
        print(f"{'─' * 72}")
        result = orchestrator.query(q)
        _print_result(result, indent=2)

    # Statistics
    print(f"\n{'─' * 72}")
    print("\n[3/3] System Statistics:")
    stats = orchestrator.get_stats()
    print(json.dumps(stats, indent=2, default=str))

    print("\n" + "=" * 72)
    print("  Demo complete.")
    print("=" * 72)
