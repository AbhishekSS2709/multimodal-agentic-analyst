"""Enterprise RAG System -- top-level source package.

Key components can be imported directly::

    from src.pipeline_orchestrator import EnterpriseRAGOrchestrator
    from src.ingestion.pipeline import IngestPipeline
    from src.ingestion.pdf_loader import Document
    from src.chunking.chunker import SemanticChunker, Chunk
    from src.embedding.embed_chunks import EmbeddingEngine
    from src.embedding.store_vector_db import VectorStore
    from src.embedding.retrieve import Retriever
    from src.qa_pipeline import RAGPipeline

All imports are lazy to avoid circular-import issues and to keep the
import of ``src`` lightweight when only a subset of components is needed.
"""


def __getattr__(name: str):
    """Lazy attribute loader -- imports components on first access."""
    _IMPORT_MAP = {
        "EnterpriseRAGOrchestrator": "src.pipeline_orchestrator",
        "IngestPipeline": "src.ingestion.pipeline",
        "Document": "src.ingestion.pdf_loader",
        "SemanticChunker": "src.chunking.chunker",
        "Chunk": "src.chunking.chunker",
        "EmbeddingEngine": "src.embedding.embed_chunks",
        "VectorStore": "src.embedding.store_vector_db",
        "Retriever": "src.embedding.retrieve",
        "RAGPipeline": "src.qa_pipeline",
    }

    if name in _IMPORT_MAP:
        import importlib
        module = importlib.import_module(_IMPORT_MAP[name])
        return getattr(module, name)

    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = [
    "EnterpriseRAGOrchestrator",
    "IngestPipeline",
    "Document",
    "SemanticChunker",
    "Chunk",
    "EmbeddingEngine",
    "VectorStore",
    "Retriever",
    "RAGPipeline",
]
