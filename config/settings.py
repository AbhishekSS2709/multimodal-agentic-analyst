"""Central configuration for the RAG pipeline."""
import os
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_store"
SQLITE_DB_PATH = PROJECT_ROOT / "data" / "enterprise.db"
FEEDBACK_DB_PATH = PROJECT_ROOT / "data" / "feedback.db"
EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"

# Embedding
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50

# Retrieval
TOP_K = 5
HYBRID_VECTOR_WEIGHT = 0.6
HYBRID_BM25_WEIGHT = 0.4

# LLM — supports multiple free providers
# Options: "ollama", "huggingface", "groq", "none"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")

# HuggingFace (FREE — no API key needed for local models)
HF_MODEL = "google/flan-t5-small"  # tiny (~300MB), free, runs locally

# Ollama (FREE — local, run: ollama pull mistral)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Groq (FREE tier — 14,400 requests/day)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenAI (PAID — optional)
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Vector DB
FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_index"
COLLECTION_NAME = "enterprise_docs"

# SQL
SQL_MAX_ROWS = 100

# Ensure directories exist
VECTOR_DB_DIR.mkdir(exist_ok=True)
EVALUATION_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
