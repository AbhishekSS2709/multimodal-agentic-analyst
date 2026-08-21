# Multimodal Enterprise Research Analyst

A production-shaped RAG system with a **supervisor-orchestrated multi-agent
layer** built on LangGraph, LangChain, and LangSmith.

It ingests eleven file types across text, image, video, audio, and tabular
data; indexes them in dual vector spaces (BGE for text, CLIP for visual); and
answers questions through a graph of specialist agents that plan, retrieve in
parallel, grade their own retrievals, verify their own answers, and pause for a
human before doing anything destructive.

---

## What makes it more than a RAG demo

**It plans instead of pattern-matching.** A supervisor decomposes a question
and dispatches to as many specialists as it needs, in parallel. The keyword
router it replaced could pick exactly one handler.

**It notices when retrieval failed.** The document specialist is a cyclic
subgraph: retrieve → grade → rewrite → retry, bounded at two attempts, then an
honest low-confidence result instead of a confident wrong one.

**It checks its own work.** Every answer is graded for groundedness against the
evidence that produced it before it is returned. A failed grade forces a
re-synthesis that quotes sources directly.

**It asks permission.** Generated SQL is inspected before execution; anything
that is not a plain read pauses the graph for human approval over durable
checkpointed state.

**It runs without an API key.** Every LLM call site has a deterministic
heuristic fallback, so the full 180-test suite runs offline — and LLM-vs-
heuristic becomes a measurable LangSmith experiment rather than an assumption.

See **[docs/AGENTIC_ARCHITECTURE.md](docs/AGENTIC_ARCHITECTURE.md)** for the
graph topology, state reducers, and evaluation design.

---

## Quick start

```bash
pip install -r requirements.txt

# Ingest a corpus and build the indices (first run downloads the BGE model)
python -c "from src.pipeline_orchestrator import EnterpriseRAGOrchestrator; \
           print(EnterpriseRAGOrchestrator().setup(data_dir='data'))"

# Ask the agentic graph, streaming node by node
python -m src.graph.demo "why are there dispatch delays?"

# Or serve the API
uvicorn src.api.main:app --reload
```

No API key is required — the graph runs in heuristic mode. To enable the LLM
paths and tracing, add to `.env`:

```bash
GEMINI_API_KEY=...        # free tier at aistudio.google.com
LANGSMITH_API_KEY=...     # free tier at smith.langchain.com
LANGSMITH_PROJECT=multimodal-agentic-analyst
```

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/v2/query` | Run the multi-agent analyst graph |
| `POST /api/v2/resume` | Resume a thread paused for human approval |
| `GET /api/v2/graph` | Graph topology (Mermaid) and current mode |
| `POST /api/ask` | Original single-pass pipeline (unchanged) |
| `POST /api/upload` | Ingest a document |
| `GET /api/metrics` | System metrics |

```bash
curl -X POST localhost:8000/api/v2/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "why are there dispatch delays?"}'
```

---

## Architecture

```
Ingestion          11 loaders -> MIME detection -> chunking -> captioning
Indexing           BGE text index + CLIP visual index (FAISS)
Retrieval          hybrid BM25 + vector, cross-encoder rerank, multi-hop, knowledge graph
Orchestration      LangGraph supervisor -> 4 parallel specialists -> synthesis -> verification
Safety             human-in-the-loop approval gate, checkpointed threads
Evaluation         LangSmith datasets, 6 evaluators, experiment runner
Serving            FastAPI + Streamlit
```

Two front doors share one retrieval stack: `src/pipeline_orchestrator.py`
(original linear pipeline) and `src/graph/` (agentic layer). Adding the second
required no changes to the ingestion, retrieval, embedding, knowledge-graph,
SQL, or Gemini modules — `src/graph/adapters.py` wraps them.

---

## Evaluation

```python
from src.evaluation.langsmith_eval import push_dataset, run_experiment

push_dataset()                                     # 25 cases, 9 categories
run_experiment(config={"routing": "heuristic"})    # compare configurations
```

Six evaluators — faithfulness, citation accuracy, routing accuracy, modality
match, answer correctness, retry efficiency — implemented as pure functions so
they run in CI without a network.

---

## Tests

```bash
python -m pytest tests/ -q
```

All tests run offline with no API keys.

---

## Stack

Python 3.11 · LangGraph 1.0 · LangChain 1.2 · LangSmith · Gemini 2.5 Flash ·
FAISS · sentence-transformers (BGE) · CLIP · faster-whisper · FastAPI ·
Streamlit · NetworkX · SQLAlchemy
