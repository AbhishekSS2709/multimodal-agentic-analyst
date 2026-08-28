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

**It dispatches several specialists at once.** A supervisor decomposes a
question and fans out in parallel; the keyword router it replaced could pick
exactly one handler. Both a keyword planner and an LLM planner are supported —
and measurement decided which is default: the keyword planner won (see below),
so the LLM planner is opt-in rather than the recommended path.

**It notices when retrieval failed.** The document specialist is a cyclic
subgraph: retrieve → grade → rewrite → retry, bounded at two attempts, then an
honest low-confidence result instead of a confident wrong one.

**It checks its own work.** Every answer is graded for groundedness against the
evidence that produced it before it is returned. A failed grade forces a
re-synthesis that quotes sources directly.

**It asks permission.** Generated SQL is inspected before execution; anything
that is not a plain read pauses the graph for human approval over durable
checkpointed state.

**It runs without an API key.** Every LLM call site in `src/graph/` has a
deterministic heuristic fallback, so the full 342-test suite runs offline — and
LLM-vs-heuristic becomes a measurable experiment rather than an assumption.
(The suite is hermetic about this: `tests/conftest.py` blanks every credential,
because otherwise it only *happened* to be offline when no key was configured.)

**And the two paths split cleanly.** On the same 25 questions the keyword
planner routes more precisely — accuracy 0.960 vs 0.920, precision 0.800 vs
0.377, at 1.5 s per query and zero API cost — while the LLM *answers* better,
correctness 1.000 vs 0.792 and modality match 1.000 vs 0.800. Every question
the LLM wins is one whose answer has to be composed rather than quoted, which
is precisely what an extractive synthesizer cannot do. Numbers, caveats and
method below — including why that 1.000 is a ceiling on a proxy metric rather
than a solved problem.

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
GROQ_API_KEY=...          # free tier at console.groq.com  (preferred)
GEMINI_API_KEY=...        # free tier at aistudio.google.com
LANGSMITH_API_KEY=...     # free tier at smith.langchain.com
LANGSMITH_PROJECT=multimodal-agentic-analyst
```

The provider is autodetected from whichever credential is present, via
LangChain's `init_chat_model`; `GRAPH_LLM_PROVIDER` / `GRAPH_LLM_MODEL` override
it and `GRAPH_LLM_RPM` paces calls client-side.

**Free-tier budgets are the binding constraint on LLM evaluation**, measured
from live 429 payloads rather than docs:

| Provider | Free limit | Full 25-case run? |
|---|---|---|
| Gemini | **20 requests/day per model** | No — ~2 questions/day |
| Groq | **200,000 tokens/day per model** | Roughly one run per model |

The graph spends ~8 LLM calls per question (1 plan + 5 document grades +
1 synthesis + 1 verification), which is why grading dominates the bill and why
`GRADE_DOC_CHARS` bounds what each grade sends.

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
Evaluation         LangSmith datasets, 7 evaluators, experiment runner
Serving            FastAPI + Streamlit
```

Two front doors share one retrieval stack: `src/pipeline_orchestrator.py`
(original linear pipeline) and `src/graph/` (agentic layer). The agentic layer
was added without touching the ingestion, retrieval, embedding, SQL or Gemini
modules — `src/graph/adapters.py` wraps them.

`src/knowledge_graph/graph_builder.py` is the one exception, and it was a
correctness fix rather than plumbing: its extraction rules expected prose while
the corpus is pipe-delimited records, so it produced **4 triples from 451
documents**. Adding record-aware extraction took it to **704 triples across 404
nodes**.

---

## Evaluation

```python
from src.evaluation.langsmith_eval import push_dataset, run_experiment

push_dataset()                                     # 25 cases, 9 categories
run_experiment(config={"routing": "heuristic"})    # compare configurations
```

Seven evaluators — faithfulness, citation accuracy, **routing accuracy**,
**routing precision**, modality match, answer correctness, retry efficiency —
implemented as pure functions so they run in CI without a network.

Routing precision exists because accuracy alone is recall: a planner that
dispatches every specialist scores a perfect 1.000 while doing several times
the work. That was not hypothetical — it is exactly what the LLM planner did.

### Measured results

Heuristic mode, 25 cases, bge-base, no API calls:

| Metric | Score |
|---|---:|
| Citation accuracy | 1.000 |
| Routing accuracy | 0.960 |
| Routing precision | 0.800 |
| Modality match | 0.800 |
| Faithfulness | 0.803 |
| Answer correctness | 0.760 |
| Retry efficiency | 0.720 |
| Mean latency | 0.318 s |

| Category | n | Routing | Correctness | Faithfulness | Modality |
|---|---:|---:|---:|---:|---:|
| factual | 3 | 1.000 | 1.000 | 0.967 | — |
| edge | 5 | 1.000 | 1.000 | 0.399 | — |
| summary | 3 | 1.000 | 0.667 | 0.989 | — |
| sql | 3 | 1.000 | 0.667 | 1.000 | — |
| comparison | 3 | 1.000 | 0.667 | 0.978 | — |
| reasoning | 3 | 1.000 | 0.722 | 0.959 | — |
| visual_reasoning | 2 | 1.000 | — | 0.850 | 1.000 |
| ocr_extraction | 1 | 1.000 | — | 0.727 | 1.000 |
| multimodal_document | 2 | 0.500 | — | 0.486 | 0.500 |

The visual half needs `python scripts/make_demo_assets.py` first — `data/assets/`
ships empty, and with no images `modality_match` is capped at 0.200 no matter
how good retrieval is.

One number should be read carefully rather than quoted:

- **`citation_accuracy` of 1.000 is near-tautological** — citations are derived
  from findings, so it verifies plumbing, not correctness.

`edge` scoring 1.000 correctness with low faithfulness is the intended
behaviour: those are unanswerable questions, the system abstains, and an
abstention shares few tokens with the retrieved findings by construction.

---

## Tests

```bash
python -m pytest tests/ -q
```

342 tests, ~30s, fully offline. `tests/conftest.py` blanks every credential for
the session, so the suite cannot reach a live model even when `.env` holds real
keys — before that, tests calling `run_query` issued real Gemini requests and
wedged for 18 minutes inside retry backoff.

---

## Stack

Python 3.11 · LangGraph 1.0 · LangChain 1.2 · LangSmith · Groq · Gemini ·
FAISS · sentence-transformers (BGE) · CLIP · faster-whisper · FastAPI ·
Streamlit · NetworkX · SQLAlchemy
