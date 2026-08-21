# Multimodal Enterprise Research Analyst — LangGraph Agentic Layer

**Date:** 2026-08-21
**Approach:** Supervisor-led multi-agent graph over the existing retrieval stack (wrap, don't rewrite)
**Status:** Approved
**Stack added:** LangGraph 1.0.7, LangChain 1.2.8, LangSmith 0.6.8, langchain-google-genai 4.2.0

---

## 1. Problem Statement

The existing system is a **linear pipeline with a regex router**. `src/agents/query_router.py`
classifies queries by keyword matching and dispatches to one of a fixed set of handlers.
It cannot decompose a compound question, cannot use two retrieval backends for one
question, cannot tell when its own retrieval failed, and cannot recover when it does.

Build a **supervisor-orchestrated multi-agent analyst** on top of the existing
components that:

1. **Plans** — decomposes a question into sub-tasks and selects which specialists to run.
2. **Delegates in parallel** — fans out to four heterogeneous specialist subgraphs.
3. **Self-corrects** — the document specialist grades its own retrievals, then rewrites
   the query and retries when they are weak.
4. **Verifies** — grades the final answer for groundedness and relevance before returning.
5. **Pauses for humans** — interrupts for approval before destructive or expensive operations.
6. **Remembers** — checkpointed threads survive process restarts.
7. **Is measured** — every node traced in LangSmith; evaluation runs as LangSmith experiments.

### Non-goals

- Rewriting loaders, chunkers, embeddings, FAISS stores, or retrievers in LangChain primitives.
- Replacing `src/pipeline_orchestrator.py`. It keeps working; the graph is a parallel front door.
- Any paid API. Gemini free tier plus LangSmith free tier only.

---

## 2. Architecture

```
                        +--------------+
   question ----------->|  supervisor  |  plan + select specialists (structured output)
                        +------+-------+
                               | Send() fan-out (parallel superstep)
        +--------------+-------+--------+------------------+
        v              v                v                  v
 +------------+ +------------+  +--------------+   +------------+
 |  document  | |   visual   |  |  analytics   |   |   graph    |
 |  (CRAG     | |  CLIP +    |  |  SQL + chart |   |  KG multi- |
 |  subgraph) | |  Gemini    |  |  + HITL gate |   |  hop       |
 +-----+------+ +-----+------+  +------+-------+   +-----+------+
       |              |                |                 |
       +--------------+----------------+-----------------+
                               |  findings (operator.add reducer)
                               v
                        +--------------+
                        | synthesizer  |  merge findings -> cited answer
                        +------+-------+
                               v
                        +--------------+
                        |   verifier   |  groundedness + relevance grade
                        +------+-------+
                     pass -----+----- fail (retry <= 1) --> synthesizer
                               v
                             answer
```

### 2.1 New package layout

```
src/graph/
  state.py           AnalystState TypedDict, Finding/Citation models, reducers
  schemas.py         Pydantic models for structured LLM output
  llm.py             LLM factory + capability detection (key present? -> LLM : heuristic)
  prompts.py         All prompt templates in one place
  adapters.py        Existing components exposed as BaseRetriever / @tool
  observability.py   LangSmith tracing setup, run metadata helpers
  nodes/
    supervisor.py       plan + route
    document.py         corrective-RAG subgraph
    visual.py           CLIP + Gemini vision specialist
    analytics.py        SQL specialist + approval gate
    graph_specialist.py knowledge-graph / multi-hop specialist
    synthesizer.py      cited-answer composition
    verifier.py         groundedness + relevance grading
  build.py           StateGraph assembly, compile(checkpointer=...)
  demo.py            CLI that streams node-by-node execution
src/evaluation/
  langsmith_eval.py  dataset push, evaluators, experiment runner
```

### 2.2 State schema

```python
class AnalystState(TypedDict):
    messages:      Annotated[list[AnyMessage], add_messages]
    question:      str
    plan:          list[SubTask]
    specialists:   list[str]
    findings:      Annotated[list[Finding], operator.add]   # parallel-safe
    answer:        str
    citations:     list[Citation]
    verification:  dict
    retry_count:   int
    approval:      str | None
    trace:         Annotated[list[str], operator.add]       # node breadcrumbs
```

`findings` and `trace` use `operator.add` so parallel specialist writes merge instead of
colliding. Everything else is last-write-wins, which is safe because exactly one node
writes each of those keys.

### 2.3 Specialist subgraphs

| Specialist | Wraps | Emits |
|---|---|---|
| `document` | `HybridRetriever` + `CrossEncoderReranker` + `SourceAttributor` | text findings + citations |
| `visual` | `CLIPEngine` + `VisualStore` + `AnswerGenerator` (vision) | image findings + asset ids |
| `analytics` | `SQLAgent` + `SQLAnalyticsPipeline` + `ChartGenerator` | table/chart findings |
| `graph` | `GraphRetriever` + `MultiHopRetriever` | entity-path findings |

Each is an independent subgraph with its own state, invoked as a node. Each returns a
list of `Finding` so the synthesizer treats all four uniformly.

### 2.4 Corrective-RAG subgraph (document specialist)

```
retrieve --> grade_documents --+- relevant ------------> emit_finding
                               +- weak & retries left -> rewrite_query --> retrieve
                               +- weak & exhausted ----> emit_low_confidence
```

`grade_documents` scores each retrieved chunk for relevance to the sub-task.
Retry budget is 2. This is a genuine cycle in the graph, guarded by a recursion limit.

---

## 3. Graceful degradation (no-API-key mode)

No keys are set in this environment, and free-tier Gemini is rate-limited (10 RPM).
**Every LLM-dependent node has a deterministic heuristic fallback**, selected at
runtime by `llm.get_llm()` returning `None`:

| Node | LLM path | Heuristic fallback |
|---|---|---|
| supervisor | `with_structured_output(RoutePlan)` | existing `QueryRouter` keywords + `QueryAnalyzer` modality |
| grade_documents | LLM relevance grader | token-overlap score vs. sub-task (reuses `eval_pipeline` tokenizer) |
| rewrite_query | LLM query rewriter | entities from `multi_hop.extract_entities` + template |
| synthesizer | Gemini `AnswerGenerator` | extractive concatenation of top findings with citations |
| verifier | LLM groundedness judge | n-gram containment of answer claims in finding text |

This is not a testing convenience. It is the reason the whole graph is unit-testable
offline, and it turns LLM-vs-heuristic into a **measurable LangSmith experiment** rather
than an assumption.

---

## 4. Human-in-the-loop

`interrupt()` fires in the analytics specialist before executing any SQL that is not a
plain `SELECT`, and before any operation flagged expensive. The graph pauses, the
checkpointer persists state, and the caller resumes with
`graph.invoke(Command(resume="approve"), config)`.

Controlled per-invocation by `require_approval` in the graph config, default `True`.

---

## 5. Persistence

`SqliteSaver` (`langgraph-checkpoint-sqlite`) at `data/graph_checkpoints.db`.
`thread_id` scopes a conversation. This enables multi-turn follow-ups, HITL resume, and
time-travel replay of any past run for debugging.

---

## 6. LangSmith observability and evaluation

**Tracing.** Enabled by env (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`,
`LANGSMITH_PROJECT`). Absent keys means tracing is silently off and the graph is unaffected.

**Evaluation** (`src/evaluation/langsmith_eval.py`):

- Push `src/evaluation/test_cases.py` (including multimodal cases) as a LangSmith dataset.
- Custom evaluators:
  - `faithfulness` — answer claims grounded in retrieved findings
  - `citation_accuracy` — cited sources actually support the claim
  - `routing_accuracy` — did the supervisor pick the expected specialists
  - `modality_match` — did the answer use the expected modality
  - `latency` / `retry_count` — operational metrics
- Experiment runner compares configurations: LLM routing vs. keyword routing,
  corrective retry on vs. off, rerank on vs. off.
- `FeedbackStore` writes mirror to the LangSmith feedback API, closing the loop from
  production thumbs up/down back into the dataset.

---

## 7. Surfaces

- `POST /api/v2/query` — run the graph; accepts `thread_id`, `require_approval`.
- `POST /api/v2/resume` — resume an interrupted thread with an approval decision.
- `GET  /api/v2/graph` — render the compiled graph topology (Mermaid) for the demo/README.
- `python -m src.graph.demo "question"` — CLI that streams node-by-node execution.

---

## 8. Testing strategy

`unittest`, no network, matching existing `tests/` conventions.

- State reducers merge parallel writes correctly.
- Supervisor selects expected specialists per query class (heuristic path).
- Corrective subgraph loops on weak docs and terminates at the retry budget.
- Verifier flags an ungrounded answer.
- HITL raises an interrupt and resumes to completion.
- Checkpointer round-trips a thread across two invocations.
- Full graph end-to-end on a fixture corpus, heuristic mode.
- Adapters conform to `BaseRetriever` and tool schemas.

---

## 9. Phasing

| Phase | Deliverable |
|---|---|
| 1 | State, schemas, LLM factory, adapters, observability |
| 2 | Supervisor + synthesizer + verifier; minimal graph runs end-to-end |
| 3 | Four specialist subgraphs, including the corrective-RAG loop |
| 4 | Checkpointer + HITL interrupt/resume |
| 5 | LangSmith datasets, evaluators, experiment runner |
| 6 | API v2 endpoints, CLI demo, README and architecture docs |

---

## 10. Resume framing

Bullets this project supports, each backed by code:

- Multi-agent supervisor orchestrating four heterogeneous retrieval backends with
  parallel fan-out and state reducers (LangGraph).
- Corrective RAG: self-grading retrieval with query rewriting and bounded retry cycles.
- Human-in-the-loop approval gates over durable checkpointed state.
- LLM-as-judge evaluation suite run as versioned LangSmith experiments, with
  production feedback wired back into datasets.
- Framework integration into a pre-existing 60-module system without rewriting it.
