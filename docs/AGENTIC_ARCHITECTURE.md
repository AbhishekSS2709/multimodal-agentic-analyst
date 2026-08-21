# Agentic Architecture — LangGraph Analyst Layer

The system has two front doors over the same retrieval stack:

| | v1 — `pipeline_orchestrator.py` | v2 — `src/graph/` |
|---|---|---|
| Routing | keyword match, one handler | LLM planner, multiple specialists |
| Retrieval | single pass | self-grading with query rewrite and retry |
| Answering | generate | generate, then verify groundedness |
| Safety | none | human approval before non-read-only SQL |
| Memory | stateless | checkpointed threads |
| Observability | local logs | LangSmith traces + experiments |

v1 is untouched and still serves `/api/ask`. v2 adds `/api/v2/*`.

---

## Topology

```mermaid
graph TD
    START([START]) --> supervisor[supervisor<br/>plan + select specialists]
    supervisor -->|Send| document[document<br/>corrective RAG subgraph]
    supervisor -->|Send| visual[visual<br/>CLIP + vision]
    supervisor -->|Send| analytics[analytics<br/>SQL + HITL gate]
    supervisor -->|Send| graph[graph<br/>knowledge graph multi-hop]
    document --> synthesizer[synthesizer<br/>cited answer]
    visual --> synthesizer
    analytics --> synthesizer
    graph --> synthesizer
    synthesizer --> verifier[verifier<br/>groundedness grade]
    verifier -->|retry| synthesizer
    verifier -->|done| FINISH([END])
```

### Parallel fan-out

The supervisor returns a list of `Send` objects, so every selected specialist
runs in **one superstep**, concurrently. Their writes to `findings` would
normally collide — the last write would win and the rest would vanish. They
survive because `findings` and `trace` carry `operator.add` reducers:

```python
findings: Annotated[list[Finding], operator.add]
```

Every other state key is written by exactly one node, where last-write-wins is
correct.

### Corrective RAG (document specialist)

```
retrieve --> grade --+- relevant ------------> emit
                     +- weak & budget left --> rewrite --> retrieve
                     +- weak & exhausted ----> low_confidence
```

A single-shot retriever silently returns its best bad match. This subgraph
notices the match is bad, reformulates the query, tries again, and after two
failed attempts reports low confidence rather than pretending. The cycle is
bounded by the retry budget, so it always terminates.

### Verification and the retry cycle

The verifier grades the answer for groundedness against the findings that
produced it. On failure the graph re-enters the synthesizer once — and on that
second pass the synthesizer drops to the **extractive** path, which quotes the
findings directly and is therefore grounded by construction. Without that
switch the cycle would regenerate the same ungrounded answer.

An honest "I could not find that" is not treated as a hallucination and is
never retried.

---

## Heuristic mode

No API key is required to run the graph. Every LLM call site has a
deterministic fallback, selected by `get_llm()` returning `None`:

| Node | LLM path | Heuristic fallback |
|---|---|---|
| supervisor | `with_structured_output(RoutePlan)` | `QueryRouter` keywords + `QueryAnalyzer` modality |
| grade_documents | LLM relevance grader | token overlap vs. the sub-task |
| rewrite_query | LLM rewriter | staged expansion templates |
| synthesizer | Gemini generation | extractive concatenation with citations |
| verifier | LLM-as-judge | token containment of the answer in the findings |

This is why the whole test suite runs offline. It also makes
LLM-vs-heuristic a measurable comparison rather than an assumption — both
modes are stamped into LangSmith run metadata as `llm_mode`.

---

## Human-in-the-loop

The analytics specialist inspects generated SQL before executing it.
`needs_approval()` errs toward requiring a human: empty, multi-statement, or
write-shaped SQL all pause the graph.

```python
interrupt({"type": "sql_approval", "sql": sql, ...})
```

The checkpointer persists the paused state; a human resumes it:

```bash
curl -X POST localhost:8000/api/v2/resume \
  -H 'Content-Type: application/json' \
  -d '{"thread_id": "abc123", "decision": "approve"}'
```

Rejection records `analytics:denied` in the trace and the graph continues
without the SQL evidence. Set `require_approval: false` to disable the gate.

---

## Persistence

`SqliteSaver` at `data/graph_checkpoints.db`, scoped by `thread_id`. This gives
multi-turn follow-ups, HITL resume across process restarts, and time-travel
replay of any past run:

```python
graph.get_state({"configurable": {"thread_id": "abc123"}})
```

---

## LangSmith

Set a key and tracing turns on; leave it unset and everything still runs:

```bash
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=multimodal-agentic-analyst
```

`src/evaluation/langsmith_eval.py` provides six evaluators as **pure
functions**, so they run in CI with no network and are the same functions
LangSmith calls during an experiment:

| Evaluator | Measures |
|---|---|
| `faithfulness` | answer claims supported by retrieved findings |
| `citation_accuracy` | cited sources actually trace to a finding |
| `routing_accuracy` | supervisor picked the specialists the category needs |
| `modality_match` | evidence came from the expected modality |
| `answer_correctness` | expected keywords present in the answer |
| `retry_efficiency` | penalises runs that needed corrective retries |

Routing accuracy is the notable one: the supervisor's specialist choice becomes
a **scored prediction** rather than an untested assumption.

```python
from src.evaluation.langsmith_eval import push_dataset, run_experiment
push_dataset()                                    # 25 cases, 9 categories
run_experiment(config={"routing": "heuristic"})   # compare configurations
```

---

## Layout

```
src/graph/
  state.py           AnalystState, Finding, Citation, reducers
  schemas.py         structured-output models
  llm.py             LLM factory + capability detection
  observability.py   LangSmith tracing
  adapters.py        existing components -> Documents / tools
  textutil.py        shared token-overlap scoring
  prompts.py         all prompts
  build.py           graph assembly, checkpointing, run/resume
  demo.py            streaming CLI
  nodes/
    supervisor.py  document.py  visual.py  analytics.py
    graph_specialist.py  synthesizer.py  verifier.py
src/evaluation/langsmith_eval.py
```

Nothing under `src/ingestion/`, `src/retrieval/`, `src/embedding/`,
`src/knowledge_graph/`, `src/sql_tool/`, or `src/gemini/` was modified.
`adapters.py` is the only module that knows those components' return shapes.
