# Agentic Architecture — LangGraph Analyst Layer

The system has two front doors over the same retrieval stack:

| | v1 — `pipeline_orchestrator.py` | v2 — `src/graph/` |
|---|---|---|
| Routing | keyword match, one handler | multi-specialist planner; keyword+cues by default, LLM opt-in |
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
| supervisor | `with_structured_output(RoutePlan)` | `QueryRouter` keywords + modality, plus intent cues when classifier confidence is low |
| grade_documents | LLM relevance grader (bounded to `GRADE_DOC_CHARS`) | token overlap vs. the sub-task |
| rewrite_query | LLM rewriter | staged expansion templates |
| synthesizer | Gemini generation | extractive concatenation with citations |
| verifier | LLM-as-judge | token containment of the answer in the findings |

This is why the whole test suite runs offline. It also makes
LLM-vs-heuristic a measurable comparison rather than an assumption — both
modes are stamped into LangSmith run metadata as `llm_mode`.

One caveat worth stating plainly: this holds for `src/graph/`. Text-to-SQL
inherits the v1 pipeline's `SQLAnalyticsPipeline`, which builds its own
`GeminiClient` and ignores the provider configuration entirely, so the
analytics specialist is *not* key-free. That was found the hard way — a run
labelled "heuristic" spent an entire day's Gemini quota.

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

`src/evaluation/langsmith_eval.py` provides seven evaluators as **pure
functions**, so they run in CI with no network and are the same functions
LangSmith calls during an experiment:

| Evaluator | Measures |
|---|---|
| `faithfulness` | answer claims supported by retrieved findings |
| `citation_accuracy` | cited sources actually trace to a finding |
| `routing_accuracy` | **recall** — did it pick the specialists the category needs |
| `routing_precision` | **precision** — were the specialists it picked needed |
| `modality_match` | evidence came from the expected modality |
| `answer_correctness` | expected keywords present in the answer |
| `retry_efficiency` | penalises runs that needed corrective retries |

Routing is the notable one: the supervisor's specialist choice becomes a
**scored prediction** rather than an untested assumption.

`routing_precision` had to be added because accuracy alone is recall, so
dispatching every specialist scores a perfect 1.000 — which an early LLM
planner did, at 2.5 specialists per query. `document` is excluded from
precision because the supervisor routes it as a deliberate floor, never as a
prediction.

---

## Measured results

Heuristic mode, 25 cases, bge-base, no API calls:

| Metric | Score |
|---|---:|
| Routing accuracy | 0.960 |
| Routing precision | 0.800 |
| Modality match | 0.800 |
| Faithfulness | 0.803 |
| Answer correctness | 0.760 |
| Retry efficiency | 0.720 |
| Mean latency | 0.480 s |

### Heuristic vs LLM planner, all 25 questions

Same 25 questions, same corpus, same indices, one run. The LLM arm is
`gpt-oss-safeguard-20b` on Groq, rate-limited to 20 rpm.

| Metric | Heuristic | LLM (`gpt-oss-safeguard-20b`) |
|---|---:|---:|
| Routing accuracy | **0.960** | 0.680 |
| Routing precision | **0.800** | 0.500 |
| Answer correctness | **0.760** | 0.667 |
| Retry efficiency | 0.720 | 0.720 |
| Modality match | 0.800 | **1.000** |
| Mean latency | **0.48 s** | 32.31 s |

The deterministic planner wins on routing and correctness at 1/67th the
latency; the LLM wins on modality match, which it gets right every time.

The failure mode is consistent and it is a *recall* failure, not the
over-dispatch an LLM planner is usually accused of: it **drops the `document`
floor** on 5 of 25 questions, routing `['analytics']` alone on three SQL
questions, `['analytics']` on a comparative one, and `['visual']` alone on the
OCR case. It averages 1.28 specialists per query against the heuristic's 1.60.
`plan_heuristic` guarantees `document` is always present, so it cannot make
that mistake.

**Two caveats, both against the headline.**

*Faithfulness is not comparable across the arms and is left out of the table
above.* It is measured as lexical overlap between the answer's content words
and the findings text, and `synthesize_heuristic` is **extractive** — it quotes
findings verbatim, so it scores near 1.000 by construction (0.803 measured).
An abstractive LLM answer paraphrases, so it scores 0.339 for reasons that have
nothing to do with hallucination. Reading that gap as a groundedness result
would be wrong.

*Three of the 25 LLM plans fell back to the heuristic* — Groq rejected the tool
call on `Show monthly order trend`, the empty question and the SQL-injection
string (`tool_use_failed`, plus one TPM 429). Those three rows are therefore
partly heuristic. Excluding them, the LLM arm scores **0.636** routing accuracy
and **0.455** precision against the heuristic's 0.955 / 0.786 on the same 22 —
so the contaminated rows *helped* the LLM, and the n=25 numbers above are the
conservative reading.

Both arms' raw per-example scores are in
[`eval_results_25.json`](eval_results_25.json).

### What moved the numbers

| Change | Effect |
|---|---|
| Record-aware KG extraction | 4 → **704 triples**; `reasoning` correctness 0.278 → 0.500 |
| Graph specialist reads the KG | it previously never did — see below |
| Confidence-gated intent cues | routing recall 0.720 → **0.960** |
| Connecting the multimodal path | `modality_match` 0.200 → **0.800** |
| Folding `-ies` onto `-y` in the stemmer | `reasoning` correctness 0.389 → **0.722** |
| Rebuilding rather than appending the index | removed a silent 2x duplication of every chunk |

Both fixes share a shape worth naming: a component with green unit tests that
was never actually reachable. The knowledge graph was read by nothing; CLIP was
broken in four places at once and its tests passed because they mocked the
model. Unit tests proved the logic while the integration was dead, and no
amount of tuning could move the scores until the wiring was fixed.

The knowledge-graph fix is the instructive one. Growing the graph from 4 to 704
triples changed **no score at all**, because two separate bugs meant nothing
read it: multi-hop retrieval (which runs on the *vector* index) returned first,
and the fallback called `augment_retrieval(question, [])`, which returns its
second argument unchanged when empty. The graph had never contributed to an
answer. Only after wiring it in did the triples show up in the scores.

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
