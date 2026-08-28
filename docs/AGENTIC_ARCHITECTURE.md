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
dispatching every specialist scores a perfect 1.000. Every LLM planner tried
here has leaned that way — the current one averages 2.08 specialists per query
against the heuristic's 1.60, and scores 0.364 precision for 0.820 recall.
`document` is excluded from precision because the supervisor routes it as a
deliberate floor, never as a prediction.

---

## Measured results

Heuristic mode, 25 cases, bge-base, no API calls at all:

| Metric | Score |
|---|---:|
| Routing accuracy | 0.960 |
| Routing precision | 0.800 |
| Modality match | 0.800 |
| Faithfulness | 0.803 |
| Answer correctness | 0.792 |
| Retry efficiency | 0.720 |
| Mean latency | 1.53 s |

### Heuristic vs LLM planner, all 25 questions

Same 25 questions, same corpus, same indices, one run. The LLM arm is
`google/gemma-4-E4B-it` on a self-hosted llama.cpp server — chosen over the
free tiers because a 25-example run costs ~200 LLM calls, and both Gemini
(20 requests/day) and Groq (200,000 tokens/day) exhaust before it finishes.

| Metric | Heuristic | LLM (`gemma-4-E4B-it`) |
|---|---:|---:|
| Routing accuracy | **0.960** | 0.920 |
| Routing precision | **0.800** | 0.377 |
| Mean latency | **1.53 s** | 67.98 s |
| Answer correctness | 0.792 | **1.000** |
| Modality match | 0.800 | **1.000** |
| Retry efficiency | **0.720** | 0.680 |

The split is between *routing* and *answering*: the heuristic dispatches more
precisely, the LLM composes better answers once dispatched.

**The heuristic still routes more precisely.** The LLM averages 2.52
specialists per query against 1.60, which is where its precision goes. Its one
remaining routing weakness is `comparison`, at 0.333 — it reaches for
`analytics` where the knowledge graph is wanted, and adds `visual` to questions
with no visual content. Every other category routes at 1.000, and it beats the
heuristic on `multimodal_document` (1.000 vs 0.500).

**The LLM answers better, and the gap is entirely abstractive work.**
`synthesize_heuristic` is extractive by design — it stitches the top findings
together with citation markers. On `summary`, `comparison` and `reasoning`
questions that surfaces the right evidence and never turns it into an answer,
which is why the heuristic scores 0.000 on "What are the key takeaways from the
last quarter?" while answering it with the correct figures.

**Read the 1.000 with care.** `answer_correctness` is keyword containment
scored on the 16 of 25 examples that carry an expectation, not a judgement of
whether an answer is right. It is a ceiling on a proxy, not a solved problem.

**What is and is not attributable to the fixes.** Enforcing the `document`
floor and correcting the supervisor prompt moved routing accuracy 0.820 ->
0.920 and took plans missing the floor from 6 of 25 to 0 — those are direct,
and were confirmed on a 22-example partial run before this one. Answer
correctness moved 0.875 -> 1.000 in the same change, but the two questions that
had failed were routed identically in both runs, so the plan is not what
changed for them. A 7.5B model has real run-to-run variance and this is a
single run: treat the routing numbers as measured and the correctness ceiling
as partly luck.

**Two caveats.**

*Faithfulness is not comparable across the arms and is left out of the table.*
It is lexical overlap between the answer's content words and the findings text,
and the heuristic synthesizer quotes findings verbatim, so it scores high by
construction (0.803 against the LLM's 0.666). That gap measures abstraction,
not hallucination — and abstraction is what wins the correctness column.

*Latency is not a fair reading of the model.* The 67.98 s mean was measured on
a machine at 93% memory use and swapping heavily; an earlier run of the same
model on the same server averaged 43 s. Neither number is a benchmark of the
server.

*Three of 25 gradings fell back*, all because the model answered `"N/A"` when
asked whether a document was relevant. That is a non-answer rather than a
grade, so the coercion refuses it and the node grades heuristically, which is
the safe outcome. Planning, synthesis and verification never fell back.

Both arms' raw per-example scores are in
[`eval_results_25.json`](eval_results_25.json).

### A second model: Qwen3.5-9B

Run again with the graph on `Qwen/Qwen3.5-9B` (vLLM, port 8005) to test whether
the remaining routing weakness was the model or the design. The SQL writer is
held at `gemma-4-E4B-it` in both runs, so the graph's LLM is the only variable.

The server dropped out for five consecutive examples (17-21, all
`Connection error`), so those rows fell back to the heuristic planner. The
table below is the **20 examples all three arms answered cleanly**.

| Metric | Heuristic | `gemma-4-E4B` | `Qwen3.5-9B` |
|---|---:|---:|---:|
| Routing accuracy | **0.950** | 0.900 | 0.925 |
| Routing precision | **0.846** | 0.383 | 0.656 |
| Answer correctness | 0.778 | **1.000** | **1.000** |
| Modality match | 0.750 | **1.000** | 0.750 |
| Faithfulness | **0.904** | 0.754 | 0.659 |
| Mean latency | **1.48 s** | 65.60 s | 52.31 s |

**The routing weakness was the model, not the design.** `comparison` routing
went 0.333 -> 0.667, and the over-dispatch that cost gemma its precision
largely disappeared: 1.80 specialists per query against 2.52, close to the
heuristic's 1.60, with no `visual` bolted onto text-only comparisons and no
plan missing the `document` floor. Precision nearly doubled, 0.383 -> 0.656.

**It did not fix everything.** `modality_match` fell to 0.750, where gemma was
perfect — Qwen under-routes `visual`, which is the mirror of gemma's habit of
over-routing it. Neither model reaches the heuristic on precision.

**A reasoning model needs handling.** Qwen puts its chain of thought in
`content`, so answers arrived as *"Thinking Process: 1. **Analyze the
Request:**..."* and were scored as if that were the answer. It also made every
call generate thousands of extra tokens — 3 minutes per question, against
gemma's 68 seconds. `GRAPH_LLM_EXTRA_BODY` passes
`{"chat_template_kwargs": {"enable_thinking": false}}` to vLLM, which fixed
both. The graph's own nodes were never the problem: they use
`with_structured_output`, and Qwen's tool calling worked first time. It was the
v1 SQL agent's plain-text prompt that had nothing to constrain the reply, which
is why the SQL writer stays on gemma.

Raw scores: [`eval_results_25.json`](eval_results_25.json) (gemma) and
[`eval_results_qwen.json`](eval_results_qwen.json).

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
