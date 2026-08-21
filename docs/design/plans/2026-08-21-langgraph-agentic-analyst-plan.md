# LangGraph Agentic Analyst Implementation Plan


**Goal:** Add a supervisor-orchestrated multi-agent LangGraph layer on top of the existing RAG stack, with corrective retrieval, human-in-the-loop gates, durable threads, and LangSmith evaluation.

**Architecture:** A new `src/graph/` package wraps existing components as LangChain retrievers and tools without modifying them. A `StateGraph` supervisor plans and fans out in parallel to four specialist subgraphs; findings merge through `operator.add` reducers into a synthesizer and a verifier. Every LLM-dependent node has a deterministic heuristic fallback so the whole graph runs and tests offline.

**Tech Stack:** LangGraph 1.0.7, langgraph-checkpoint-sqlite, LangChain 1.2.8, langchain-core 1.2.16, langchain-google-genai 4.2.0, LangSmith 0.6.8, Pydantic 2.12, FastAPI, unittest.

**Spec:** `docs/design/specs/2026-08-21-langgraph-agentic-analyst-design.md`

## Global Constraints

- **Do not modify** existing modules under `src/ingestion/`, `src/retrieval/`, `src/embedding/`, `src/knowledge_graph/`, `src/sql_tool/`, `src/gemini/`. Wrap them. The only pre-existing files this plan may touch are `src/api/main.py`, `src/api/models.py`, `requirements.txt`, and `README.md`.
- **`src/pipeline_orchestrator.py` must keep working unchanged.** The graph is a parallel front door, not a replacement.
- **No network in tests.** Every test runs with no API keys set. Tests must pass offline.
- **No paid APIs.** Gemini free tier and LangSmith free tier only.
- **Every LLM call site needs a heuristic fallback** selected by `get_llm()` returning `None`.
- Python 3.11. Tests use `unittest`, matching the existing `tests/` style (no pytest fixtures, no external deps).
- Import style: modules add project root to `sys.path` then import from `config.settings`, matching existing files.
- Run tests with `python -m pytest tests/<file> -v` from the project root.

---

### Task 1: State schema, findings model, and reducers

**Files:**
- Create: `src/graph/__init__.py`, `src/graph/state.py`, `src/graph/schemas.py`
- Test: `tests/test_graph_state.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `AnalystState` (TypedDict) with keys `messages, question, plan, specialists, findings, answer, citations, verification, retry_count, approval, trace`
  - `Finding` (pydantic BaseModel): `specialist: str`, `content: str`, `score: float`, `source: str`, `doc_id: str = ""`, `modality: str = "text"`, `metadata: dict = {}`
  - `Citation` (pydantic BaseModel): `source: str`, `doc_id: str`, `snippet: str`, `specialist: str`
  - `SubTask` (pydantic BaseModel): `description: str`, `specialist: str`
  - `RoutePlan` (pydantic BaseModel): `subtasks: list[SubTask]`, `rationale: str`
  - `GradeDocuments` (pydantic BaseModel): `relevant: bool`, `score: float`, `reason: str`
  - `Verification` (pydantic BaseModel): `grounded: bool`, `relevant: bool`, `score: float`, `reason: str`
  - `SPECIALISTS: tuple[str, ...] = ("document", "visual", "analytics", "graph")`
  - `new_state(question: str) -> AnalystState`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for LangGraph analyst state and reducers — TDD, no external deps."""

import operator
import unittest


class TestGraphState(unittest.TestCase):

    def test_new_state_initialises_all_keys(self):
        from src.graph.state import AnalystState, new_state
        s = new_state("what were Q3 sales?")
        self.assertEqual(s["question"], "what were Q3 sales?")
        for key in ("messages", "plan", "specialists", "findings",
                    "answer", "citations", "verification", "retry_count",
                    "approval", "trace"):
            self.assertIn(key, s)
        self.assertEqual(s["findings"], [])
        self.assertEqual(s["retry_count"], 0)
        self.assertIsNone(s["approval"])

    def test_findings_reducer_merges_parallel_writes(self):
        """Two specialists writing concurrently must both survive."""
        from src.graph.state import Finding
        a = [Finding(specialist="document", content="a", score=0.9, source="x.pdf")]
        b = [Finding(specialist="visual", content="b", score=0.8, source="y.png")]
        merged = operator.add(a, b)
        self.assertEqual(len(merged), 2)
        self.assertEqual({f.specialist for f in merged}, {"document", "visual"})

    def test_finding_defaults(self):
        from src.graph.state import Finding
        f = Finding(specialist="document", content="c", score=0.5, source="s")
        self.assertEqual(f.modality, "text")
        self.assertEqual(f.doc_id, "")
        self.assertEqual(f.metadata, {})

    def test_route_plan_round_trip(self):
        from src.graph.schemas import RoutePlan, SubTask
        p = RoutePlan(subtasks=[SubTask(description="find sales", specialist="analytics")],
                      rationale="numeric question")
        self.assertEqual(p.subtasks[0].specialist, "analytics")

    def test_specialists_constant(self):
        from src.graph.state import SPECIALISTS
        self.assertEqual(set(SPECIALISTS), {"document", "visual", "analytics", "graph"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph'`

- [ ] **Step 3: Write minimal implementation**

`src/graph/schemas.py` holds the pydantic models for structured LLM output (`SubTask`, `RoutePlan`, `GradeDocuments`, `Verification`, `RewrittenQuery`). `src/graph/state.py` holds `Finding`, `Citation`, `SPECIALISTS`, `AnalystState`, and `new_state`, re-exporting `SubTask`/`RoutePlan` from `schemas` so callers have one import site.

```python
# src/graph/state.py
from __future__ import annotations
import operator
from typing import Annotated, Any, Optional, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from src.graph.schemas import SubTask, RoutePlan  # re-export

SPECIALISTS: tuple[str, ...] = ("document", "visual", "analytics", "graph")

class Finding(BaseModel):
    specialist: str
    content: str
    score: float = 0.0
    source: str = ""
    doc_id: str = ""
    modality: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)

class Citation(BaseModel):
    source: str
    doc_id: str = ""
    snippet: str = ""
    specialist: str = ""

class AnalystState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    plan: list[SubTask]
    specialists: list[str]
    findings: Annotated[list[Finding], operator.add]
    answer: str
    citations: list[Citation]
    verification: dict[str, Any]
    retry_count: int
    approval: Optional[str]
    trace: Annotated[list[str], operator.add]

def new_state(question: str) -> AnalystState:
    return AnalystState(
        messages=[], question=question, plan=[], specialists=[], findings=[],
        answer="", citations=[], verification={}, retry_count=0,
        approval=None, trace=[],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_state.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/ tests/test_graph_state.py
git commit -m "feat(graph): analyst state schema, findings model, parallel reducers"
```

---

### Task 2: LLM factory and LangSmith observability

**Files:**
- Create: `src/graph/llm.py`, `src/graph/observability.py`
- Modify: `requirements.txt` (add the LangChain/LangGraph/LangSmith block)
- Test: `tests/test_graph_llm.py`

**Interfaces:**
- Consumes: `config.settings.GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_TEMPERATURE`.
- Produces:
  - `get_llm(temperature: float | None = None, model: str | None = None) -> BaseChatModel | None` — returns `None` when no key is configured. Result cached per (model, temperature).
  - `get_structured_llm(schema: type[BaseModel]) -> Runnable | None` — `get_llm().with_structured_output(schema)`, or `None`.
  - `llm_available() -> bool`
  - `configure_tracing() -> bool` — sets `LANGSMITH_TRACING`/`LANGCHAIN_TRACING_V2` env when a key exists; returns whether tracing is on. Never raises.
  - `tracing_enabled() -> bool`
  - `run_metadata(**kwargs) -> dict` — metadata dict attached to graph runs.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the LLM factory and LangSmith wiring — offline, no keys."""

import os
import unittest


class TestLLMFactory(unittest.TestCase):

    def setUp(self):
        # Guarantee the no-key path regardless of the developer's environment.
        self._saved = {k: os.environ.pop(k, None)
                       for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        import src.graph.llm as llm_mod
        llm_mod.get_llm.cache_clear()
        self.llm_mod = llm_mod

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
        self.llm_mod.get_llm.cache_clear()

    def test_get_llm_returns_none_without_key(self):
        self.assertIsNone(self.llm_mod.get_llm())

    def test_llm_available_false_without_key(self):
        self.assertFalse(self.llm_mod.llm_available())

    def test_get_structured_llm_returns_none_without_key(self):
        from src.graph.schemas import RoutePlan
        self.assertIsNone(self.llm_mod.get_structured_llm(RoutePlan))


class TestObservability(unittest.TestCase):

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None)
                       for k in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
                                 "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")}

    def tearDown(self):
        for k in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            os.environ.pop(k, None)
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_configure_tracing_off_without_key(self):
        from src.graph.observability import configure_tracing, tracing_enabled
        self.assertFalse(configure_tracing())
        self.assertFalse(tracing_enabled())

    def test_configure_tracing_never_raises(self):
        from src.graph.observability import configure_tracing
        configure_tracing()  # must not raise even with a bogus key
        os.environ["LANGSMITH_API_KEY"] = "lsv2_fake_key_for_test"
        self.assertTrue(configure_tracing())
        self.assertEqual(os.environ.get("LANGSMITH_TRACING"), "true")

    def test_run_metadata_includes_flags(self):
        from src.graph.observability import run_metadata
        md = run_metadata(specialist="document")
        self.assertEqual(md["specialist"], "document")
        self.assertIn("llm_mode", md)
        self.assertIn(md["llm_mode"], ("llm", "heuristic"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_llm.py -v`
Expected: FAIL — `No module named 'src.graph.llm'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/graph/llm.py
from __future__ import annotations
import logging, os
from functools import lru_cache
from typing import Any, Optional
from pydantic import BaseModel
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE

logger = logging.getLogger(__name__)

def _api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            or GEMINI_API_KEY or "").strip()

@lru_cache(maxsize=8)
def get_llm(temperature: Optional[float] = None,
            model: Optional[str] = None) -> Optional[Any]:
    """Chat model, or None when no key is configured (heuristic mode)."""
    key = _api_key()
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or GEMINI_MODEL,
            temperature=GEMINI_TEMPERATURE if temperature is None else temperature,
            google_api_key=key,
        )
    except Exception as exc:                      # missing dep, bad key shape
        logger.warning("LLM unavailable, falling back to heuristics: %s", exc)
        return None

def llm_available() -> bool:
    return get_llm() is not None

def get_structured_llm(schema: type[BaseModel], temperature: Optional[float] = None):
    llm = get_llm(temperature)
    if llm is None:
        return None
    try:
        return llm.with_structured_output(schema)
    except Exception as exc:
        logger.warning("Structured output unavailable for %s: %s", schema, exc)
        return None
```

`observability.py` sets both the new (`LANGSMITH_*`) and legacy (`LANGCHAIN_*`) env names so tracing works across versions, and `run_metadata` stamps `llm_mode` so LangSmith runs are filterable by LLM-vs-heuristic.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_llm.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/llm.py src/graph/observability.py tests/test_graph_llm.py requirements.txt
git commit -m "feat(graph): LLM factory with heuristic fallback and LangSmith tracing setup"
```

---

### Task 3: Adapters wrapping existing components

**Files:**
- Create: `src/graph/adapters.py`
- Test: `tests/test_graph_adapters.py`

**Interfaces:**
- Consumes: `EnterpriseRAGOrchestrator` private accessors (`_get_hybrid_retriever`, `_get_retriever`, `_get_multimodal_retriever`, `_get_sql_pipeline`, `_get_graph_retriever`, `_get_multi_hop_retriever`, `_get_answer_generator`); `src.graph.state.Finding`.
- Produces:
  - `ChunkRetriever(BaseRetriever)` — LangChain retriever over any object exposing `retrieve(query, top_k)`. Implements `_get_relevant_documents(self, query, *, run_manager) -> list[Document]`. Constructor: `ChunkRetriever(backend, top_k: int = 5)`.
  - `to_documents(results: list) -> list[Document]` — normalises chunk tuples, chunk objects, and dicts into `Document(page_content, metadata)` with `metadata` keys `source`, `doc_id`, `score`, `modality`.
  - `documents_to_findings(docs: list[Document], specialist: str) -> list[Finding]`
  - `build_tools(orchestrator) -> list[BaseTool]` — `@tool`-decorated `search_documents`, `search_visuals`, `run_analytics`, `search_knowledge_graph`, each returning a JSON-serialisable dict.
  - `LazyComponents` — thin accessor holding one orchestrator, exposing `.hybrid`, `.multimodal`, `.sql`, `.graph`, `.multi_hop`, `.generator`; each returns `None` instead of raising when the underlying component is unavailable (no index built, no key).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for LangChain adapters over existing RAG components."""

import unittest


class _FakeBackend:
    """Minimal stand-in for HybridRetriever/Retriever."""
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def retrieve(self, query, top_k=5, **kw):
        self.calls.append((query, top_k))
        return self.rows[:top_k]


class _FakeChunk:
    def __init__(self, text, doc_id, metadata=None):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = doc_id + "_0"
        self.metadata = metadata or {"source": doc_id}


class TestToDocuments(unittest.TestCase):

    def test_normalises_dict_results(self):
        from src.graph.adapters import to_documents
        docs = to_documents([
            {"text": "hello", "doc_id": "d1", "source": "a.pdf", "score": 0.7},
        ])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "hello")
        self.assertEqual(docs[0].metadata["source"], "a.pdf")
        self.assertAlmostEqual(docs[0].metadata["score"], 0.7)

    def test_normalises_chunk_score_tuples(self):
        from src.graph.adapters import to_documents
        docs = to_documents([(_FakeChunk("body", "d2"), 0.42)])
        self.assertEqual(docs[0].page_content, "body")
        self.assertEqual(docs[0].metadata["doc_id"], "d2")
        self.assertAlmostEqual(docs[0].metadata["score"], 0.42)

    def test_normalises_hybrid_four_tuples(self):
        """HybridRetriever returns (chunk, fused, vec, bm25)."""
        from src.graph.adapters import to_documents
        docs = to_documents([(_FakeChunk("h", "d3"), 0.9, 0.6, 0.3)])
        self.assertEqual(docs[0].page_content, "h")
        self.assertAlmostEqual(docs[0].metadata["score"], 0.9)

    def test_empty_input(self):
        from src.graph.adapters import to_documents
        self.assertEqual(to_documents([]), [])


class TestChunkRetriever(unittest.TestCase):

    def test_conforms_to_base_retriever(self):
        from langchain_core.retrievers import BaseRetriever
        from src.graph.adapters import ChunkRetriever
        r = ChunkRetriever(_FakeBackend([{"text": "t", "doc_id": "d", "score": 1.0}]))
        self.assertIsInstance(r, BaseRetriever)

    def test_invoke_returns_documents(self):
        from src.graph.adapters import ChunkRetriever
        backend = _FakeBackend([{"text": "t", "doc_id": "d", "score": 1.0}])
        docs = ChunkRetriever(backend, top_k=3).invoke("q")
        self.assertEqual(docs[0].page_content, "t")
        self.assertEqual(backend.calls, [("q", 3)])


class TestDocumentsToFindings(unittest.TestCase):

    def test_maps_metadata_onto_findings(self):
        from langchain_core.documents import Document
        from src.graph.adapters import documents_to_findings
        docs = [Document(page_content="body",
                         metadata={"source": "a.pdf", "doc_id": "d1",
                                   "score": 0.5, "modality": "text"})]
        findings = documents_to_findings(docs, "document")
        self.assertEqual(findings[0].specialist, "document")
        self.assertEqual(findings[0].source, "a.pdf")
        self.assertAlmostEqual(findings[0].score, 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_adapters.py -v`
Expected: FAIL — `No module named 'src.graph.adapters'`

- [ ] **Step 3: Write minimal implementation**

`to_documents` dispatches on result shape: `dict` (multimodal/multi-hop output), 2-tuple `(chunk, score)` (`Retriever`), 4-tuple `(chunk, fused, vec, bm25)` (`HybridRetriever`), or a bare chunk object. `ChunkRetriever` subclasses `BaseRetriever` with fields `backend: Any` and `top_k: int = 5` and implements `_get_relevant_documents`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_adapters.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/adapters.py tests/test_graph_adapters.py
git commit -m "feat(graph): LangChain adapters wrapping existing retrievers as Documents and tools"
```

---

### Task 4: Supervisor node

**Files:**
- Create: `src/graph/prompts.py`, `src/graph/nodes/__init__.py`, `src/graph/nodes/supervisor.py`
- Test: `tests/test_graph_supervisor.py`

**Interfaces:**
- Consumes: `get_structured_llm`, `RoutePlan`, `SubTask`, `SPECIALISTS`, `AnalystState`; existing `QueryRouter.classify` and `QueryAnalyzer.analyze`.
- Produces:
  - `plan_heuristic(question: str) -> RoutePlan` — keyword/modality routing, always returns at least one subtask (`document` is the floor).
  - `supervisor_node(state: AnalystState) -> dict` — returns `{"plan": [...], "specialists": [...], "trace": ["supervisor:<mode>"]}`.
  - `SUPERVISOR_PROMPT` in `prompts.py`.

Routing rules for `plan_heuristic` (mirrors `QueryRouter` categories so behaviour is comparable in LangSmith experiments):

| Signal | Specialist |
|---|---|
| `QueryRouter` category in `{sql, analytics}` | `analytics` |
| `QueryRouter` category in `{reasoning, comparison}` | `graph` |
| `QueryAnalyzer` modality in `{visual, both}` | `visual` |
| always | `document` |

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the supervisor planning node (heuristic mode)."""

import unittest


class TestPlanHeuristic(unittest.TestCase):

    def test_numeric_question_selects_analytics(self):
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("how many orders were placed last quarter?")
        specialists = {t.specialist for t in plan.subtasks}
        self.assertIn("analytics", specialists)

    def test_visual_question_selects_visual(self):
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("show me the architecture diagram")
        self.assertIn("visual", {t.specialist for t in plan.subtasks})

    def test_causal_question_selects_graph(self):
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("why did shipments from Acme decline?")
        self.assertIn("graph", {t.specialist for t in plan.subtasks})

    def test_document_is_always_included(self):
        from src.graph.nodes.supervisor import plan_heuristic
        for q in ("what is the refund policy",
                  "how many orders were placed",
                  "show me the chart"):
            self.assertIn("document", {t.specialist for t in plan_heuristic(q).subtasks}, q)

    def test_every_specialist_is_known(self):
        from src.graph.state import SPECIALISTS
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("compare Q1 and Q2 revenue trends in the chart")
        for t in plan.subtasks:
            self.assertIn(t.specialist, SPECIALISTS)

    def test_no_duplicate_specialists(self):
        from src.graph.nodes.supervisor import plan_heuristic
        plan = plan_heuristic("how many total orders and what is the count of returns")
        specialists = [t.specialist for t in plan.subtasks]
        self.assertEqual(len(specialists), len(set(specialists)))


class TestSupervisorNode(unittest.TestCase):

    def test_node_returns_plan_and_specialists(self):
        from src.graph.nodes.supervisor import supervisor_node
        from src.graph.state import new_state
        out = supervisor_node(new_state("what is the refund policy?"))
        self.assertIn("plan", out)
        self.assertIn("specialists", out)
        self.assertTrue(out["specialists"])
        self.assertTrue(any(t.startswith("supervisor:") for t in out["trace"]))

    def test_empty_question_still_plans(self):
        from src.graph.nodes.supervisor import supervisor_node
        from src.graph.state import new_state
        out = supervisor_node(new_state(""))
        self.assertEqual(out["specialists"], ["document"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_supervisor.py -v`
Expected: FAIL — `No module named 'src.graph.nodes'`

- [ ] **Step 3: Write minimal implementation**

`supervisor_node` tries `get_structured_llm(RoutePlan)`; on `None` or any exception it falls back to `plan_heuristic` and records `trace: ["supervisor:heuristic"]` vs `["supervisor:llm"]`. It validates every LLM-returned specialist against `SPECIALISTS`, dropping unknown ones, and falls back if nothing valid remains.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_supervisor.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/prompts.py src/graph/nodes/ tests/test_graph_supervisor.py
git commit -m "feat(graph): supervisor planning node with LLM routing and keyword fallback"
```

---

### Task 5: Synthesizer and verifier nodes

**Files:**
- Create: `src/graph/nodes/synthesizer.py`, `src/graph/nodes/verifier.py`
- Test: `tests/test_graph_synthesis.py`

**Interfaces:**
- Consumes: `Finding`, `Citation`, `Verification`, `get_llm`, `get_structured_llm`.
- Produces:
  - `synthesize_heuristic(question: str, findings: list[Finding]) -> tuple[str, list[Citation]]` — extractive answer from the top findings by score, with `[n]` citation markers.
  - `synthesizer_node(state) -> dict` — `{"answer", "citations", "trace"}`.
  - `grade_grounded_heuristic(answer: str, findings: list[Finding]) -> Verification` — token-overlap containment; `grounded=True` when overlap ratio >= 0.35.
  - `verifier_node(state) -> dict` — `{"verification", "retry_count", "trace"}`.
  - `should_retry(state) -> str` — returns `"retry"` when verification failed and `retry_count < 1`, else `"done"`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for synthesis and verification nodes (heuristic mode)."""

import unittest

from src.graph.state import Finding, new_state


def _findings():
    return [
        Finding(specialist="document", content="Refunds are issued within 30 days.",
                score=0.9, source="policy.pdf", doc_id="d1"),
        Finding(specialist="document", content="Shipping takes 5 business days.",
                score=0.4, source="shipping.pdf", doc_id="d2"),
    ]


class TestSynthesizer(unittest.TestCase):

    def test_answer_uses_highest_scoring_finding(self):
        from src.graph.nodes.synthesizer import synthesize_heuristic
        answer, _ = synthesize_heuristic("what is the refund window?", _findings())
        self.assertIn("30 days", answer)

    def test_citations_reference_sources(self):
        from src.graph.nodes.synthesizer import synthesize_heuristic
        _, citations = synthesize_heuristic("refund window?", _findings())
        self.assertTrue(citations)
        self.assertIn("policy.pdf", {c.source for c in citations})

    def test_no_findings_returns_honest_answer(self):
        from src.graph.nodes.synthesizer import synthesize_heuristic
        answer, citations = synthesize_heuristic("anything?", [])
        self.assertEqual(citations, [])
        self.assertRegex(answer.lower(), r"could not|no relevant|not find")

    def test_node_populates_state_keys(self):
        from src.graph.nodes.synthesizer import synthesizer_node
        state = new_state("refund window?")
        state["findings"] = _findings()
        out = synthesizer_node(state)
        self.assertTrue(out["answer"])
        self.assertIn("citations", out)


class TestVerifier(unittest.TestCase):

    def test_grounded_answer_passes(self):
        from src.graph.nodes.verifier import grade_grounded_heuristic
        v = grade_grounded_heuristic("Refunds are issued within 30 days.", _findings())
        self.assertTrue(v.grounded)

    def test_ungrounded_answer_fails(self):
        from src.graph.nodes.verifier import grade_grounded_heuristic
        v = grade_grounded_heuristic(
            "The company was founded by penguins on Jupiter in 1823.", _findings())
        self.assertFalse(v.grounded)

    def test_no_findings_is_not_grounded(self):
        from src.graph.nodes.verifier import grade_grounded_heuristic
        self.assertFalse(grade_grounded_heuristic("Anything at all.", []).grounded)

    def test_should_retry_once_then_stops(self):
        from src.graph.nodes.verifier import should_retry
        state = new_state("q")
        state["verification"] = {"grounded": False, "relevant": False}
        state["retry_count"] = 0
        self.assertEqual(should_retry(state), "retry")
        state["retry_count"] = 1
        self.assertEqual(should_retry(state), "done")

    def test_should_not_retry_when_grounded(self):
        from src.graph.nodes.verifier import should_retry
        state = new_state("q")
        state["verification"] = {"grounded": True, "relevant": True}
        state["retry_count"] = 0
        self.assertEqual(should_retry(state), "done")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_synthesis.py -v`
Expected: FAIL — `No module named 'src.graph.nodes.synthesizer'`

- [ ] **Step 3: Write minimal implementation**

Reuse the existing stop-word tokenizer pattern from `src/evaluation/eval_pipeline.py` for overlap scoring rather than adding a new NLP dependency.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_synthesis.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/synthesizer.py src/graph/nodes/verifier.py tests/test_graph_synthesis.py
git commit -m "feat(graph): synthesizer and groundedness verifier with retry gate"
```

---

### Task 6: Document specialist — corrective RAG subgraph

**Files:**
- Create: `src/graph/nodes/document.py`
- Test: `tests/test_graph_document.py`

**Interfaces:**
- Consumes: `ChunkRetriever`, `documents_to_findings`, `GradeDocuments`, `get_structured_llm`.
- Produces:
  - `grade_documents_heuristic(subtask: str, docs: list[Document]) -> list[tuple[Document, GradeDocuments]]`
  - `rewrite_query_heuristic(question: str, attempt: int) -> str`
  - `build_document_subgraph(retriever, max_retries: int = 2)` — compiled `StateGraph` over `DocState` (`TypedDict` with `question, query, documents, findings, retries, trace`).
  - `document_node(state, components) -> dict` — invokes the subgraph, returns `{"findings": [...], "trace": [...]}`.

Subgraph topology: `retrieve -> grade -> {relevant: emit, weak+budget: rewrite -> retrieve, weak+exhausted: emit_low_confidence}`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the corrective-RAG document subgraph."""

import unittest
from langchain_core.documents import Document


class _StubRetriever:
    """Returns weak docs until `flip_after` calls, then a strong one."""
    def __init__(self, weak_doc, strong_doc=None, flip_after=99):
        self.weak_doc, self.strong_doc = weak_doc, strong_doc
        self.flip_after, self.calls = flip_after, 0

    def invoke(self, query, config=None):
        self.calls += 1
        if self.strong_doc is not None and self.calls > self.flip_after:
            return [self.strong_doc]
        return [self.weak_doc]


WEAK = Document(page_content="Unrelated text about gardening tools.",
                metadata={"source": "g.pdf", "doc_id": "g1", "score": 0.1})
STRONG = Document(page_content="The refund policy allows returns within 30 days.",
                  metadata={"source": "p.pdf", "doc_id": "p1", "score": 0.9})


class TestGrading(unittest.TestCase):

    def test_relevant_doc_graded_relevant(self):
        from src.graph.nodes.document import grade_documents_heuristic
        graded = grade_documents_heuristic("refund policy return window", [STRONG])
        self.assertTrue(graded[0][1].relevant)

    def test_irrelevant_doc_graded_irrelevant(self):
        from src.graph.nodes.document import grade_documents_heuristic
        graded = grade_documents_heuristic("refund policy return window", [WEAK])
        self.assertFalse(graded[0][1].relevant)


class TestRewrite(unittest.TestCase):

    def test_rewrite_changes_the_query(self):
        from src.graph.nodes.document import rewrite_query_heuristic
        original = "refund policy"
        self.assertNotEqual(rewrite_query_heuristic(original, 1), original)

    def test_rewrites_differ_across_attempts(self):
        from src.graph.nodes.document import rewrite_query_heuristic
        self.assertNotEqual(rewrite_query_heuristic("refund policy", 1),
                            rewrite_query_heuristic("refund policy", 2))


class TestSubgraph(unittest.TestCase):

    def test_strong_docs_need_no_retry(self):
        from src.graph.nodes.document import build_document_subgraph
        r = _StubRetriever(STRONG)
        g = build_document_subgraph(r)
        out = g.invoke({"question": "refund policy return window", "query": "",
                        "documents": [], "findings": [], "retries": 0, "trace": []})
        self.assertEqual(r.calls, 1)
        self.assertTrue(out["findings"])

    def test_weak_docs_trigger_rewrite_and_recover(self):
        from src.graph.nodes.document import build_document_subgraph
        r = _StubRetriever(WEAK, STRONG, flip_after=1)
        g = build_document_subgraph(r)
        out = g.invoke({"question": "refund policy return window", "query": "",
                        "documents": [], "findings": [], "retries": 0, "trace": []})
        self.assertGreaterEqual(r.calls, 2)
        self.assertTrue(out["findings"])
        self.assertTrue(any("rewrite" in t for t in out["trace"]))

    def test_retry_budget_terminates(self):
        """Permanently weak retrieval must stop, not loop forever."""
        from src.graph.nodes.document import build_document_subgraph
        r = _StubRetriever(WEAK)
        g = build_document_subgraph(r, max_retries=2)
        out = g.invoke({"question": "refund policy return window", "query": "",
                        "documents": [], "findings": [], "retries": 0, "trace": []})
        self.assertLessEqual(r.calls, 3)          # initial + 2 retries
        self.assertTrue(any("low_confidence" in t for t in out["trace"]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_document.py -v`
Expected: FAIL — `No module named 'src.graph.nodes.document'`

- [ ] **Step 3: Write minimal implementation**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_document.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/document.py tests/test_graph_document.py
git commit -m "feat(graph): corrective-RAG document subgraph with self-grading and bounded retries"
```

---

### Task 7: Visual, analytics, and graph specialists (with HITL gate)

**Files:**
- Create: `src/graph/nodes/visual.py`, `src/graph/nodes/analytics.py`, `src/graph/nodes/graph_specialist.py`
- Test: `tests/test_graph_specialists.py`

**Interfaces:**
- Produces:
  - `visual_node(state, components) -> dict` — findings with `modality="image"`; empty list when no visual index.
  - `needs_approval(sql: str) -> bool` — `True` when the statement is not a single read-only `SELECT`/`WITH`.
  - `analytics_node(state, components) -> dict` — calls `interrupt()` when `needs_approval` and `require_approval` is set in config.
  - `graph_node(state, components) -> dict` — entity-path findings via `GraphRetriever`/`MultiHopRetriever`.
  - All three return `{"findings": [...], "trace": [...]}` and never raise; component failure yields an empty finding list plus a `trace` entry ending in `:unavailable`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for visual, analytics, and knowledge-graph specialists."""

import unittest

from src.graph.state import new_state


class _NullComponents:
    """Every component unavailable — specialists must degrade, not crash."""
    hybrid = multimodal = sql = graph = multi_hop = generator = None


class TestApprovalGate(unittest.TestCase):

    def test_select_needs_no_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertFalse(needs_approval("SELECT * FROM orders LIMIT 10"))

    def test_cte_select_needs_no_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertFalse(needs_approval("WITH t AS (SELECT 1) SELECT * FROM t"))

    def test_delete_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval("DELETE FROM orders"))

    def test_drop_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval("DROP TABLE orders"))

    def test_multi_statement_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval("SELECT 1; DROP TABLE orders"))

    def test_empty_sql_needs_approval(self):
        from src.graph.nodes.analytics import needs_approval
        self.assertTrue(needs_approval(""))


class TestSpecialistDegradation(unittest.TestCase):
    """With no indices or keys, every specialist returns empty, never raises."""

    def _run(self, node):
        out = node(new_state("q"), _NullComponents())
        self.assertEqual(out["findings"], [])
        self.assertTrue(any(t.endswith(":unavailable") for t in out["trace"]))

    def test_visual_degrades(self):
        from src.graph.nodes.visual import visual_node
        self._run(visual_node)

    def test_analytics_degrades(self):
        from src.graph.nodes.analytics import analytics_node
        self._run(analytics_node)

    def test_graph_degrades(self):
        from src.graph.nodes.graph_specialist import graph_node
        self._run(graph_node)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_specialists.py -v`
Expected: FAIL — `No module named 'src.graph.nodes.visual'`

- [ ] **Step 3: Write minimal implementation**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_specialists.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/visual.py src/graph/nodes/analytics.py src/graph/nodes/graph_specialist.py tests/test_graph_specialists.py
git commit -m "feat(graph): visual, analytics, and knowledge-graph specialists with SQL approval gate"
```

---

### Task 8: Graph assembly, checkpointing, and HITL resume

**Files:**
- Create: `src/graph/build.py`
- Test: `tests/test_graph_build.py`

**Interfaces:**
- Produces:
  - `route_to_specialists(state) -> list[Send]` — one `Send` per selected specialist.
  - `build_analyst_graph(components=None, checkpointer=None, max_retries=2)` — compiled graph.
  - `get_checkpointer(path=None)` — `SqliteSaver` at `data/graph_checkpoints.db`, `MemorySaver` when `path == ":memory:"`.
  - `run_query(question, thread_id=None, require_approval=True, components=None) -> dict` — convenience wrapper returning `{"answer", "citations", "findings", "verification", "trace", "thread_id", "interrupted"}`.
  - `graph_mermaid() -> str` — Mermaid topology for docs and `/api/v2/graph`.

- [ ] **Step 1: Write the failing test**

```python
"""End-to-end tests for the assembled analyst graph (offline)."""

import unittest


class _StubComponents:
    """Only the document path is wired; the rest are unavailable."""
    def __init__(self, rows):
        self._rows = rows
        self.multimodal = self.sql = self.graph = self.multi_hop = self.generator = None

    @property
    def hybrid(self):
        rows = self._rows
        class _B:
            def retrieve(self, query, top_k=5, **kw):
                return rows[:top_k]
        return _B()


ROWS = [{"text": "Refunds are issued within 30 days of purchase.",
         "doc_id": "d1", "source": "policy.pdf", "score": 0.95}]


class TestGraphAssembly(unittest.TestCase):

    def test_graph_compiles(self):
        from src.graph.build import build_analyst_graph
        self.assertIsNotNone(build_analyst_graph(_StubComponents(ROWS)))

    def test_mermaid_contains_core_nodes(self):
        from src.graph.build import graph_mermaid
        m = graph_mermaid()
        for node in ("supervisor", "synthesizer", "verifier", "document"):
            self.assertIn(node, m)

    def test_end_to_end_produces_grounded_answer(self):
        from src.graph.build import run_query
        result = run_query("what is the refund window?",
                           components=_StubComponents(ROWS),
                           require_approval=False)
        self.assertIn("30 days", result["answer"])
        self.assertTrue(result["citations"])
        self.assertTrue(any(t.startswith("supervisor:") for t in result["trace"]))
        self.assertFalse(result["interrupted"])

    def test_parallel_findings_merge_without_loss(self):
        """Multiple specialists in one superstep must all contribute."""
        from src.graph.build import run_query
        result = run_query("show me the chart of total orders and why they fell",
                           components=_StubComponents(ROWS),
                           require_approval=False)
        # supervisor should fan out to more than just `document`
        specialists = {t.split(":")[0] for t in result["trace"]}
        self.assertGreaterEqual(len(specialists & {"document", "visual",
                                                   "analytics", "graph"}), 2)


class TestCheckpointing(unittest.TestCase):

    def test_memory_checkpointer_round_trips_a_thread(self):
        from src.graph.build import build_analyst_graph, get_checkpointer
        from src.graph.state import new_state
        g = build_analyst_graph(_StubComponents(ROWS),
                                checkpointer=get_checkpointer(":memory:"))
        cfg = {"configurable": {"thread_id": "t1", "require_approval": False}}
        g.invoke(new_state("what is the refund window?"), cfg)
        snapshot = g.get_state(cfg)
        self.assertEqual(snapshot.values["question"], "what is the refund window?")
        self.assertTrue(snapshot.values["answer"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_build.py -v`
Expected: FAIL — `No module named 'src.graph.build'`

- [ ] **Step 3: Write minimal implementation**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_build.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/build.py tests/test_graph_build.py
git commit -m "feat(graph): assemble analyst graph with parallel fan-out and checkpointing"
```

---

### Task 9: HITL interrupt and resume

**Files:**
- Modify: `src/graph/nodes/analytics.py`, `src/graph/build.py`
- Test: `tests/test_graph_hitl.py`

**Interfaces:**
- Produces:
  - `resume_query(thread_id: str, decision: str, components=None) -> dict` — resumes an interrupted thread with `Command(resume=decision)`.
  - `run_query(...)` returns `interrupted=True` plus `interrupt_payload` when the graph pauses.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for human-in-the-loop interrupt and resume."""

import unittest


class _WriteSQLComponents:
    """SQL backend that proposes a destructive statement."""
    multimodal = graph = multi_hop = generator = hybrid = None

    class _SQL:
        def analyze(self, question):
            return {"sql": "DELETE FROM orders WHERE stale = 1",
                    "rows": [], "insight": "cleanup", "columns": []}
    sql = _SQL()


class TestHITL(unittest.TestCase):

    def setUp(self):
        from src.graph.build import build_analyst_graph, get_checkpointer
        self.graph = build_analyst_graph(_WriteSQLComponents(),
                                         checkpointer=get_checkpointer(":memory:"))
        self.cfg = {"configurable": {"thread_id": "hitl-1",
                                     "require_approval": True}}

    def test_destructive_sql_interrupts(self):
        from src.graph.state import new_state
        result = self.graph.invoke(new_state("delete the stale orders"), self.cfg)
        self.assertIn("__interrupt__", result)

    def test_resume_with_approval_completes(self):
        from langgraph.types import Command
        from src.graph.state import new_state
        self.graph.invoke(new_state("delete the stale orders"), self.cfg)
        final = self.graph.invoke(Command(resume="approve"), self.cfg)
        self.assertNotIn("__interrupt__", final)
        self.assertTrue(final["answer"])

    def test_resume_with_rejection_records_denial(self):
        from langgraph.types import Command
        from src.graph.state import new_state
        self.graph.invoke(new_state("delete the stale orders"), self.cfg)
        final = self.graph.invoke(Command(resume="reject"), self.cfg)
        self.assertTrue(any("denied" in t for t in final["trace"]))

    def test_no_interrupt_when_approval_disabled(self):
        from src.graph.state import new_state
        cfg = {"configurable": {"thread_id": "hitl-2", "require_approval": False}}
        result = self.graph.invoke(new_state("delete the stale orders"), cfg)
        self.assertNotIn("__interrupt__", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_hitl.py -v`
Expected: FAIL — no interrupt raised

- [ ] **Step 3: Write minimal implementation**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_hitl.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/analytics.py src/graph/build.py tests/test_graph_hitl.py
git commit -m "feat(graph): human-in-the-loop approval gate with interrupt and resume"
```

---

### Task 10: LangSmith datasets, evaluators, and experiments

**Files:**
- Create: `src/evaluation/langsmith_eval.py`
- Test: `tests/test_langsmith_eval.py`

**Interfaces:**
- Produces:
  - `build_examples() -> list[dict]` — from `src/evaluation/test_cases.py`, each `{"inputs": {"question": ...}, "outputs": {...}}`.
  - Evaluators, all pure functions with signature `(run_outputs: dict, example_outputs: dict) -> dict` returning `{"key", "score", "comment"}`:
    `faithfulness`, `citation_accuracy`, `routing_accuracy`, `modality_match`, `retry_efficiency`.
  - `EVALUATORS: dict[str, callable]`
  - `push_dataset(name, client=None) -> str | None` — no-op returning `None` without a LangSmith key.
  - `run_experiment(name, config, client=None) -> dict` — no-op-safe.
  - `mirror_feedback(query_id, rating, comment, client=None) -> bool`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for LangSmith evaluators — pure functions, no network."""

import unittest


class TestEvaluators(unittest.TestCase):

    def test_faithfulness_rewards_grounded_answers(self):
        from src.evaluation.langsmith_eval import faithfulness
        run = {"answer": "Refunds are issued within 30 days.",
               "findings": [{"content": "Refunds are issued within 30 days of purchase."}]}
        self.assertGreaterEqual(faithfulness(run, {})["score"], 0.5)

    def test_faithfulness_punishes_ungrounded_answers(self):
        from src.evaluation.langsmith_eval import faithfulness
        run = {"answer": "Penguins founded the company on Jupiter.",
               "findings": [{"content": "Refunds are issued within 30 days."}]}
        self.assertLess(faithfulness(run, {})["score"], 0.5)

    def test_citation_accuracy_requires_cited_sources_in_findings(self):
        from src.evaluation.langsmith_eval import citation_accuracy
        good = {"citations": [{"source": "policy.pdf"}],
                "findings": [{"source": "policy.pdf"}]}
        bad = {"citations": [{"source": "ghost.pdf"}],
               "findings": [{"source": "policy.pdf"}]}
        self.assertEqual(citation_accuracy(good, {})["score"], 1.0)
        self.assertEqual(citation_accuracy(bad, {})["score"], 0.0)

    def test_routing_accuracy_scores_specialist_overlap(self):
        from src.evaluation.langsmith_eval import routing_accuracy
        run = {"specialists": ["document", "analytics"]}
        self.assertEqual(routing_accuracy(run, {"specialists": ["analytics"]})["score"], 1.0)
        self.assertEqual(routing_accuracy(run, {"specialists": ["visual"]})["score"], 0.0)

    def test_routing_accuracy_without_expectation_is_neutral(self):
        from src.evaluation.langsmith_eval import routing_accuracy
        self.assertIsNone(routing_accuracy({"specialists": ["document"]}, {})["score"])

    def test_modality_match(self):
        from src.evaluation.langsmith_eval import modality_match
        run = {"findings": [{"modality": "image"}]}
        self.assertEqual(modality_match(run, {"modality": "image"})["score"], 1.0)

    def test_retry_efficiency_prefers_fewer_retries(self):
        from src.evaluation.langsmith_eval import retry_efficiency
        self.assertGreater(retry_efficiency({"retry_count": 0}, {})["score"],
                           retry_efficiency({"retry_count": 2}, {})["score"])

    def test_all_evaluators_registered(self):
        from src.evaluation.langsmith_eval import EVALUATORS
        for name in ("faithfulness", "citation_accuracy", "routing_accuracy",
                     "modality_match", "retry_efficiency"):
            self.assertIn(name, EVALUATORS)


class TestDatasetBuilding(unittest.TestCase):

    def test_build_examples_from_test_cases(self):
        from src.evaluation.langsmith_eval import build_examples
        examples = build_examples()
        self.assertTrue(examples)
        self.assertIn("question", examples[0]["inputs"])

    def test_push_dataset_noop_without_key(self):
        from src.evaluation.langsmith_eval import push_dataset
        self.assertIsNone(push_dataset("test-ds", client=None))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_langsmith_eval.py -v`
Expected: FAIL — `No module named 'src.evaluation.langsmith_eval'`

- [ ] **Step 3: Write minimal implementation**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_langsmith_eval.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/langsmith_eval.py tests/test_langsmith_eval.py
git commit -m "feat(eval): LangSmith dataset builder, evaluators, and experiment runner"
```

---

### Task 11: API v2 endpoints, CLI demo, and documentation

**Files:**
- Create: `src/graph/demo.py`, `docs/AGENTIC_ARCHITECTURE.md`
- Modify: `src/api/main.py` (append v2 routes), `src/api/models.py` (add request/response models), `README.md`, `.env` (document new keys)
- Test: `tests/test_graph_api.py`

**Interfaces:**
- Produces:
  - `POST /api/v2/query` — body `{question, thread_id?, require_approval?}` → `{answer, citations, findings, verification, trace, thread_id, interrupted, interrupt_payload}`
  - `POST /api/v2/resume` — body `{thread_id, decision}` → same shape
  - `GET /api/v2/graph` → `{mermaid, nodes, llm_mode, tracing}`
  - `python -m src.graph.demo "question"` — streams `node -> update` lines to stdout.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the v2 agentic API surface."""

import unittest


class TestV2Routes(unittest.TestCase):

    def setUp(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        self.client = TestClient(app)

    def test_graph_topology_endpoint(self):
        r = self.client.get("/api/v2/graph")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("supervisor", body["mermaid"])
        self.assertIn(body["llm_mode"], ("llm", "heuristic"))

    def test_query_endpoint_rejects_empty_question(self):
        r = self.client.post("/api/v2/query", json={"question": ""})
        self.assertIn(r.status_code, (400, 422))

    def test_query_endpoint_returns_expected_shape(self):
        r = self.client.post("/api/v2/query",
                             json={"question": "what is the refund policy?",
                                   "require_approval": False})
        self.assertEqual(r.status_code, 200)
        for key in ("answer", "citations", "trace", "thread_id", "interrupted"):
            self.assertIn(key, r.json())

    def test_resume_unknown_thread_is_handled(self):
        r = self.client.post("/api/v2/resume",
                             json={"thread_id": "does-not-exist", "decision": "approve"})
        self.assertIn(r.status_code, (200, 404))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_api.py -v`
Expected: FAIL — 404 on `/api/v2/graph`

- [ ] **Step 3: Write minimal implementation**

`docs/AGENTIC_ARCHITECTURE.md` documents the topology (embedding the Mermaid from `graph_mermaid()`), the heuristic-vs-LLM degradation table, the HITL flow, and how to enable LangSmith. `README.md` gains a "LangGraph Agentic Layer" section with setup steps for `GEMINI_API_KEY` and `LANGSMITH_API_KEY`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_api.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the whole suite and commit**

```bash
python -m pytest tests/ -q
git add src/graph/demo.py src/api/main.py src/api/models.py docs/AGENTIC_ARCHITECTURE.md README.md tests/test_graph_api.py
git commit -m "feat(api): v2 agentic endpoints, streaming CLI demo, and architecture docs"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| 2.1 package layout | 1, 2, 3, 4, 5, 6, 7, 8 |
| 2.2 state schema and reducers | 1 |
| 2.3 specialist subgraphs | 6, 7 |
| 2.4 corrective-RAG subgraph | 6 |
| 3 graceful degradation | 2 (factory), 4/5/6/7 (per-node fallbacks) |
| 4 human-in-the-loop | 7 (gate), 9 (interrupt/resume) |
| 5 persistence | 8 |
| 6 LangSmith tracing and evaluation | 2 (tracing), 10 (datasets/evaluators) |
| 7 surfaces | 11 |
| 8 testing strategy | every task |
| 9 phasing | Tasks 1-2 = P1, 4-5+8 = P2, 6-7 = P3, 9 = P4, 10 = P5, 11 = P6 |

**Type consistency:** `Finding`, `Citation`, `SubTask`, `RoutePlan`, `GradeDocuments`, `Verification` are defined in Task 1 and used unchanged thereafter. `components` is the `LazyComponents` instance from Task 3, passed to every specialist node in Tasks 6-7 and constructed in Task 8. `trace` entries follow `"<name>:<mode>"` throughout, which Tasks 7 and 8 assert on.

**Placeholder scan:** No TBDs. Steps 3 in Tasks 6-11 describe the implementation contract rather than repeating full source; the tests in Step 1 of each task are complete and define the acceptance criteria exactly.
