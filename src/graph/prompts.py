"""Prompts for the analyst graph, kept in one place.

Centralised so they can be versioned, diffed, and A/B-tested as LangSmith
experiments without hunting through node modules.
"""

from __future__ import annotations

SUPERVISOR_PROMPT = """You are the supervisor of a research team answering a question \
about an enterprise document corpus.

You have four specialists:
- document:  searches text passages across PDFs, Word, HTML, code, and transcripts.
- visual:    searches images, charts, diagrams, slides, and video frames.
- analytics: answers quantitative questions by querying a SQL warehouse.
- graph:     traverses a knowledge graph of entities to answer causal, \
multi-hop, or comparative questions -- why something happened, what \
contributed to it, or how two entities differ.

Decompose the question into the smallest set of sub-tasks that answers it.
Rules:
- Use at most one sub-task per specialist.
- Only pick a specialist that genuinely contributes; do not pick all four by default.
- Always include `document`. Text retrieval is the floor: it is almost
  always worth running, and it is what the answer falls back to when a
  specialist returns nothing.
- Comparing, ranking, or contrasting two things is a `graph` job, not an
  `analytics` one, unless the comparison is a plain aggregate over the
  warehouse.
- Each sub-task description must be a self-contained search instruction.

Question: {question}"""


GRADE_DOCUMENTS_PROMPT = """You are grading whether a retrieved document helps \
answer a sub-task.

Sub-task: {subtask}

Document:
{document}

Grade it relevant only if it contains information that would appear in a correct \
answer. Keyword overlap alone is not enough."""


REWRITE_QUERY_PROMPT = """The search below returned nothing useful. Rewrite it to \
retrieve better results.

Original question: {question}
Failed query: {query}
Attempt number: {attempt}

Produce a different query — broaden overly narrow phrasing, introduce likely \
synonyms and domain terms, or split a compound question into its most \
important part. Do not simply restate the original."""


SYNTHESIZE_PROMPT = """Answer the question using only the findings below. \
Each finding is numbered; cite the ones you use with bracketed markers like [1].

If the findings do not contain the answer, say so plainly rather than guessing.

Question: {question}

Findings:
{findings}"""


VERIFY_PROMPT = """You are auditing an answer for groundedness.

Question: {question}

Findings the answer was supposed to be based on:
{findings}

Answer:
{answer}

Mark it grounded only if every factual claim is supported by the findings. \
Mark it relevant only if it actually addresses the question."""
