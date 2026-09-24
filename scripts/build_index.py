"""Build every index the app needs, and fail loudly if any of it is missing.

``EnterpriseRAGOrchestrator.setup()`` logs and swallows step failures so a
developer machine degrades instead of crashing. That is the wrong behaviour
at image-build time: a container that boots with an empty vector store looks
healthy and answers every question with "I could not find relevant
information". Run in the Dockerfile, this turns that into a failed build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline_orchestrator import EnterpriseRAGOrchestrator  # noqa: E402


def main() -> int:
    report = EnterpriseRAGOrchestrator().setup(data_dir="data")
    print(json.dumps(report, indent=2, default=str))

    problems = []
    for step in ("ingestion", "chunking", "embedding", "bm25", "sql_database"):
        if isinstance(report.get(step), dict) and "error" in report[step]:
            problems.append(f"{step}: {report[step]['error']}")
    if (report.get("embedding") or {}).get("vectors_stored", 0) == 0:
        problems.append("embedding: no vectors stored")
    if (report.get("chunking") or {}).get("chunks", 0) == 0:
        problems.append("chunking: no chunks created")

    # The v1 "Ask Questions" path falls back to a small local model
    # (flan-t5) when no API key is set. Cache it now: the container runs with
    # HF_HUB_OFFLINE=1, so anything not downloaded here is unavailable later.
    try:
        from src.llm_provider import _get_hf_pipeline

        _get_hf_pipeline()
        print("Cached local fallback LLM.")
    except Exception as exc:  # optional path; the graph does not need it
        print(f"Warning: could not cache local fallback LLM: {exc}", file=sys.stderr)

    if problems:
        print("\nIndex build FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("\nIndex build OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
