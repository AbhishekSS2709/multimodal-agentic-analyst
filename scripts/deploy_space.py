"""Publish this repo to a Hugging Face Space (Docker SDK).

Run by .github/workflows/deploy.yml after CI passes on main. Needs HF_TOKEN
(a write token). The Space is created on first run; later runs update it and
Hugging Face rebuilds the image.

Optional env:
  HF_SPACE_ID            owner/name (default: <token owner>/multimodal-agentic-analyst)
  GROQ_API_KEY,
  GEMINI_API_KEY,
  LANGSMITH_API_KEY      copied into the Space as secrets when present, so the
                         graph runs in LLM mode instead of heuristic mode.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

FRONT_MATTER = """---
title: Multimodal Agentic Analyst
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Multi-agent RAG with LangGraph over text, images and SQL
---

"""

IGNORE = [
    ".git/*", ".github/*", ".pytest_cache/*", "**/__pycache__/*", "*.pyc",
    ".env", "*.env", "vector_store/*", "evaluation_results/*",
    "static/charts/*", "data/graph_checkpoints.db*",
]

SECRETS = ("GROQ_API_KEY", "GEMINI_API_KEY", "LANGSMITH_API_KEY")


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set; nothing to deploy.", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    space_id = os.environ.get("HF_SPACE_ID", "").strip() or f"{owner}/multimodal-agentic-analyst"

    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    for name in SECRETS:
        value = os.environ.get(name, "").strip()
        if value:
            api.add_space_secret(space_id, name, value)
            print(f"Set Space secret {name}")
    if os.environ.get("LANGSMITH_API_KEY", "").strip():
        api.add_space_variable(space_id, "LANGSMITH_PROJECT", "multimodal-agentic-analyst")

    # The Space needs YAML front matter in its README; GitHub renders it as a
    # stray table, so it is only added to the copy that goes to the Space.
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "space"
        shutil.copytree(ROOT, stage, ignore=shutil.ignore_patterns(
            ".git", ".github", "__pycache__", ".pytest_cache", "vector_store",
            "evaluation_results", ".env", "*.env"))
        readme = stage / "README.md"
        readme.write_text(FRONT_MATTER + readme.read_text(encoding="utf-8"), encoding="utf-8")

        sha = os.environ.get("GITHUB_SHA", "")[:7]
        api.upload_folder(
            folder_path=str(stage),
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=IGNORE,
            delete_patterns=["*"],
            commit_message=f"Deploy from GitHub {sha}".strip(),
        )

    print(f"Deployed: https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
