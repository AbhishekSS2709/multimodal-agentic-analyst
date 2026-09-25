"""Record the Agentic Analyst tab answering a question, for the README.

Run against a live container (CI does this after the smoke test):

    python scripts/record_demo.py --url http://localhost:7860 --out demo

Writes ``demo/demo.webm`` (screen recording) and ``demo/demo.png`` (the
finished answer). ``ffmpeg`` turns the recording into the README GIF.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

QUESTION = "Why are there dispatch delays?"
VIEWPORT = {"width": 1280, "height": 860}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--out", default="demo")
    parser.add_argument("--timeout", type=int, default=240, help="seconds to wait for the answer")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_dir = out / "_video"

    launch = {}
    if os.getenv("CHROMIUM_PATH"):
        launch["executable_path"] = os.environ["CHROMIUM_PATH"]

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        context = browser.new_context(viewport=VIEWPORT, record_video_dir=str(video_dir),
                                      record_video_size=VIEWPORT)
        page = context.new_page()

        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("Agentic Analyst", exact=False).first.wait_for(timeout=60_000)
        page.wait_for_timeout(1500)

        box = page.get_by_label("Question", exact=True)
        box.click()
        box.type(QUESTION, delay=45)
        box.press("Enter")
        page.wait_for_timeout(800)
        page.get_by_role("button", name="Run analyst").click()

        page.get_by_text("Specialists dispatched").wait_for(timeout=args.timeout * 1000)
        page.wait_for_timeout(1500)
        # Streamlit scrolls its own container, so a full-page shot would only
        # capture the viewport anyway; take it before scrolling to the trace.
        page.screenshot(path=str(out / "demo.png"))

        trace = page.get_by_text("Agent trace (node by node)")
        if trace.count():
            trace.first.click()
            page.wait_for_timeout(800)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(2500)

        video = page.video
        context.close()
        browser.close()
        shutil.move(video.path(), out / "demo.webm")

    shutil.rmtree(video_dir, ignore_errors=True)
    print(f"Wrote {out / 'demo.webm'} and {out / 'demo.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
