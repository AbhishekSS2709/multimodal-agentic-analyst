"""Launch script for the Enterprise RAG System.

Starts both the FastAPI backend (port 8000) and the Streamlit frontend
(port 8501) as child processes.  Press Ctrl+C to stop both.
"""

import os
import signal
import subprocess
import sys
import time


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Enterprise RAG System — Launcher")
    print("=" * 60)
    print()
    print("  API server : http://localhost:8000")
    print("  API docs   : http://localhost:8000/docs")
    print("  Streamlit  : http://localhost:8501")
    print()
    print("  Press Ctrl+C to stop both services.")
    print("=" * 60)
    print()

    # Start FastAPI server
    print("[*] Starting FastAPI backend ...")
    api_process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info",
        ],
        cwd=cwd,
    )

    # Give the API a moment to bind the port
    time.sleep(2)

    # Start Streamlit frontend
    print("[*] Starting Streamlit frontend ...")
    ui_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=cwd,
    )

    try:
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down ...")
        api_process.terminate()
        ui_process.terminate()

        # Give them a few seconds to exit cleanly
        try:
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()

        try:
            ui_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ui_process.kill()

        print("[*] All services stopped.")


if __name__ == "__main__":
    main()
