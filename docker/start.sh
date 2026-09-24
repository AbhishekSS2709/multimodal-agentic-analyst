#!/usr/bin/env bash
# Run the FastAPI backend (internal only) and the Streamlit UI (public) in one
# container. If either process dies, exit non-zero so the platform restarts
# the container instead of serving a UI with no backend behind it.
set -uo pipefail

PORT="${PORT:-7860}"

python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --log-level info &
API_PID=$!

for _ in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/api/stats >/dev/null; then
    echo "[start] API is up"
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[start] API exited during startup" >&2
    exit 1
  fi
  sleep 1
done

python -m streamlit run app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false &
UI_PID=$!

wait -n "$API_PID" "$UI_PID"
echo "[start] a process exited; stopping container" >&2
kill "$API_PID" "$UI_PID" 2>/dev/null
exit 1
