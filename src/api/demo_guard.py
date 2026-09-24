"""Guard rails for running the API as a public demo.

Off by default. With ``DEMO_MODE=1`` the API stays read-only and bounded:

* uploads and the evaluation run are refused -- a stranger should not be able
  to write files to the server, and one evaluation run is ~200 LLM calls,
  which is more than a free-tier key allows in a day;
* questions are capped per UTC day (``DEMO_DAILY_QUERY_LIMIT``, default 200),
  so a public link cannot drain the LLM key behind it.

The counter lives in process memory. That is deliberate: the demo runs as a
single container, and a restart resetting the budget is acceptable where a
shared store would be one more thing to operate.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

BLOCKED = {
    ("POST", "/api/upload"),
    ("GET", "/api/evaluation"),
}
METERED = {
    ("POST", "/api/ask"),
    ("POST", "/api/v2/query"),
    ("POST", "/api/v2/resume"),
}


def demo_mode_enabled() -> bool:
    return os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def daily_limit() -> int:
    try:
        return max(0, int(os.getenv("DEMO_DAILY_QUERY_LIMIT", "200")))
    except ValueError:
        return 200


class DailyBudget:
    """Thread-safe counter that resets at UTC midnight."""

    def __init__(self, limit: int, today: Optional[Callable[[], str]] = None) -> None:
        self.limit = limit
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._day = self._today()
        self._used = 0
        self._lock = threading.Lock()

    def try_consume(self) -> bool:
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day, self._used = day, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            if self._today() != self._day:
                return self.limit
            return max(0, self.limit - self._used)


class DemoGuardMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, budget: Optional[DailyBudget] = None) -> None:
        super().__init__(app)
        self.budget = budget or DailyBudget(daily_limit())

    async def dispatch(self, request: Request, call_next) -> Response:
        key = (request.method.upper(), request.url.path.rstrip("/") or "/")
        if key in BLOCKED:
            return JSONResponse(
                status_code=403,
                content={"detail": "Disabled in the public demo. Run the project locally to use it."},
            )
        if key in METERED and not self.budget.try_consume():
            return JSONResponse(
                status_code=429,
                content={"detail": "The public demo has reached today's question limit. Please try again tomorrow."},
            )
        return await call_next(request)
