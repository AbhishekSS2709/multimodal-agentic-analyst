"""Public-demo guard rails: blocked endpoints and the daily question budget."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.demo_guard import DailyBudget, DemoGuardMiddleware, demo_mode_enabled


def _app(budget: DailyBudget) -> FastAPI:
    app = FastAPI()
    app.add_middleware(DemoGuardMiddleware, budget=budget)

    @app.post("/api/ask")
    def ask():
        return {"ok": True}

    @app.post("/api/v2/query")
    def query():
        return {"ok": True}

    @app.post("/api/upload")
    def upload():
        return {"ok": True}

    @app.get("/api/evaluation")
    def evaluation():
        return {"ok": True}

    @app.get("/api/stats")
    def stats():
        return {"ok": True}

    return app


def test_upload_and_evaluation_are_refused():
    client = TestClient(_app(DailyBudget(10)))
    assert client.post("/api/upload").status_code == 403
    assert client.get("/api/evaluation").status_code == 403


def test_questions_share_one_budget_then_429():
    client = TestClient(_app(DailyBudget(2)))
    assert client.post("/api/ask").status_code == 200
    assert client.post("/api/v2/query").status_code == 200
    blocked = client.post("/api/ask")
    assert blocked.status_code == 429
    assert "limit" in blocked.json()["detail"]


def test_read_only_endpoints_are_not_metered():
    client = TestClient(_app(DailyBudget(0)))
    for _ in range(5):
        assert client.get("/api/stats").status_code == 200


def test_budget_resets_on_a_new_day():
    day = {"v": "2026-09-24"}
    budget = DailyBudget(1, today=lambda: day["v"])
    assert budget.try_consume()
    assert not budget.try_consume()
    day["v"] = "2026-09-25"
    assert budget.remaining == 1
    assert budget.try_consume()


def test_demo_mode_flag(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert not demo_mode_enabled()
    monkeypatch.setenv("DEMO_MODE", "1")
    assert demo_mode_enabled()
