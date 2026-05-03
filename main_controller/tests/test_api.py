"""Tests for the MainController FastAPI endpoints."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from main_controller.src.api import app, RunState
from shared.models.prediction import PredictionResult, SignalType


def _make_prediction_result() -> PredictionResult:
    return PredictionResult(
        run_id=str(uuid4()),
        symbol="BTCUSDT",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        signal=SignalType.BUY,
        confidence=0.85,
        explanation="Strong buy signal.",
        reasoning_steps=["step1", "step2"],
        similar_cases=[],
        sentiment_score=0.7,
        key_factors=[],
        errors=[],
    )


class _MockPipeline:
    async def run(self, symbol: str, run_id=None):
        result = _make_prediction_result()
        result.run_id = str(run_id) if run_id else result.run_id
        result.symbol = symbol
        await asyncio.sleep(0)  # yield to event loop
        return result


@pytest.fixture
def client():
    """TestClient with mocked pipeline injected into app.state."""
    with TestClient(app, raise_server_exceptions=True) as c:
        app.state.pipeline = _MockPipeline()
        app.state.run_states = {}
        app.state.background_tasks = set()
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_run_returns_pending(client):
    resp = client.post("/run?symbol=BTCUSDT")
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "pending"


def test_status_404_unknown_run_id(client):
    resp = client.get("/status/nonexistent-run-id")
    assert resp.status_code == 404


def test_result_404_unknown_run_id(client):
    resp = client.get("/result/nonexistent-run-id")
    assert resp.status_code == 404


def test_result_425_if_not_done(client):
    # Manually insert a pending state
    run_id = str(uuid4())
    app.state.run_states[run_id] = RunState(run_id=run_id, symbol="BTCUSDT")
    resp = client.get(f"/result/{run_id}")
    assert resp.status_code == 425


def test_run_then_status_reflects_state(client):
    # Trigger run
    post_resp = client.post("/run?symbol=BTCUSDT")
    run_id = post_resp.json()["run_id"]

    # Status should exist (pending or running or done depending on timing)
    status_resp = client.get(f"/status/{run_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["run_id"] == run_id
    assert body["status"] in ("pending", "running", "done")
