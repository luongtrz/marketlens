"""MainController FastAPI application — /run, /status, /result endpoints."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main_controller.src.clients.aihub_client import AIHubClient
from main_controller.src.clients.crawler_client import CrawlerClient
from main_controller.src.clients.factorledge_client import FactorLedgeClient
from main_controller.src.clients.market_client import MarketClient
from main_controller.src.clients.stockmem_client import StockMemClient
from main_controller.src.config import MainControllerConfig
from main_controller.src.orchestrator.pipeline import Pipeline, PipelineConfig
from main_controller.src.orchestrator.steps import ModuleClients
from shared.models.prediction import PredictionResult

logger = logging.getLogger(__name__)


class RunState(BaseModel):
    run_id: str
    symbol: str
    status: Literal["pending", "running", "done", "failed"] = "pending"
    started_at: datetime
    finished_at: datetime | None = None
    result: PredictionResult | None = None
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = MainControllerConfig()
    clients = ModuleClients(
        crawler=CrawlerClient(config.crawler_url),
        aihub=AIHubClient(config.aihub_url),
        market=MarketClient(config.market_data_url),
        stockmem=StockMemClient(config.stockmem_url),
        factorledge=FactorLedgeClient(config.factorledge_url),
    )
    app.state.pipeline = Pipeline(clients, PipelineConfig(k_similar=config.k_similar))
    app.state.run_states: dict[str, RunState] = {}
    app.state.background_tasks: set[asyncio.Task] = set()
    yield
    if app.state.background_tasks:
        await asyncio.gather(*app.state.background_tasks, return_exceptions=True)


app = FastAPI(title="MainController", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
async def run(symbol: str, trigger: str = "manual") -> dict:
    run_id = str(uuid4())
    state = RunState(run_id=run_id, symbol=symbol, started_at=datetime.now(timezone.utc))
    app.state.run_states[run_id] = state

    task = asyncio.create_task(
        _execute_run(run_id, symbol, app.state.pipeline, app.state.run_states)
    )
    app.state.background_tasks.add(task)
    task.add_done_callback(app.state.background_tasks.discard)

    return {"run_id": run_id, "status": "pending"}


@app.get("/status/{run_id}")
async def status(run_id: str) -> dict:
    state = app.state.run_states.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": state.run_id,
        "symbol": state.symbol,
        "status": state.status,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "has_result": state.result is not None,
    }


@app.get("/result/{run_id}", response_model=PredictionResult)
async def result(run_id: str) -> PredictionResult:
    state = app.state.run_states.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    if state.status != "done":
        raise HTTPException(status_code=425, detail=f"run status is '{state.status}'")
    return state.result  # type: ignore[return-value]


async def _execute_run(
    run_id: str,
    symbol: str,
    pipeline: Pipeline,
    run_states: dict[str, RunState],
) -> None:
    state = run_states[run_id]
    state.status = "running"
    try:
        result = await pipeline.run(symbol, run_id=UUID(run_id))
        state.result = result
        state.status = "done"
    except Exception as exc:
        logger.exception("Pipeline run %s failed: %s", run_id, exc)
        state.error = str(exc)
        state.status = "failed"
    finally:
        state.finished_at = datetime.now(timezone.utc)
