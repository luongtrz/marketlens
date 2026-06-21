from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .config import settings
from .models import (
    HealthResponse,
    RecordCreateRequest,
    RecordCreateResponse,
    SearchRequest,
    SearchResponse,
    StockMemRecord,
)
from .service import StockMemService

logger = logging.getLogger(__name__)


class UpdateReturnsRequest(BaseModel):
    future_return_1d: Optional[float] = None
    future_return_3d: Optional[float] = None
    future_return_7d: Optional[float] = None
    future_return_15d: Optional[float] = None
    future_return_30d: Optional[float] = None


service = StockMemService(
    db_url=settings.db_url,
    vector_backend=settings.vector_backend,
    weights=settings.weights,
    learned_retriever_file=settings.learned_retriever_file,
)
_auto_task: asyncio.Task | None = None
_retrain_lock = asyncio.Lock()
_dist_lock_name = "stockmem:auto-retrain"


async def _run_retrain_once() -> dict:
    async with _retrain_lock:
        acquired = await service.repository.acquire_distributed_lock(_dist_lock_name)
        if not acquired:
            raise RuntimeError("distributed_lock_not_acquired")
        try:
            return await service.auto_retrain_weights(
                horizon=settings.auto_optimize_horizon,
                k=settings.auto_optimize_k,
                warmup=settings.auto_optimize_warmup,
                trials=settings.auto_optimize_trials,
                min_records=settings.auto_optimize_min_records,
                output_path=settings.auto_optimize_output,
            )
        finally:
            await service.repository.release_distributed_lock(_dist_lock_name)


async def _auto_optimize_loop() -> None:
    logger.info(
        "StockMem auto-optimize loop enabled: daily %02d:%02d UTC",
        settings.auto_optimize_hour_utc,
        settings.auto_optimize_minute_utc,
    )
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(
            hour=settings.auto_optimize_hour_utc,
            minute=settings.auto_optimize_minute_utc,
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        try:
            payload = await _run_retrain_once()
            logger.info(
                "StockMem auto-optimize done: n=%s combined=%.4f",
                payload.get("n_records"),
                float(payload.get("metrics", {}).get("combined", 0.0)),
            )
        except Exception as exc:
            if str(exc) == "distributed_lock_not_acquired":
                logger.info("StockMem auto-optimize skipped: lock held by another replica")
            else:
                logger.warning("StockMem auto-optimize failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _auto_task
    await service.startup()
    if settings.auto_optimize_enabled:
        _auto_task = asyncio.create_task(_auto_optimize_loop())
    try:
        yield
    finally:
        if _auto_task is not None:
            _auto_task.cancel()
            await asyncio.gather(_auto_task, return_exceptions=True)
            _auto_task = None
        await service.repository.close()


app = FastAPI(title="StockMem", version="0.1.0", lifespan=lifespan)


@app.post("/record", response_model=RecordCreateResponse)
async def create_record(payload: RecordCreateRequest) -> RecordCreateResponse:
    rid = await service.save_record(payload.record)
    return RecordCreateResponse(id=rid)


@app.get("/record/{record_id}", response_model=StockMemRecord)
async def get_record(record_id: str) -> StockMemRecord:
    record = await service.get_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    return record


@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    k = max(1, payload.k)
    results = await service.search(
        payload.query,
        k=k,
        before_date=payload.before_date,
        retriever_type=payload.retriever_type,
    )
    return SearchResponse(results=results)


@app.get("/records/missing-returns", response_model=list[StockMemRecord])
async def missing_returns(symbol: str | None = Query(default=None)) -> list[StockMemRecord]:
    return await service.list_missing_returns(symbol)


@app.patch("/record/{record_id}/returns")
async def update_returns(record_id: str, payload: UpdateReturnsRequest) -> dict:
    ok = await service.update_future_returns(
        record_id,
        future_return_1d=payload.future_return_1d,
        future_return_3d=payload.future_return_3d,
        future_return_7d=payload.future_return_7d,
        future_return_15d=payload.future_return_15d,
        future_return_30d=payload.future_return_30d,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="record not found")
    return {"updated": record_id}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_backend=service.vector_backend,
        db_url=settings.db_url,
    )


@app.post("/weights/retrain")
async def retrain_weights_now() -> dict:
    """Trigger Bayesian weight retraining immediately."""
    try:
        payload = await _run_retrain_once()
    except RuntimeError as exc:
        if str(exc) == "distributed_lock_not_acquired":
            raise HTTPException(status_code=409, detail="retrain already running on another replica")
        raise
    return {
        "status": "ok",
        "weights": payload.get("weights"),
        "metrics": payload.get("metrics"),
        "optimized_at": payload.get("optimized_at"),
        "output_path": settings.auto_optimize_output,
    }
