from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

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


service = StockMemService(
    db_url=settings.db_url,
    vector_backend=settings.vector_backend,
    weights=settings.weights,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await service.startup()
    yield


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
    results = await service.search(payload.query, k=k)
    return SearchResponse(results=results)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_backend=service.vector_backend,
        db_url=settings.db_url,
    )
