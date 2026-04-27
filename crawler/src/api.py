"""Crawler FastAPI application — serves articles from Supabase."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query

from crawler.src.config import CrawlerConfig
from crawler.src.db.supabase_reader import SupabaseReader
from shared.models.article import IngestionRecord


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = CrawlerConfig()
    app.state.reader = SupabaseReader(
        supabase_url=config.supabase_url,
        service_key=config.supabase_service_key,
        anon_key=config.supabase_anon_key,
    )
    app.state.config = config
    yield


app = FastAPI(title="Crawler", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/articles/latest", response_model=list[IngestionRecord])
async def get_latest(
    symbol: str = Query(..., description="Ticker symbol e.g. BTCUSDT"),
) -> list[IngestionRecord]:
    reader: SupabaseReader = app.state.reader
    config: CrawlerConfig = app.state.config
    try:
        return await reader.get_latest(symbol, limit=config.articles_limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
