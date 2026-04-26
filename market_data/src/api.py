"""MarketData FastAPI application — /snapshot, /history, /indicators endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query

from market_data.src.config import MarketDataConfig
from market_data.src.indicators.macd import calculate_macd
from market_data.src.indicators.rsi import calculate_rsi
from market_data.src.sources.binance import BinanceSource
from shared.models.market import OHLCV, MarketSnapshot

from datetime import datetime, timezone


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = MarketDataConfig()
    app.state.source = BinanceSource(
        api_key=config.binance_api_key,
        api_secret=config.binance_api_secret,
    )
    app.state.config = config
    yield


app = FastAPI(title="MarketData", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/snapshot", response_model=MarketSnapshot)
async def snapshot(
    symbol: str = Query(..., description="Trading pair, e.g. BTCUSDT"),
    interval: str = Query("1d", description="Candle interval"),
) -> MarketSnapshot:
    source: BinanceSource = app.state.source
    try:
        candles = await source.fetch_ohlcv(symbol, interval, limit=100)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Binance error: {exc}") from exc

    if not candles:
        raise HTTPException(status_code=404, detail="No data returned from Binance")

    indicators: dict[str, Any] = {
        "rsi": calculate_rsi(candles),
        "macd": calculate_macd(candles),
    }
    return MarketSnapshot(
        symbol=symbol.upper(),
        timestamp=datetime.now(timezone.utc),
        ohlcv=candles[-1],
        indicators=indicators,
        source="binance",
    )


@app.get("/history", response_model=list[OHLCV])
async def history(
    symbol: str = Query(...),
    interval: str = Query("1d"),
    limit: int = Query(200, ge=1, le=1000),
) -> list[OHLCV]:
    source: BinanceSource = app.state.source
    try:
        return await source.fetch_ohlcv(symbol, interval, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Binance error: {exc}") from exc


@app.post("/indicators")
async def indicators(
    ohlcv: list[OHLCV],
    indicator_names: list[str],
) -> dict[str, Any]:
    supported = {"rsi": calculate_rsi, "macd": calculate_macd}
    unknown = [n for n in indicator_names if n not in supported]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown indicators: {unknown}")
    return {name: supported[name](ohlcv) for name in indicator_names}
