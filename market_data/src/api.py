"""MarketData FastAPI application — /snapshot, /history, /indicators endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query

from market_data.src.config import MarketDataConfig
from market_data.src.indicators.macd import calculate_macd
from market_data.src.indicators.rsi import calculate_rsi
from market_data.src.indicators.sma import calculate_sma
from market_data.src.sources.binance import BinanceSource, BinanceSourceError
from market_data.src.sources.fear_greed import FearGreedSource, FearGreedSourceError
from shared.models.market import OHLCV, MarketSnapshot

from datetime import datetime, timezone


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = MarketDataConfig()
    app.state.binance = BinanceSource(
        api_key=config.binance_api_key,
        api_secret=config.binance_api_secret,
    )
    app.state.fear_greed = FearGreedSource()
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
    binance: BinanceSource = app.state.binance
    fear_greed: FearGreedSource = app.state.fear_greed

    try:
        candles = await binance.fetch_ohlcv(symbol, interval, limit=100)
    except BinanceSourceError as exc:
        raise HTTPException(status_code=500, detail=f"Binance error: {exc}") from exc

    if not candles:
        raise HTTPException(status_code=404, detail="No data returned from Binance")

    try:
        fg_value = await fear_greed.fetch()
    except FearGreedSourceError as exc:
        raise HTTPException(status_code=500, detail=f"Fear & Greed error: {exc}") from exc

    macd = calculate_macd(candles)
    price_change_pct = (
        (candles[-1].close - candles[-2].close) / candles[-2].close
        if len(candles) >= 2
        else 0.0
    )
    indicators: dict[str, Any] = {
        "rsi": calculate_rsi(candles),
        "macd_hist": macd["histogram"],
        "sma": calculate_sma(candles),
        "fear_greed_index": float(fg_value),
        "price_change_pct": round(price_change_pct, 6),
    }
    return MarketSnapshot(
        symbol=symbol.upper(),
        timestamp=datetime.now(timezone.utc),
        ohlcv=candles[-1],
        recent_candles=candles[-10:],
        indicators=indicators,
        source="binance",
    )


@app.get("/history", response_model=list[OHLCV])
async def history(
    symbol: str = Query(...),
    interval: str = Query("1d"),
    limit: int = Query(200, ge=1, le=1000),
) -> list[OHLCV]:
    binance: BinanceSource = app.state.binance
    try:
        return await binance.fetch_ohlcv(symbol, interval, limit=limit)
    except BinanceSourceError as exc:
        raise HTTPException(status_code=500, detail=f"Binance error: {exc}") from exc


@app.post("/indicators")
async def indicators(
    ohlcv: list[OHLCV],
    indicator_names: list[str],
) -> dict[str, Any]:
    supported = {"rsi": calculate_rsi, "macd": calculate_macd, "sma": calculate_sma}
    unknown = [n for n in indicator_names if n not in supported]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown indicators: {unknown}")
    return {name: supported[name](ohlcv) for name in indicator_names}
