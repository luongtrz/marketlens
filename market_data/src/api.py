"""MarketData FastAPI application — /snapshot, /history, /indicators endpoints."""

from typing import Any

from fastapi import FastAPI, Query

from shared.models.market import OHLCV, MarketSnapshot

app = FastAPI(title="MarketData", description="Market data and indicator service")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/snapshot", response_model=MarketSnapshot)
async def snapshot(
    symbol: str = Query(..., description="Trading pair, e.g. BTCUSDT"),
    interval: str = Query("1h", description="Candle interval"),
) -> MarketSnapshot:
    """Get current market snapshot with computed indicators.

    Args:
        symbol: Trading pair.
        interval: Candle interval.

    Returns:
        MarketSnapshot with latest OHLCV and indicators.
    """
    raise NotImplementedError


@app.get("/history", response_model=list[OHLCV])
async def history(
    symbol: str = Query(...),
    interval: str = Query("1h"),
    limit: int = Query(200),
) -> list[OHLCV]:
    """Get historical OHLCV candles.

    Args:
        symbol: Trading pair.
        interval: Candle interval.
        limit: Maximum number of candles.

    Returns:
        List of OHLCV candles.
    """
    raise NotImplementedError


@app.post("/indicators")
async def indicators(
    ohlcv: list[OHLCV],
    indicator_names: list[str],
) -> dict[str, Any]:
    """Calculate indicators from provided OHLCV data.

    Args:
        ohlcv: List of OHLCV candles.
        indicator_names: Indicator names to calculate (e.g. ["macd", "rsi"]).

    Returns:
        Dict of indicator name to result.
    """
    raise NotImplementedError
