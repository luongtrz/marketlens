"""MarketData FastAPI application — /snapshot, /history, /indicators endpoints and WebSocket."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, Query

from market_data.src.config import MarketDataConfig
from market_data.src.indicators.macd import calculate_macd
from market_data.src.indicators.rsi import calculate_rsi
from market_data.src.indicators.sma import calculate_sma
from market_data.src.sources.binance import BinanceSource, BinanceSourceError
from market_data.src.sources.fear_greed import FearGreedSource, FearGreedSourceError
from shared.models.market import OHLCV, MarketSnapshot
from market_data.src.config import MarketDataConfig
from market_data.src.sources.binance import BinanceSource
from market_data.src.indicators.registry import calculate_indicators
from market_data.src.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global Binance source instance
_binance_source: BinanceSource | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize Binance source on startup."""
    global _binance_source
    config = MarketDataConfig()
    _binance_source = BinanceSource(
        api_key=config.binance_api_key,
        api_secret=config.binance_api_secret,
    )
    await websocket_manager.start()
    logger.info("MarketData service started")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    global _binance_source
    if _binance_source:
        await _binance_source.close()
    await websocket_manager.stop()
    logger.info("MarketData service stopped")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/symbols")
async def list_symbols() -> dict[str, list[str]]:
    """List all tracked symbols available for WebSocket streaming.

    Returns:
        Dict with "symbols" key containing list of symbol names.
    """
    config = MarketDataConfig()
    return {"symbols": config.tracked_symbols}


@app.get("/snapshot", response_model=MarketSnapshot)
async def snapshot(
    symbol: str = Query(..., description="Trading pair, e.g. BTCUSDT"),
    interval: str = Query("1d", description="Candle interval"),
) -> MarketSnapshot:
    """Get current market snapshot with computed indicators.

    Args:
        symbol: Trading pair.
        interval: Candle interval.

    Returns:
        MarketSnapshot with latest OHLCV and indicators.
    """
    if _binance_source is None:
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content={"error": "Service not initialized"},
        )

    # Binance expects pairs like BTCUSDT
    query_symbol = f"{symbol.upper()}USDT" if not symbol.upper().endswith("USDT") else symbol.upper()

    # Fetch latest candles
    candles = await _binance_source.fetch_ohlcv(query_symbol, interval, limit=50)
    if not candles:
        return JSONResponse(  # type: ignore[return-value]
            status_code=404,
            content={"error": f"No data found for {symbol}"},
        )

    latest_candle = candles[-1]
    config = MarketDataConfig()

    # Calculate indicators
    indicators = calculate_indicators(candles, config.default_indicators)

    return MarketSnapshot(
        symbol=symbol.upper(),
        timestamp=datetime.now(timezone.utc),
        ohlcv=latest_candle,
        indicators=indicators,
        source="binance",
    )


@app.get("/history", response_model=list[OHLCV])
async def history(
    symbol: str = Query(...),
    interval: str = Query("1h"),
    limit: int = Query(200),
    end_time: int | None = Query(None),
) -> list[OHLCV]:
    """Get historical OHLCV candles.

    Args:
        symbol: Trading pair.
        interval: Candle interval.
        limit: Maximum number of candles.

    Returns:
        List of OHLCV candles.
    """
    if _binance_source is None:
        return []

    query_symbol = f"{symbol.upper()}USDT" if not symbol.upper().endswith("USDT") else symbol.upper()
    candles = await _binance_source.fetch_ohlcv(query_symbol, interval, limit=limit, end_time=end_time)
    return candles


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
    try:
        return calculate_indicators(ohlcv, indicator_names)
    except KeyError as e:
        return JSONResponse(  # type: ignore[return-value]
            status_code=400,
            content={"error": f"Unknown indicator: {e}"},
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time market data.

    Clients can:
    - Subscribe to kline (candlestick) updates: {"action": "subscribe", "type": "kline", "symbol": "BTC"}
    - Subscribe to trade updates: {"action": "subscribe", "type": "trade", "symbol": "ETH"}
    - Unsubscribe: {"action": "unsubscribe", "type": "kline", "symbol": "BTC"}

    Server sends:
    - {"type": "kline", "data": {"symbol": "BTC", "time": ..., "open": ..., ...}}
    - {"type": "trade", "data": {...raw trade data...}}
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())
    config = MarketDataConfig()
    tracked = {s.upper() for s in config.tracked_symbols}

    await websocket_manager.add_client(client_id, websocket)

    try:
        while True:
            message = await websocket.receive_json()
            action = message.get("action")
            symbol = message.get("symbol", "").upper()
            stream_type = message.get("type", "kline")

            if not symbol:
                await websocket.send_json({"error": "Symbol is required"})
                continue

            if symbol not in tracked:
                await websocket.send_json({
                    "error": f"Symbol {symbol} not tracked. Available: {sorted(tracked)}"
                })
                continue

            if action == "subscribe":
                await websocket_manager.subscribe_to_room(
                    client_id, symbol, stream_type
                )
            elif action == "unsubscribe":
                await websocket_manager.unsubscribe_from_room(
                    client_id, symbol, stream_type
                )
            else:
                await websocket.send_json(
                    {"error": "Unknown action. Use 'subscribe' or 'unsubscribe'"}
                )

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket_manager.remove_client(client_id)
