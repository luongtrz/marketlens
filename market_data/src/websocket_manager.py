"""WebSocket manager for real-time Binance market data streaming."""

import asyncio
import json
import logging
from typing import Any

import websockets
from fastapi import WebSocket

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"


class BinanceWebSocketManager:
    """Manages WebSocket connections to Binance and proxies data to frontend clients.

    This manager:
    - Maintains a single upstream connection to Binance WebSocket
    - Tracks client subscriptions by symbol and stream type
    - Proxies real-time kline and trade data to connected frontend clients
    - Handles reconnection on connection loss
    """

    def __init__(self) -> None:
        self._binance_ws: websockets.WebSocketClientProtocol | None = None
        self._clients: dict[str, WebSocket] = {}  # client_id -> WebSocket
        self._client_subscriptions: dict[str, set[str]] = {}  # client_id -> {stream_names}
        self._subscription_counts: dict[str, int] = {}  # stream_name -> client_count
        self._active_streams: set[str] = set()  # Currently subscribed Binance streams
        self._running = False
        self._receive_task: asyncio.Task | None = None
        self._reconnect_delay = 5.0  # seconds

    async def start(self) -> None:
        """Start the WebSocket manager and connect to Binance."""
        self._running = True
        await self._connect_to_binance()

    async def stop(self) -> None:
        """Stop the WebSocket manager and close all connections."""
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._binance_ws:
            await self._binance_ws.close()
        logger.info("WebSocket manager stopped")

    async def _connect_to_binance(self) -> None:
        """Establish connection to Binance WebSocket."""
        try:
            self._binance_ws = await websockets.connect(
                BINANCE_WS_URL,
                ping_interval=30,
                ping_timeout=10,
            )
            logger.info("Connected to Binance WebSocket")

            # Resubscribe to active streams if reconnecting
            if self._active_streams:
                await self._send_binance_action(
                    "SUBSCRIBE", list(self._active_streams)
                )

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            if self._running:
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                await self._connect_to_binance()

    async def _receive_loop(self) -> None:
        """Listen to Binance WebSocket and forward messages to clients."""
        if not self._binance_ws:
            return

        try:
            async for message in self._binance_ws:
                await self._handle_binance_message(message)
        except websockets.ConnectionClosed as e:
            logger.warning(f"Binance WebSocket closed: {e}")
            if self._running:
                logger.info("Reconnecting to Binance...")
                await self._connect_to_binance()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                await self._connect_to_binance()

    async def _handle_binance_message(self, message: str) -> None:
        """Parse Binance message and forward to subscribed clients."""
        try:
            data = json.loads(message)

            # Handle kline (candlestick) updates
            if data.get("e") == "kline":
                symbol = self._normalize_symbol(data.get("s", ""))
                kline_data = data.get("k", {})
                standardized = {
                    "symbol": symbol,
                    "time": kline_data.get("t"),
                    "open": float(kline_data.get("o", 0)),
                    "high": float(kline_data.get("h", 0)),
                    "low": float(kline_data.get("l", 0)),
                    "close": float(kline_data.get("c", 0)),
                    "volume": float(kline_data.get("v", 0)),
                    "isFinal": kline_data.get("x", False),
                }
                await self._broadcast_to_room(f"kline:{symbol}", "kline", standardized)

            # Handle trade updates
            elif data.get("e") == "trade":
                symbol = self._normalize_symbol(data.get("s", ""))
                await self._broadcast_to_room(f"trade:{symbol}", "trade", data)

        except Exception as e:
            logger.error(f"Error parsing Binance message: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol by removing USDT suffix."""
        if symbol.endswith("USDT"):
            return symbol[: -len("USDT")]
        return symbol

    async def _broadcast_to_room(
        self, room: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Broadcast data to all clients in a room."""
        disconnected = []
        for client_id, client_ws in self._clients.items():
            if room in self._client_subscriptions.get(client_id, set()):
                try:
                    await client_ws.send_json({"type": event_type, "data": data})
                except Exception:
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.remove_client(client_id)

    async def add_client(self, client_id: str, websocket: WebSocket) -> None:
        """Add a new client connection."""
        self._clients[client_id] = websocket
        self._client_subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected")

    async def remove_client(self, client_id: str) -> None:
        """Remove a client and clean up subscriptions."""
        if client_id not in self._clients:
            return

        # Remove all subscriptions for this client
        subscriptions = self._client_subscriptions.get(client_id, set())
        for stream_name in subscriptions:
            await self._unsubscribe_stream(client_id, stream_name)

        del self._clients[client_id]
        self._client_subscriptions.pop(client_id, None)
        logger.info(f"Client {client_id} disconnected")

    async def subscribe_to_room(
        self, client_id: str, symbol: str, stream_type: str
    ) -> None:
        """Subscribe a client to a specific room (e.g., kline:BTC, trade:ETH)."""
        symbol = symbol.upper()
        room = f"{stream_type}:{symbol}"

        if client_id not in self._clients:
            logger.warning(f"Client {client_id} not found")
            return

        # Track client subscription
        self._client_subscriptions.setdefault(client_id, set()).add(room)

        # Build Binance stream name
        pair = f"{symbol}USDT".lower()
        stream_name = f"{pair}@{stream_type}" if stream_type == "trade" else f"{pair}@kline_1m"

        # Subscribe to Binance if first client
        count = self._subscription_counts.get(stream_name, 0) + 1
        self._subscription_counts[stream_name] = count

        if count == 1:
            self._active_streams.add(stream_name)
            await self._send_binance_action("SUBSCRIBE", [stream_name])

        logger.info(f"Client {client_id} subscribed to {room}")

    async def unsubscribe_from_room(
        self, client_id: str, symbol: str, stream_type: str
    ) -> None:
        """Unsubscribe a client from a room."""
        symbol = symbol.upper()
        room = f"{stream_type}:{symbol}"

        if client_id not in self._client_subscriptions:
            return

        self._client_subscriptions[client_id].discard(room)

        # Build Binance stream name
        pair = f"{symbol}USDT".lower()
        stream_name = f"{pair}@{stream_type}" if stream_type == "trade" else f"{pair}@kline_1m"

        await self._unsubscribe_stream(client_id, stream_name)
        logger.info(f"Client {client_id} unsubscribed from {room}")

    async def _unsubscribe_stream(self, client_id: str, stream_name: str) -> None:
        """Unsubscribe from a Binance stream when no clients need it."""
        count = self._subscription_counts.get(stream_name, 0) - 1
        if count <= 0:
            self._subscription_counts.pop(stream_name, None)
            self._active_streams.discard(stream_name)
            await self._send_binance_action("UNSUBSCRIBE", [stream_name])
        else:
            self._subscription_counts[stream_name] = count

    async def _send_binance_action(
        self, method: str, params: list[str]
    ) -> None:
        """Send SUBSCRIBE/UNSUBSCRIBE action to Binance WebSocket."""
        if not self._binance_ws or not params:
            return

        try:
            payload = {
                "method": method,
                "params": params,
                "id": id(self),
            }
            await self._binance_ws.send(json.dumps(payload))
            logger.debug(f"Sent {method} to Binance: {params}")
        except Exception as e:
            logger.error(f"Failed to send {method} to Binance: {e}")


# Global singleton instance
websocket_manager = BinanceWebSocketManager()
