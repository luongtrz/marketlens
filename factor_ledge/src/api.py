"""factor_ledge Python gateway — port 8004.

Acts as the single HTTP entry-point for the factor_ledge sub-system.
Routes requests to the three TypeScript micro-services running internally:

  POST /ledger/update     → ledger-service  :3002  POST /ledger/update
  GET  /ledger/current    → ledger-service  :3002  GET  /ledger/current
  GET  /ledger/snapshot   → ledger-service  :3002  GET  /ledger/snapshot
  GET  /query/vector      → query-service   :3003  GET  /query/vector
  GET  /query/factor-vector → query-service :3003  GET  /query/factor-vector
  GET  /query/context     → query-service   :3003  GET  /query/context
  GET  /query/top         → query-service   :3003  GET  /query/top
  POST /classify          → classify-service:3001  POST /classify
  POST /classify/batch    → classify-service:3001  POST /classify/batch
  POST /classify/vector   → classify-service:3001  POST /classify/vector  (75d for StockMem)
  GET  /health
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── Service URLs (override via env vars) ──────────────────────────────────────
CLASSIFY_URL  = os.getenv("CLASSIFY_URL",  "http://localhost:3001")
LEDGER_URL    = os.getenv("LEDGER_URL",    "http://localhost:3002")
QUERY_URL     = os.getenv("QUERY_URL",     "http://localhost:3003")

# Timeout in seconds for proxied requests
_TIMEOUT = float(os.getenv("PROXY_TIMEOUT", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http = httpx.AsyncClient(timeout=_TIMEOUT)
    yield
    await app.state.http.aclose()


app = FastAPI(
    title="FactorLedge Gateway",
    description="Python proxy / entry-point for the factor_ledge TypeScript services",
    lifespan=lifespan,
)


# ── Helper ────────────────────────────────────────────────────────────────────

async def _proxy(
    request: Request,
    target_url: str,
    *,
    method: str | None = None,
    body: Any = None,
) -> Response:
    """Forward the incoming request to *target_url* and stream the response back."""
    client: httpx.AsyncClient = request.app.state.http
    m = method or request.method

    try:
        if m == "GET":
            resp = await client.get(target_url, params=dict(request.query_params))
        else:
            payload = body if body is not None else await request.json()
            resp = await client.request(m, target_url, json=payload)
    except httpx.ConnectError as exc:
        logger.error("Proxy connect error → %s: %s", target_url, exc)
        return JSONResponse({"error": f"upstream unavailable: {target_url}"}, status_code=503)
    except Exception as exc:
        logger.error("Proxy error → %s: %s", target_url, exc)
        return JSONResponse({"error": str(exc)}, status_code=502)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "factor-ledge-gateway", "port": 8004}


# ── Ledger endpoints (→ ledger-service :3002) ─────────────────────────────────

@app.post("/ledger/update")
async def ledger_update(request: Request) -> Response:
    return await _proxy(request, f"{LEDGER_URL}/ledger/update")


@app.get("/ledger/current")
async def ledger_current(request: Request) -> Response:
    return await _proxy(request, f"{LEDGER_URL}/ledger/current")


@app.get("/ledger/snapshot")
async def ledger_snapshot(request: Request) -> Response:
    return await _proxy(request, f"{LEDGER_URL}/ledger/snapshot")


# ── Query endpoints (→ query-service :3003) ───────────────────────────────────

@app.get("/query/vector")
async def query_vector(request: Request) -> Response:
    return await _proxy(request, f"{QUERY_URL}/query/vector")


@app.get("/query/factor-vector")
async def query_factor_vector(request: Request) -> Response:
    return await _proxy(request, f"{QUERY_URL}/query/factor-vector")


@app.get("/query/context")
async def query_context(request: Request) -> Response:
    return await _proxy(request, f"{QUERY_URL}/query/context")


@app.get("/query/top")
async def query_top(request: Request) -> Response:
    return await _proxy(request, f"{QUERY_URL}/query/top")


# ── Classify endpoints (→ classify-service :3001) ────────────────────────────

@app.post("/classify")
async def classify_single(request: Request) -> Response:
    return await _proxy(request, f"{CLASSIFY_URL}/classify")


@app.post("/classify/batch")
async def classify_batch(request: Request) -> Response:
    return await _proxy(request, f"{CLASSIFY_URL}/classify/batch")


@app.post("/classify/vector")
async def classify_vector(request: Request) -> Response:
    return await _proxy(request, f"{CLASSIFY_URL}/classify/vector")
