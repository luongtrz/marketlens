"""FastAPI application for the LLM gateway."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import ValidationError

from llm_gateway.src.client import LLMGatewayError, OpenCodeGoClient
from llm_gateway.src.config import LLMGatewayConfig
from llm_gateway.src.schema import LLMDecisionRequest, LLMDecisionResponse


async def _read_decision_request(request: Request) -> LLMDecisionRequest:
    """Accept either JSON {"prompt": "..."} or a raw string/text body."""
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        payload: Any = await request.json()
        if isinstance(payload, str):
            return LLMDecisionRequest(prompt=payload)
        if isinstance(payload, dict):
            return LLMDecisionRequest.model_validate(payload)
        raise ValueError("JSON body must be a string or object with a prompt field")

    body = (await request.body()).decode("utf-8").strip()
    return LLMDecisionRequest(prompt=body)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = LLMGatewayConfig()
    app.state.config = config
    app.state.client = OpenCodeGoClient(config)
    yield


app = FastAPI(
    title="LLM Gateway",
    description="OpenCode Go gateway for final crypto BUY/HOLD/SELL decisions",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    config: LLMGatewayConfig = request.app.state.config
    return {
        "status": "ok",
        "configured": bool(config.opencode_go_api_key.strip()),
        "default_model": config.default_model,
        "max_attempts": config.bounded_max_attempts,
    }


@app.get("/models")
async def models(request: Request) -> dict[str, Any]:
    client: OpenCodeGoClient = request.app.state.client
    try:
        return await client.list_models()
    except LLMGatewayError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert upstream diagnostics to HTTP.
        raise HTTPException(status_code=502, detail=f"OpenCode Go /models failed: {exc}") from exc


@app.post("/llm", response_model=LLMDecisionResponse, response_model_exclude_none=True)
async def decide(
    request: Request,
    model: str | None = Query(default=None, description="OpenCode Go model id"),
    reason: bool | None = Query(default=None, description="Include a short reason in the response"),
) -> LLMDecisionResponse:
    try:
        decision_request = await _read_decision_request(request)
    except (UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    client: OpenCodeGoClient = request.app.state.client
    include_reason = decision_request.include_reason if reason is None else reason
    try:
        response = await client.decide(
            prompt=decision_request.prompt,
            model=model,
            system=decision_request.system,
            include_reason=include_reason,
        )
    except LLMGatewayError as exc:
        status = 503 if "API key" in str(exc) else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return response
