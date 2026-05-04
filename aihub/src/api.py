"""AIHub FastAPI application — exposes /sentiment, /factors, /predict endpoints."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request

from aihub.src.sentiment.schema import SentimentRequest, SentimentResponse
from aihub.src.factors.schema import FactorRequest, FactorResponse
from aihub.src.predict.schema import PredictRequest, PredictResponse
from aihub.src.factors.extractor import FactorExtractor
from aihub.src.predict.rag_builder import RAGContextBuilder, StockMemClient
from aihub.src.config import AIHubConfig
from aihub.src.llm.base import LLMClient
from aihub.src.llm.models.factory import AIModelFactory
from aihub.src.llm.models.sentiment import SentimentModel
from shared.models.prediction import SignalType

logger = logging.getLogger(__name__)


def _predict_error_response(exc: Exception) -> PredictResponse:
    """Map exceptions to HOLD + actionable copy (network vs keys vs upstream)."""
    raw = str(exc).strip()
    lowered = raw.lower()
    detail = raw[:500] if raw else type(exc).__name__

    if isinstance(exc, httpx.RequestError) or "connection error" in lowered or (
        "failed to establish" in lowered
    ):
        explanation = (
            "AIHub cannot reach the Groq/OpenAI API over the internet (DNS, firewall, proxy, "
            "or outage). Technical detail: "
            f"{detail}. "
            "Docker: confirm the ``aihub`` container has outbound HTTPS; VPN/corporate "
            "networks often block ``api.groq.com``. Try another network or allowlist the endpoint."
        )
        steps = [
            "Kiểm tra: docker compose ps — service ``aihub`` phải Up; máy chủ/host có Internet?",
            "Từ container ``aihub``: thử wget/curl tới ``https://api.groq.com`` (timeout = mạng bị chặn).",
            "Tắt VPN/mạng công ty thử lại hoặc allowlist ``api.groq.com``. MainController chỉ gọi AIHub; AIHub mới gọi Groq.",
        ]
    else:
        explanation = (
            "Forecast model call failed — check AIHub logs, AIHUB_* API keys in .env, "
            f"and StockMem connectivity. Details: {detail}"
        )
        steps = [
            "Set a valid LLM API key (e.g. AIHUB_GROQ_API_KEY) and rebuild/restart AIHub.",
        ]

    return PredictResponse(
        signal=SignalType.HOLD,
        confidence=0.0,
        explanation=explanation[:2000],
        reasoning_steps=steps,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load config and heavy models once at startup; clean up on shutdown."""
    config = AIHubConfig()
    factory = AIModelFactory(config)

    app.state.config = config
    app.state.predict_client = factory.get_client(
        factory.resolve_backend(config.predict_llm_backend)
    )
    stockmem_client = StockMemClient(base_url=config.stockmem_url)
    app.state.rag_builder = RAGContextBuilder(stockmem_client=stockmem_client)
    app.state.sentiment_model = factory.create_sentiment_model()
    app.state.factor_extractor = FactorExtractor(factory.get_default_client())
    backends_in_use = {
        factory.resolve_backend(""),
        factory.resolve_backend(config.predict_llm_backend),
    }
    for bk in backends_in_use:
        if bk == "groq" and not (config.groq_api_key or "").strip():
            logger.warning(
                "AIHub LLM backend is groq but AIHUB_GROQ_API_KEY is empty — "
                "set it in .env or set AIHUB_LLM_BACKEND to a configured provider."
            )
        elif bk == "openai" and not (config.openai_api_key or "").strip():
            logger.warning(
                "AIHub uses openai but AIHUB_OPENAI_API_KEY is empty — "
                "set it or switch AIHUB_LLM_BACKEND."
            )
        elif bk == "gemini" and not (config.gemini_api_key or "").strip():
            logger.warning(
                "AIHub uses gemini but AIHUB_GEMINI_API_KEY is empty — "
                "set it or switch AIHUB_LLM_BACKEND."
            )
    yield
    # (teardown if needed)


app = FastAPI(
    title="AIHub",
    description="AI inference service for crypto pipeline",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict: 
    return {"status": "ok", "models_loaded": True}


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(request: SentimentRequest, http: Request) -> SentimentResponse:
    """Analyse sentiment of the provided text using CryptoBert."""
    model: SentimentModel = http.app.state.sentiment_model
    result = await model.run(request.text)
    return SentimentResponse(score=result.score, label=result.label)


@app.post("/factors", response_model=FactorResponse)
async def factors(request: FactorRequest, http: Request) -> FactorResponse:
    """Extract market factors from the provided text using SKGP + LLM."""
    extractor: FactorExtractor = http.app.state.factor_extractor
    try:
        out = await extractor.extract(request.ticker, request.text)
        return FactorResponse(factors=out)
    except Exception as exc:
        logger.exception("factors endpoint failed: %s", exc)
        return FactorResponse(factors=[])


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, http: Request) -> PredictResponse:
    """Generate a trading signal using RAG with similar historical cases."""
    predict_llm: LLMClient = http.app.state.predict_client
    rag_builder: RAGContextBuilder = http.app.state.rag_builder

    from aihub.src.predict.client import PredictClient

    predict_client = PredictClient(llm=predict_llm)

    try:
        current_text, similar_text = await rag_builder.build(
            request.current, request.similar or None
        )
        result = await predict_client.generate(current_text, similar_text)
        sig_raw = str(result.get("signal", "HOLD")).upper().strip()
        try:
            signal = SignalType(sig_raw)
        except ValueError:
            signal = SignalType.HOLD

        conf = float(result.get("confidence", 0))
        conf = max(0.0, min(1.0, conf))

        return PredictResponse(
            signal=signal,
            confidence=conf,
            explanation=result.get("explanation", ""),
            reasoning_steps=list(result.get("reasoning_steps", [])),
        )
    except Exception as exc:
        logger.exception("predict endpoint failed: %s", exc)
        return _predict_error_response(exc)
