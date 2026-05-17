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


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pct_change(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / old * 100.0


def _apply_post_signal_rules(
    signal: SignalType, conf: float, request: PredictRequest, config: AIHubConfig
) -> tuple[SignalType, list[str]]:
    """Apply deterministic guardrails after LLM signal generation.

    Rules:
    - Block SELL when 30d trend is strongly bullish unless short-term breaks down.
    - Block HOLD when 3d momentum is clearly directional.
    """
    if not config.post_rule_enabled:
        return signal, []

    notes: list[str] = []

    snapshot = request.current.market_snapshot
    indicators = snapshot.indicators or {}
    candles = list(snapshot.recent_candles or [])

    macd_hist = _safe_float(indicators.get("macd_hist"))
    close_now = _safe_float(getattr(snapshot.ohlcv, "close", None))

    close_3d = _safe_float(getattr(candles[-4], "close", None)) if len(candles) >= 4 else None
    close_30d = _safe_float(getattr(candles[-20], "close", None)) if len(candles) >= 20 else None

    ret_3d = _pct_change(close_3d, close_now)
    ret_30d = _pct_change(close_30d, close_now)

    bullish_30d = ret_30d is not None and ret_30d >= config.post_rule_bull_30d_pct
    bearish_30d = ret_30d is not None and ret_30d <= config.post_rule_bear_30d_pct
    clear_up_3d = ret_3d is not None and ret_3d >= config.post_rule_up_3d_pct
    clear_down_3d = ret_3d is not None and ret_3d <= config.post_rule_down_3d_pct
    macd_buy_ok = macd_hist is not None and macd_hist >= config.post_rule_macd_confirm_eps
    macd_sell_ok = macd_hist is not None and macd_hist <= -config.post_rule_macd_confirm_eps

    if signal == SignalType.SELL and bullish_30d and not (clear_down_3d and macd_sell_ok):
        signal = SignalType.HOLD
        notes.append(
            "post-rule: blocked SELL because 30d trend is strongly bullish without confirmed 3d+MACD breakdown."
        )

    if signal == SignalType.HOLD and conf <= config.post_rule_hold_override_max_conf:
        if clear_up_3d and not bearish_30d and macd_buy_ok:
            signal = SignalType.BUY
            notes.append(
                "post-rule: upgraded HOLD -> BUY because 3d momentum+MACD confirms upside and 30d regime is not bearish."
            )
        elif clear_down_3d and not bullish_30d and macd_sell_ok:
            signal = SignalType.SELL
            notes.append(
                "post-rule: downgraded HOLD -> SELL because 3d momentum+MACD confirms downside and 30d regime is not bullish."
            )

    return signal, notes


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
    config: AIHubConfig = http.app.state.config

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

        signal, post_rule_notes = _apply_post_signal_rules(signal, conf, request, config)
        reasoning_steps = list(result.get("reasoning_steps", []))
        if post_rule_notes:
            reasoning_steps.extend(post_rule_notes)

        # kNN confirmation: veto directional signal when similar-case outcomes contradict it.
        # Also suppress SELL when no kNN return data is available (cannot confirm).
        knn_threshold = config.knn_confirm_threshold
        if knn_threshold > 0.0 and request.similar:
            sim7_vals = [
                c.record.future_return_7d
                for c in request.similar
                if c.record.future_return_7d is not None
            ]
            if not sim7_vals and signal == SignalType.SELL:
                signal = SignalType.HOLD
                reasoning_steps.append("post-rule: knn_no_data_suppress_sell")
            elif sim7_vals:
                avg7 = sum(sim7_vals) / len(sim7_vals)
                if signal == SignalType.SELL and avg7 > knn_threshold:
                    signal = SignalType.HOLD
                    reasoning_steps.append("post-rule: knn_veto_sell (similar cases bullish)")
                elif signal == SignalType.BUY and avg7 < -knn_threshold:
                    signal = SignalType.HOLD
                    reasoning_steps.append("post-rule: knn_veto_buy (similar cases bearish)")

        return PredictResponse(
            signal=signal,
            confidence=conf,
            explanation=result.get("explanation", ""),
            reasoning_steps=reasoning_steps,
        )
    except Exception as exc:
        logger.exception("predict endpoint failed: %s", exc)
        return _predict_error_response(exc)
