"""REST routes expected by the Vite/React UI under ``/api/ai/*``.

The dashboard was written against an older monolithic gateway; MainController exposes
those paths here by delegating to the same ``Pipeline`` and module clients used by
``/run``. This keeps Docker + browser CORS setups simple without changing every page."""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from main_controller.src.orchestrator.pipeline import Pipeline
from main_controller.src.orchestrator.steps import ModuleClients
from shared.models.article import IngestionRecord
from shared.models.prediction import PredictionResult, SignalType
from shared.supabase_news import (
    count_news_articles_from_supabase,
    count_news_articles_matching_symbol_from_supabase,
)
from shared.supabase_service import SupabaseReadService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ui"])

# Legacy (no ``page`` query): bounded fetch when the SPA still filters/slices client-side.
_UI_NEWS_FETCH_LIMIT = 300


def _parse_iso_maybe(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _symbol_to_pair(tag: str) -> str:
    """Map UI symbols like ``BTC`` to ``BTCUSDT`` used by crawler + Binance snapshots."""
    t = tag.strip().upper()
    if not t:
        return ""
    if t.endswith("USDT"):
        return t
    if "/" in t:
        left, _, right = t.partition("/")
        return f"{left}{right}".replace("-", "")
    return f"{t}USDT"


def _infer_ui_tag(title: str, snippet: str) -> str:
    """Match News page filter options: BTC | ETH | General."""
    t = f"{title} {snippet}".lower()
    has_btc = "btc" in t or "bitcoin" in t
    has_eth = "ethereum" in t or re.search(r"\beth\b", t) is not None
    if has_btc and not has_eth:
        return "BTC"
    if has_eth and not has_btc:
        return "ETH"
    return "General"


def _ingestion_to_news_article(record: IngestionRecord, sentiment_score: int = 0) -> dict[str, Any]:
    label_raw = (record.sentiment_label or "").lower()
    sentiment: str
    if "bull" in label_raw:
        sentiment = "Positive"
    elif "bear" in label_raw:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    snippet = (
        record.summary
        if record.summary
        else (record.article_name[:280] + "…" if len(record.article_name) > 280 else record.article_name)
    )
    resolved = snippet or ""
    ui_tag = _infer_ui_tag(record.article_name or "", resolved)
    return {
        "id": record.id,
        "title": record.article_name,
        "source": record.source or "news",
        "timestamp": record.date_published.isoformat(),
        "snippet": resolved,
        "url": record.url,
        "sentiment": sentiment,
        "summary": record.summary,
        "sentimentScore": sentiment_score,
        "tag": ui_tag,
    }


def _records_to_news_payload(
    raw: list[IngestionRecord],
    *,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in raw:
        ts = rec.date_published.replace(tzinfo=timezone.utc) if rec.date_published.tzinfo is None else rec.date_published
        if start_dt and ts < start_dt:
            continue
        if end_dt and ts > end_dt:
            continue
        score = int(round((rec.sentiment_score + 1) * 50)) if hasattr(rec, "sentiment_score") else 50
        score = max(0, min(100, score))
        out.append(_ingestion_to_news_article(rec, sentiment_score=score))
    out.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return out


class LatestNewsPageResponse(BaseModel):
    """Paged ``/ai/latest-news`` JSON when ``page`` is present."""

    items: list[dict[str, Any]]
    page: int
    page_size: int
    total: int


class AnalyzeArticlePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    snippet: str = ""
    source: str = ""


class ForecastPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    coin_name: str = Field(validation_alias=AliasChoices("coinName", "coin_name"))
    recent_trend: str = Field(default="", validation_alias=AliasChoices("recentTrend", "recent_trend"))
    current_price: float = Field(ge=0, validation_alias=AliasChoices("currentPrice", "current_price"))


class ChartAskPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    coin_symbol: str = Field(default="", validation_alias=AliasChoices("coinSymbol", "coin_symbol"))
    chart_data: list[Any] = Field(default_factory=list, validation_alias=AliasChoices("chartData", "chart_data"))
    question: str = ""


class NewsAskPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    context_text: str = Field(default="", validation_alias=AliasChoices("contextText", "context_text"))
    question: str = ""


class ChatPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)


def _resolve_base_price(pred: PredictionResult, hinted: float) -> float:
    if math.isfinite(hinted) and hinted > 0:
        return float(hinted)
    ms = pred.market_snapshot
    if ms is not None:
        c = ms.ohlcv.close
        if math.isfinite(c) and c > 0:
            return float(c)
    return 0.0


def _prediction_to_forecast(pred: PredictionResult, base_price: float, trend_hint: str) -> dict[str, Any]:
    sig = pred.signal
    if sig == SignalType.BUY:
        trend_ui = "Bullish"
        action = "Buy"
    elif sig == SignalType.SELL:
        trend_ui = "Bearish"
        action = "Sell"
    else:
        trend_ui = "Neutral"
        action = "Hold"

    px = _resolve_base_price(pred, base_price)
    magnitude = pred.confidence * (px * 0.002 if px else 1.0)
    drift = magnitude if sig == SignalType.BUY else (-magnitude if sig == SignalType.SELL else 0)
    preds: list[float] = []
    if math.isfinite(px) and px > 0:
        step = drift / max(len(pred.reasoning_steps or [""]), 1)
        for i in range(5):
            preds.append(round(px + step * i, 8))
    else:
        preds = [0.0, 0.0, 0.0, 0.0, 0.0]

    conf_pct = round(min(99.9, pred.confidence * 100), 2)
    rec = {"action": action, "entryZone": "-", "targetPrice": "-", "stopLoss": "-"}

    reasoning = pred.explanation or ""
    summary_bits = "; ".join(x for x in pred.errors if x)

    parts = [reasoning]
    if trend_hint:
        parts.append(f"[Chart trend hint: {trend_hint}]")
    if summary_bits:
        parts.append(f"[Pipeline notes: {summary_bits}]")
    reasoning_full = "\n".join(parts).strip()

    return {
        "predictedPrices": preds,
        "confidenceScore": conf_pct,
        "reasoning": reasoning_full or "Forecast completed.",
        "trend": trend_ui,
        "marketSummary": pred.explanation[:500] + ("…" if len(pred.explanation) > 500 else ""),
        "recommendation": rec,
        "sources": [],
    }


@router.get("/ai/latest-news", response_model=None)
async def api_latest_news(
    request: Request,
    start: str | None = None,
    end: str | None = None,
    tag: str | None = None,
    page: int | None = Query(default=None, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> LatestNewsPageResponse | list[dict[str, Any]]:
    """Recent articles.

    Without ``page``: returns a legacy JSON **array** up to `_UI_NEWS_FETCH_LIMIT`.

    With ``page``: returns ``{items, page, page_size, total}`` using Supabase ``LIMIT/OFFSET``
    (symbol filters use a capped scan described in ``shared.supabase_news``).
    """
    clients: ModuleClients = request.app.state.clients
    start_dt = _parse_iso_maybe(start)
    end_dt = _parse_iso_maybe(end)

    pair = _symbol_to_pair(tag.strip()) if tag and tag.strip() else ""
    if SupabaseReadService.from_env() is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Supabase is not configured on MainController: set SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY (recommended) or SUPABASE_ANON_KEY with RLS "
                "allowing SELECT on news_articles. For Docker, add them to the project root .env."
            ),
        )
    try:
        if page is not None:
            offset = (page - 1) * page_size
            raw = await clients.crawler.get_latest(
                pair,
                limit=page_size,
                offset=offset,
                lite=True,
                publish_gte=start_dt,
                publish_lte=end_dt,
            )
            total: int = 0
            if pair:
                total = await count_news_articles_matching_symbol_from_supabase(
                    symbol=pair,
                    publish_gte=start_dt,
                    publish_lte=end_dt,
                )
            else:
                total_val = await count_news_articles_from_supabase(
                    publish_gte=start_dt,
                    publish_lte=end_dt,
                )
                total = int(total_val or 0)
            items = _records_to_news_payload(raw, start_dt=start_dt, end_dt=end_dt)
            return LatestNewsPageResponse(
                items=items,
                page=page,
                page_size=page_size,
                total=max(total, 0),
            )

        raw = await clients.crawler.get_latest(
            pair,
            limit=_UI_NEWS_FETCH_LIMIT,
            offset=0,
            lite=True,
            publish_gte=start_dt,
            publish_lte=end_dt,
        )
    except Exception as exc:
        logger.exception("latest-news crawler failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _records_to_news_payload(raw, start_dt=start_dt, end_dt=end_dt)


@router.post("/ai/analyze-article")
async def api_analyze_article(request: Request, body: AnalyzeArticlePayload) -> dict[str, Any]:
    clients: ModuleClients = request.app.state.clients
    text_parts = [body.title.strip(), body.snippet.strip()]
    blob = "\n".join(x for x in text_parts if x)
    if not blob:
        return {
            "sentiment": "Neutral",
            "summary": "No text supplied.",
            "sentimentScore": 0,
        }
    try:
        res = await clients.aihub.sentiment(blob)
        score = float(res.get("score", 0.0))
        label_raw = str(res.get("label", "neutral")).lower()
    except Exception as exc:
        logger.warning("analyze-article sentiment failed: %s", exc)
        return {
            "sentiment": "Neutral",
            "summary": f"Sentiment unavailable: {exc}"[:280],
            "sentimentScore": 0,
        }

    if score > 0.15:
        sent = "Positive"
    elif score < -0.15:
        sent = "Negative"
    else:
        sent = "Neutral"
    summary = blob[:280] + ("…" if len(blob) > 280 else "")
    sentiment_score = int(round((score + 1) * 50))
    sentiment_score = max(0, min(100, sentiment_score))
    return {"sentiment": sent, "summary": summary, "sentimentScore": sentiment_score}


@router.post("/ai/forecast")
async def api_forecast(request: Request, payload: ForecastPayload) -> dict[str, Any]:
    pipeline: Pipeline = request.app.state.pipeline
    pair = _symbol_to_pair(payload.coin_name)
    if not pair:
        raise HTTPException(status_code=400, detail="coinName cannot be empty")
    try:
        pred = await pipeline.run(pair)
        return _prediction_to_forecast(pred, payload.current_price, payload.recent_trend)
    except Exception as exc:
        logger.exception("forecast pipeline failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/ai/historical-news")
async def api_historical_news(
    request: Request,
    coinName: str = Query(..., alias="coinName"),
    date: str = Query(...),
) -> list[dict[str, Any]]:
    """Same as latest-news but narrowed to the UTC calendar day parsed from ``date``."""
    day = _parse_iso_maybe(date)
    if day is None:
        return []
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)
    return await api_latest_news(request, start=start.isoformat(), end=end.isoformat(), tag=coinName)


@router.post("/ai/ask-chart")
async def api_ask_chart(body: ChartAskPayload) -> str:
    return (
        f"[{body.coin_symbol or 'PAIR'} chart] {body.question.strip()[:200]}\n"
        "Structured chart Q&A requires a dedicated chart-LLM route; pipeline forecast "
        "and indicators are already available via the Forecast action."
    )


@router.post("/ai/ask-news")
async def api_ask_news(body: NewsAskPayload) -> str:
    snippet = body.context_text[:1200]
    q = body.question.strip() or "Summarize relevance."
    return (
        f"Context ({len(snippet)} chars): {snippet[:280]}"
        + ("…" if len(snippet) > 280 else "")
        + "\nQuestion: "
        + q
        + "\n\n(Open a fuller news assistant by wiring AskNewsPayload to AIHub)"
    )


@router.post("/ai/chat")
async def api_chat(body: ChatPayload) -> dict[str, Any]:
    msg = body.message.strip() or "(empty)"
    return {
        "text": (
            "Chat sandbox: connect AIHub Groq/OpenAI/Gemini keys for full assistants. "
            f"You wrote: «{msg[:200]}». "
            "Tip: Use «Market forecast» on the dashboard for the full pipeline prediction."
        ),
        "groundingMetadata": {},
    }
