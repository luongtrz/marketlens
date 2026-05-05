"""Individual pipeline step functions.

Data-flow summary:
  Step 1 COLLECT  (parallel): crawler.get_latest [REQUIRED] + market.get_snapshot [REQUIRED]
  Step 2 AI SCORE (parallel): aihub.sentiment + aihub.factors → factorledge.ingest [graceful]
  Step 3 STOCKMEM (sequential, REQUIRED): stockmem.save → stockmem.search(k_similar, default 3)
  Step 4 PREDICT  (REQUIRED): aihub.predict(current, similar)
"""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

from main_controller.src.orchestrator.context import PipelineContext
from main_controller.src.orchestrator.exceptions import PipelineError
from shared.models.market import MarketSnapshot
from shared.models.memory import StockMemRecord


def _headline(article) -> str:
    """Return a usable headline for sentiment input.

    Why: upstream sometimes stores the URL in article_name. Sentiment over a
    raw URL collapses to ~0; extracting the slug recovers a headline-like signal.
    """
    name = (getattr(article, "article_name", "") or "").strip()
    if name.startswith("http"):
        path = urlparse(name).path.rstrip("/")
        slug = path.rsplit("/", 1)[-1] if path else ""
        return slug.replace("-", " ").replace("_", " ")
    return name

logger = logging.getLogger(__name__)


class ModuleClients:
    """Container for all module client instances (injected via DI)."""

    def __init__(
        self,
        crawler=None,
        aihub=None,
        market=None,
        stockmem=None,
        factorledge=None,
    ) -> None:
        self.crawler = crawler
        self.aihub = aihub
        self.market = market
        self.stockmem = stockmem
        self.factorledge = factorledge


async def step_collect(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 1: Collect articles and market snapshot in parallel — both REQUIRED.

    In historical mode (ctx.as_of_date set), fetches only data available on that
    date — no look-ahead. Articles are filtered to that day; OHLCV is fetched via
    /history with end_time so future candles are excluded.
    """
    if ctx.as_of_date is not None:
        target = ctx.as_of_date
        day_start = datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        end_ts = int(day_end.timestamp() * 1000)

        articles_task = clients.crawler.get_latest(
            ctx.symbol,
            publish_gte=day_start,
            publish_lte=day_end,
        )
        history_task = clients.market.get_history(
            ctx.symbol, interval="1d", limit=5, end_time=str(end_ts)
        )
        results = await asyncio.gather(articles_task, history_task, return_exceptions=True)

        if isinstance(results[0], Exception):
            raise PipelineError(f"Crawler failed: {results[0]}")
        ctx.latest_articles = results[0]

        if isinstance(results[1], Exception):
            raise PipelineError(f"MarketData history failed: {results[1]}")
        candles = results[1]
        # Pick the candle whose date matches target_date (last one ≤ target)
        match = next(
            (c for c in reversed(candles) if c.timestamp.date() <= target),
            candles[-1] if candles else None,
        )
        if match is None:
            raise PipelineError(f"No OHLCV candle found for {target}")
        ctx.market_snapshot = MarketSnapshot(
            symbol=ctx.symbol,
            timestamp=match.timestamp,
            ohlcv=match,
            recent_candles=candles,
            indicators={},
            source="binance",
        )
    else:
        results = await asyncio.gather(
            clients.crawler.get_latest(ctx.symbol),
            clients.market.get_snapshot(ctx.symbol),
            return_exceptions=True,
        )
        if isinstance(results[0], Exception):
            raise PipelineError(f"Crawler failed: {results[0]}")
        ctx.latest_articles = results[0]

        if isinstance(results[1], Exception):
            raise PipelineError(f"MarketData failed: {results[1]}")
        ctx.market_snapshot = results[1]


async def step_ai_score(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 2: Derive sentiment from pre-scored articles, call AIHub /factors."""
    # Sentiment comes directly from Supabase-scored articles — no extra LLM call needed.
    scored = [a.sentiment_score for a in ctx.latest_articles if a.sentiment_score != 0.0]
    avg_score = sum(scored) / len(scored) if scored else 0.0
    ctx.sentiment_score = max(-1.0, min(1.0, avg_score))
    if ctx.sentiment_score > 0.15:
        ctx.sentiment_label = "bullish"
    elif ctx.sentiment_score < -0.15:
        ctx.sentiment_label = "bearish"
    else:
        ctx.sentiment_label = "neutral"
    logger.info("Sentiment from %d articles: score=%.3f label=%s", len(scored), ctx.sentiment_score, ctx.sentiment_label)

    combined_text = "\n".join(
        h for a in ctx.latest_articles if (h := _headline(a))
    )[:3500]

    factors_result = await asyncio.gather(
        clients.aihub.factors(combined_text, ticker=ctx.symbol),
        return_exceptions=True,
    )
    factors_result = factors_result[0]

    raw_factors = []
    if isinstance(factors_result, Exception):
        ctx.errors.append(f"AIHub factors failed: {factors_result}")
        logger.warning("Factors degraded to []: %s", factors_result)
    else:
        raw_factors = factors_result

    try:
        ctx.factors = await clients.factorledge.ingest(
            article_id=f"batch_{ctx.run_id}",
            factors=[f.name for f in raw_factors],
            source="aihub",
        )
    except Exception as exc:
        ctx.errors.append(f"FactorLedge failed: {exc}")
        logger.warning("FactorLedge unavailable, factors=[]: %s", exc)
        ctx.factors = []


def _short_headlines_for_summary(articles: list) -> str:
    """Avoid multi-kB URL titles in StockMem summaries (bloats Groq input)."""
    parts: list[str] = []
    for a in articles[:3]:
        h = _headline(a)
        if not h:
            continue
        if len(h) > 220:
            h = h[:219].rstrip() + "…"
        parts.append(h)
    return " ".join(parts)


async def step_stockmem(
    ctx: PipelineContext, clients: ModuleClients, k_similar: int = 3
) -> None:
    """STEP 3: Save current record to StockMem, retrieve k nearest past records."""
    if ctx.market_snapshot is None:
        raise PipelineError("market_snapshot is None — cannot build StockMem record")

    current_record = StockMemRecord(
        date=ctx.as_of_date or date.today(),
        symbol=ctx.symbol,
        sentiment_score=ctx.sentiment_score or 0.0,
        sentiment_label=ctx.sentiment_label or "neutral",
        factors=[f.name for f in ctx.factors],
        normalized_factors=ctx.factors,
        market_snapshot=ctx.market_snapshot,
        indicator_vec=[],
        summary=_short_headlines_for_summary(ctx.latest_articles),
        article_ids=[a.id for a in ctx.latest_articles],
        run_id=str(ctx.run_id),
    )
    ctx.current_record = current_record

    try:
        ctx.current_record_id = await clients.stockmem.save(current_record)
        ctx.similar_records = await clients.stockmem.search(query=current_record, k=k_similar)
    except Exception as exc:
        raise PipelineError(f"StockMem failed: {exc}") from exc


async def step_predict(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 4: Call AIHub /predict with current context + similar cases (RAG)."""
    if ctx.current_record is None or ctx.market_snapshot is None:
        raise PipelineError("current_record or market_snapshot is None — cannot predict")

    try:
        ctx.prediction = await clients.aihub.predict(
            current=ctx.current_record,
            similar=ctx.similar_records,
        )
    except Exception as exc:
        raise PipelineError(f"AIHub predict failed: {exc}") from exc
