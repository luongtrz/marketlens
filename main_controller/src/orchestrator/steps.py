"""Individual pipeline step functions.

Data-flow summary:
  Step 1 COLLECT  (parallel): crawler.get_latest [REQUIRED] + market.get_snapshot [REQUIRED]
  Step 2 AI SCORE (parallel): aihub.sentiment + aihub.factors → factorledge.ingest [graceful]
  Step 3 STOCKMEM (sequential, REQUIRED): stockmem.save → stockmem.search(k=5)
  Step 4 PREDICT  (REQUIRED): aihub.predict(current, similar)
"""

import asyncio
import logging
from datetime import date
from urllib.parse import urlparse

from main_controller.src.orchestrator.context import PipelineContext
from main_controller.src.orchestrator.exceptions import PipelineError
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
    """STEP 1: Collect articles and market snapshot in parallel — both REQUIRED."""
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
    """STEP 2: Call AIHub /sentiment and /factors, attach results to ctx."""
    # Use article headlines as primary signal — raw_text from DB often contains
    # website boilerplate rather than actual article body text.
    combined_text = "\n".join(
        h for a in ctx.latest_articles if (h := _headline(a))
    )[:5000]

    sentiment_result, factors_result = await asyncio.gather(
        clients.aihub.sentiment(combined_text),
        clients.aihub.factors(combined_text, ticker=ctx.symbol),
        return_exceptions=True,
    )

    if isinstance(sentiment_result, Exception):
        ctx.errors.append(f"AIHub sentiment failed: {sentiment_result}")
        logger.warning("Sentiment degraded to 0.0: %s", sentiment_result)
        ctx.sentiment_score = 0.0
        ctx.sentiment_label = "neutral"
    else:
        ctx.sentiment_score = sentiment_result.get("score", 0.0)
        ctx.sentiment_label = sentiment_result.get("label", "neutral")

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


async def step_stockmem(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 3: Save current record to StockMem, retrieve k=5 most similar past records."""
    if ctx.market_snapshot is None:
        raise PipelineError("market_snapshot is None — cannot build StockMem record")

    current_record = StockMemRecord(
        date=date.today(),
        symbol=ctx.symbol,
        sentiment_score=ctx.sentiment_score or 0.0,
        sentiment_label=ctx.sentiment_label or "neutral",
        factors=[f.name for f in ctx.factors],
        normalized_factors=ctx.factors,
        market_snapshot=ctx.market_snapshot,
        indicator_vec=[],
        summary=" ".join(a.article_name for a in ctx.latest_articles[:3]),
        article_ids=[a.id for a in ctx.latest_articles],
        run_id=str(ctx.run_id),
    )
    ctx.current_record = current_record

    try:
        ctx.current_record_id = await clients.stockmem.save(current_record)
        ctx.similar_records = await clients.stockmem.search(query=current_record, k=5)
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
