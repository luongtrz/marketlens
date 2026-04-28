"""Individual pipeline step functions.

Data-flow summary:
  Step 1 COLLECT   (parallel)  : crawler.get_latest [REQUIRED] + market.get_snapshot [REQUIRED]
  Step 2 AI SCORE  (sequential): aggregate pre-computed sentiment from articles;
                                  aihub.factors(summaries) → factorledge.update_ledger [graceful]
  Step 3 STOCKMEM  (sequential, REQUIRED): stockmem.save → stockmem.search(k=5)
  Step 4 PREDICT   (REQUIRED)  : aihub.predict(current, similar)

Note — Step 2 no longer calls aihub.sentiment separately.
  Each IngestionRecord already carries sentiment_score + sentiment_label produced
  by the crawler enrichment pipeline (CryptoBERT / FinBERT).  We average those
  pre-computed scores across all fetched articles.
"""

import asyncio
import logging
from datetime import date

from main_controller.src.orchestrator.context import PipelineContext
from main_controller.src.orchestrator.exceptions import PipelineError
from shared.models.memory import StockMemRecord


def _best_text(article) -> str:
    """Return the richest available text for an article.

    Priority: summary → raw_text (truncated) → headline slug.
    """
    summary = (getattr(article, "summary", None) or "").strip()
    if summary:
        return summary

    raw = (getattr(article, "raw_text", None) or "").strip()
    if raw:
        return raw[:800]  # keep token budget sane per-article

    name = (getattr(article, "article_name", "") or "").strip()
    if name.startswith("http"):
        from urllib.parse import urlparse
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
    """STEP 2: Aggregate pre-computed sentiment; extract factors via LLM; push to ledger.

    Sentiment — each IngestionRecord already carries sentiment_score (CryptoBERT/FinBERT)
    produced by the crawler enrichment pipeline.  We take the mean across all
    articles so that a single outlier does not dominate.

    Factors — concatenate the richest available text per article (summary preferred)
    and ask the LLM to identify the top market-moving factors.  Summaries give the
    model structured, boilerplate-free context which produces higher-quality factors
    than raw headlines or full article bodies.
    """
    articles = ctx.latest_articles or []

    # ── Sentiment: aggregate pre-computed scores ──────────────────────────────
    scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
    if scores:
        avg = sum(scores) / len(scores)
        ctx.sentiment_score = round(avg, 4)
        ctx.sentiment_label = (
            "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral"
        )
        logger.info(
            "[step_ai_score] sentiment aggregated from %d articles: %.4f (%s)",
            len(scores), avg, ctx.sentiment_label,
        )
    else:
        ctx.sentiment_score = 0.0
        ctx.sentiment_label = "neutral"
        logger.warning("[step_ai_score] no pre-computed sentiment found, defaulting to 0.0")

    # ── Factors: use summaries (or raw_text/headline fallback) ───────────────
    combined_text = "\n\n".join(
        t for a in articles if (t := _best_text(a))
    )[:6000]  # ~1500 tokens — keeps LLM cost reasonable

    factors_result = await asyncio.gather(
        clients.aihub.factors(combined_text, ticker=ctx.symbol),
        return_exceptions=True,
    )
    factors_result = factors_result[0]  # unwrap single-item gather

    raw_factors = []
    if isinstance(factors_result, Exception):
        ctx.errors.append(f"AIHub factors failed: {factors_result}")
        logger.warning("Factors degraded to []: %s", factors_result)
    else:
        raw_factors = factors_result
        logger.info("[step_ai_score] extracted %d factors", len(raw_factors))

    # ── Factor Ledge: push factors → build/update rolling ledger ─────────────
    try:
        ctx.factors = await clients.factorledge.update_ledger(
            records=[
                {
                    "date": str(date.today()),
                    "factors": [f.name for f in raw_factors],
                }
            ],
            window_days=7,
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
