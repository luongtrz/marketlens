"""Individual pipeline step functions."""

import asyncio

from main_controller.src.orchestrator.context import PipelineContext


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
    """STEP 1: Collect data from Crawler and MarketData in parallel.

    Runs Crawler trigger and MarketData snapshot concurrently.
    Stores results in ctx.latest_articles and ctx.market_snapshot.
    On individual failure: logs warning, stores None/empty, continues.
    """
    results = await asyncio.gather(
        clients.crawler.get_latest(ctx.symbol),
        clients.market.get_snapshot(ctx.symbol),
        return_exceptions=True,
    )
    ctx.latest_articles = results[0] if not isinstance(results[0], Exception) else []
    ctx.market_snapshot = results[1] if not isinstance(results[1], Exception) else None

    if isinstance(results[0], Exception):
        ctx.errors.append(f"Crawler failed: {results[0]}")
    if isinstance(results[1], Exception):
        ctx.errors.append(f"MarketData failed: {results[1]}")


async def step_ai_score(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 2: Call AIHub /sentiment and /factors, attach results to ctx."""
    raise NotImplementedError


async def step_stockmem(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 3: Save current record to StockMem, retrieve k=5 most similar past records."""
    raise NotImplementedError


async def step_predict(ctx: PipelineContext, clients: ModuleClients) -> None:
    """STEP 4: Call AIHub /predict with current context + similar cases (RAG)."""
    raise NotImplementedError
