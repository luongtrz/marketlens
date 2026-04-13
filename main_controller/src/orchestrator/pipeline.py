"""Full pipeline orchestrator — runs all steps in sequence."""

from uuid import uuid4

from shared.models.prediction import PredictionResult
from main_controller.src.orchestrator.context import PipelineContext
from main_controller.src.orchestrator.steps import (
    ModuleClients,
    step_collect,
    step_ai_score,
    step_stockmem,
    step_predict,
)


class PipelineConfig:
    """Configuration for the pipeline orchestrator."""

    def __init__(self, k_similar: int = 5) -> None:
        self.k_similar = k_similar


class Pipeline:
    """Master pipeline orchestrator.

    Calls all modules in the correct order to produce a PredictionResult.

    Args:
        clients: Injected module client instances.
        config: Pipeline configuration.
    """

    def __init__(self, clients: ModuleClients, config: PipelineConfig | None = None) -> None:
        self._clients = clients
        self._config = config or PipelineConfig()

    async def run(self, symbol: str) -> PredictionResult:
        """Execute the full pipeline for a given symbol.

        Steps:
        1. Collect — Crawler + MarketData in parallel
        2. AI Score — AIHub sentiment + factors
        3. StockMem — Save record + retrieve similar
        4. Predict — AIHub RAG predict/explain

        Args:
            symbol: Trading pair symbol (e.g. "BTCUSDT").

        Returns:
            Complete PredictionResult.
        """
        ctx = PipelineContext(symbol=symbol, run_id=uuid4())

        await self._step_collect(ctx)
        await self._step_ai_score(ctx)
        await self._step_stockmem(ctx)
        await self._step_predict(ctx)

        return ctx.build_result()

    async def _step_collect(self, ctx: PipelineContext) -> None:
        await step_collect(ctx, self._clients)

    async def _step_ai_score(self, ctx: PipelineContext) -> None:
        await step_ai_score(ctx, self._clients)

    async def _step_stockmem(self, ctx: PipelineContext) -> None:
        await step_stockmem(ctx, self._clients)

    async def _step_predict(self, ctx: PipelineContext) -> None:
        await step_predict(ctx, self._clients)
