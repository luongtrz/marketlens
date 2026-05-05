"""Full pipeline orchestrator — runs all steps in sequence.

POST /run?symbol=BTCUSDT
  → Step 1 COLLECT (parallel): crawler.get_latest [REQUIRED] + market.get_snapshot [REQUIRED]
  → Step 2 AI SCORE (parallel): aihub.sentiment + aihub.factors → factorledge.ingest [graceful]
  → Step 3 STOCKMEM (sequential, REQUIRED): stockmem.save → stockmem.search(k_similar)
  → Step 4 PREDICT (REQUIRED): aihub.predict(current, similar)
  → PredictionResult { signal, confidence, explanation, similar_cases, errors[] }
"""

import logging
from datetime import date
from uuid import UUID, uuid4

from shared.models.prediction import PredictionResult, SignalType
from main_controller.src.orchestrator.context import PipelineContext
from main_controller.src.orchestrator.exceptions import PipelineError
from main_controller.src.orchestrator.steps import (
    ModuleClients,
    step_collect,
    step_ai_score,
    step_stockmem,
    step_predict,
)

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for the pipeline orchestrator."""

    def __init__(self, k_similar: int = 3) -> None:
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

    async def run(self, symbol: str, run_id: UUID | None = None, as_of_date: date | None = None) -> PredictionResult:
        """Execute the full pipeline for a given symbol.

        Args:
            symbol: Trading pair symbol (e.g. "BTCUSDT").
            run_id: Pre-generated run UUID from the API layer (generated here if None).

        Returns:
            Complete PredictionResult — always returns, never raises.
        """
        ctx = PipelineContext(symbol=symbol, run_id=run_id or uuid4(), as_of_date=as_of_date)

        try:
            await step_collect(ctx, self._clients)
        except Exception as exc:
            logger.exception("step_collect raised unexpectedly: %s", exc)
            ctx.errors.append(f"step_collect failed: {exc}")

        try:
            await step_ai_score(ctx, self._clients)
        except Exception as exc:
            logger.exception("step_ai_score raised unexpectedly: %s", exc)
            ctx.errors.append(f"step_ai_score failed: {exc}")

        try:
            await step_stockmem(ctx, self._clients, self._config.k_similar)
        except PipelineError as exc:
            logger.error("step_stockmem PipelineError: %s", exc)
            ctx.errors.append(str(exc))
            return _hold_result(ctx)
        except Exception as exc:
            logger.exception("step_stockmem raised unexpectedly: %s", exc)
            ctx.errors.append(f"step_stockmem failed: {exc}")
            return _hold_result(ctx)

        try:
            await step_predict(ctx, self._clients)
        except PipelineError as exc:
            logger.error("step_predict PipelineError: %s", exc)
            ctx.errors.append(str(exc))
            return _hold_result(ctx)
        except Exception as exc:
            logger.exception("step_predict raised unexpectedly: %s", exc)
            ctx.errors.append(f"step_predict failed: {exc}")
            return _hold_result(ctx)

        return ctx.build_result()


def _hold_result(ctx: PipelineContext) -> PredictionResult:
    """Build a HOLD result when a required step fails."""
    from datetime import datetime, timezone

    return PredictionResult(
        run_id=str(ctx.run_id),
        symbol=ctx.symbol,
        timestamp=datetime.now(timezone.utc),
        signal=SignalType.HOLD,
        confidence=0.0,
        explanation="Pipeline encountered a critical error; defaulting to HOLD.",
        reasoning_steps=[],
        similar_cases=ctx.similar_records,
        sentiment_score=ctx.sentiment_score or 0.0,
        key_factors=ctx.factors,
        market_snapshot=ctx.market_snapshot,
        errors=ctx.errors,
    )
