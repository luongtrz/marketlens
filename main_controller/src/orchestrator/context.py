"""PipelineContext dataclass — accumulates state across pipeline steps."""

from dataclasses import dataclass, field
from uuid import UUID

from shared.models.article import IngestionRecord
from shared.models.factor import NormalizedFactor
from shared.models.market import MarketSnapshot
from shared.models.memory import SimilarRecord
from shared.models.prediction import PredictionResult, PredictResponse, SignalType

from datetime import datetime


@dataclass
class PipelineContext:
    """Mutable context passed through all pipeline steps, accumulating results.

    Each step function reads from and writes to this context object.
    """

    symbol: str
    run_id: UUID
    latest_articles: list[IngestionRecord] = field(default_factory=list)
    market_snapshot: MarketSnapshot | None = None
    sentiment_score: float | None = None
    factors: list[NormalizedFactor] = field(default_factory=list)
    current_record_id: str | None = None
    similar_records: list[SimilarRecord] = field(default_factory=list)
    prediction: PredictResponse | None = None
    errors: list[str] = field(default_factory=list)

    def build_result(self) -> PredictionResult:
        """Assemble the final PredictionResult from all accumulated context.

        Returns:
            Complete PredictionResult for the pipeline run.
        """
        return PredictionResult(
            run_id=str(self.run_id),
            symbol=self.symbol,
            timestamp=datetime.utcnow(),
            signal=self.prediction.signal if self.prediction else SignalType.HOLD,
            confidence=self.prediction.confidence if self.prediction else 0.0,
            explanation=self.prediction.explanation if self.prediction else "",
            reasoning_steps=self.prediction.reasoning_steps if self.prediction else [],
            similar_cases=self.similar_records,
            sentiment_score=self.sentiment_score or 0.0,
            key_factors=self.factors,
            market_snapshot=self.market_snapshot,
            errors=self.errors,
        )
