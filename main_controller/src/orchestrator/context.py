"""PipelineContext dataclass — accumulates state across pipeline steps."""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from uuid import UUID

from shared.models.article import IngestionRecord
from shared.models.factor import Factor, NormalizedFactor
from shared.models.market import MarketSnapshot
from shared.models.memory import SimilarRecord, StockMemRecord
from shared.models.prediction import PredictResponse, PredictionResult, SignalType


@dataclass
class PipelineContext:
    """Mutable context passed through all pipeline steps, accumulating results."""

    symbol: str
    run_id: UUID
    as_of_date: date | None = None  # None = today (live mode), set = historical backtest
    latest_articles: list[IngestionRecord] = field(default_factory=list)
    market_snapshot: MarketSnapshot | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    raw_factors: list[Factor] = field(default_factory=list)  # Raw factors from Supabase or AIHub
    factors: list[NormalizedFactor] = field(default_factory=list)
    factor_vector: list[float] = field(default_factory=list)  # 75d binary vector from FactorLedge
    current_record: StockMemRecord | None = None
    current_record_id: str | None = None
    similar_records: list[SimilarRecord] = field(default_factory=list)
    prediction: PredictResponse | None = None
    errors: list[str] = field(default_factory=list)

    def build_result(self) -> PredictionResult:
        return PredictionResult(
            run_id=str(self.run_id),
            symbol=self.symbol,
            timestamp=datetime.now(timezone.utc),
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
