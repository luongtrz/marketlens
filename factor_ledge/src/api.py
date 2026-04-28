"""FactorLedge API: clean, normalize, and serve factor signals."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from shared.models.factor import FactorType, NormalizedFactor

_STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
_POSITIVE_KEYWORDS = (
    "approval",
    "adoption",
    "breakthrough",
    "bullish",
    "buy",
    "inflow",
    "rally",
    "surge",
    "upgrade",
)
_NEGATIVE_KEYWORDS = (
    "ban",
    "bearish",
    "dump",
    "hack",
    "lawsuit",
    "outflow",
    "risk",
    "sell",
    "warning",
)
_TYPE_KEYWORDS: dict[FactorType, tuple[str, ...]] = {
    FactorType.REGULATORY: ("sec", "etf", "law", "policy", "regulat", "reserve"),
    FactorType.TECHNICAL: ("macd", "rsi", "support", "resistance", "volume", "volatility"),
    FactorType.ON_CHAIN: ("staking", "mining", "onchain", "wallet", "hashrate"),
    FactorType.EXCHANGE: ("binance", "coinbase", "listing", "liquidation", "orderbook"),
    FactorType.SENTIMENT: ("fear", "greed", "sentiment", "social", "hype"),
    FactorType.MACRO: ("cpi", "fed", "inflation", "institutional", "liquidity"),
}
_ENRICH_LOOKUP: dict[str, tuple[str, list[str]]] = {
    "bitcoin": ("layer1", ["BTC", "BTCUSDT"]),
    "btc": ("layer1", ["BTC", "BTCUSDT"]),
    "ethereum": ("layer1", ["ETH", "ETHUSDT"]),
    "eth": ("layer1", ["ETH", "ETHUSDT"]),
    "solana": ("layer1", ["SOL", "SOLUSDT"]),
    "sol": ("layer1", ["SOL", "SOLUSDT"]),
    "binance": ("exchange", ["BNB", "BNBUSDT"]),
    "xrp": ("payments", ["XRP", "XRPUSDT"]),
}


class IngestRequest(BaseModel):
    article_id: str
    factors: list[str] = Field(default_factory=list)
    source: str


class IngestResponse(BaseModel):
    factors: list[NormalizedFactor]


class FactorSummary(BaseModel):
    symbol: str | None = None
    total: int
    avg_weight: float
    top_factors: list[str] = Field(default_factory=list)


def _canonicalize_factor(value: str) -> str:
    lowered = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    tokens = [token for token in cleaned.split() if token and token not in _STOPWORDS]
    return "_".join(tokens)


def _infer_type(name: str) -> FactorType:
    haystack = name.replace("_", " ")
    for factor_type, keywords in _TYPE_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return factor_type
    return FactorType.MACRO


def _infer_polarity(name: str) -> float:
    haystack = name.replace("_", " ")
    positive_hits = sum(1 for keyword in _POSITIVE_KEYWORDS if keyword in haystack)
    negative_hits = sum(1 for keyword in _NEGATIVE_KEYWORDS if keyword in haystack)
    if positive_hits == 0 and negative_hits == 0:
        return 0.0
    raw = 0.4 * (positive_hits - negative_hits)
    return max(-1.0, min(1.0, raw))


def _enrich(name: str) -> tuple[str | None, list[str]]:
    haystack = name.replace("_", " ")
    for keyword, (sector, symbols) in _ENRICH_LOOKUP.items():
        if keyword in haystack:
            return sector, symbols
    return None, []


@dataclass
class FactorPipeline:
    """In-memory processor for raw factor strings."""

    history: list[NormalizedFactor] = field(default_factory=list)
    frequency: Counter[str] = field(default_factory=Counter)

    def run(self, article_id: str, raw_factors: list[str]) -> list[NormalizedFactor]:
        now = datetime.now(timezone.utc)
        seen: set[str] = set()
        normalized: list[NormalizedFactor] = []

        for raw in raw_factors:
            cleaned = _canonicalize_factor(raw)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)

            self.frequency[cleaned] += 1
            frequency_bonus = min(0.35, 0.05 * float(self.frequency[cleaned] - 1))
            weight = min(1.0, 0.5 + frequency_bonus)
            sector, symbols = _enrich(cleaned)

            normalized_factor = NormalizedFactor(
                name=cleaned,
                type=_infer_type(cleaned),
                weight=weight,
                polarity=_infer_polarity(cleaned),
                sector=sector,
                related_symbols=symbols,
                source_article_id=article_id,
                observed_at=now,
            )
            normalized.append(normalized_factor)

        self.history.extend(normalized)
        return normalized

    def list(self, symbol: str | None, limit: int) -> list[NormalizedFactor]:
        if symbol is None:
            return list(reversed(self.history[-limit:]))

        symbol_upper = symbol.upper()
        filtered = [item for item in self.history if symbol_upper in item.related_symbols]
        return list(reversed(filtered[-limit:]))

    def summary(self, symbol: str | None) -> FactorSummary:
        if symbol is None:
            selected = self.history
        else:
            symbol_upper = symbol.upper()
            selected = [item for item in self.history if symbol_upper in item.related_symbols]

        if not selected:
            return FactorSummary(symbol=symbol, total=0, avg_weight=0.0, top_factors=[])

        counts = Counter(item.name for item in selected)
        top_factors = [name for name, _ in counts.most_common(5)]
        avg_weight = sum(item.weight for item in selected) / float(len(selected))
        return FactorSummary(
            symbol=symbol,
            total=len(selected),
            avg_weight=avg_weight,
            top_factors=top_factors,
        )


app = FastAPI(title="FactorLedge", version="0.1.0")
pipeline = FactorPipeline()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest) -> IngestResponse:
    factors = pipeline.run(payload.article_id, payload.factors)
    return IngestResponse(factors=factors)


@app.get("/factors", response_model=list[NormalizedFactor])
async def factors(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[NormalizedFactor]:
    return pipeline.list(symbol=symbol, limit=limit)


@app.get("/summary", response_model=FactorSummary)
async def summary(symbol: str | None = Query(default=None)) -> FactorSummary:
    return pipeline.summary(symbol=symbol)

