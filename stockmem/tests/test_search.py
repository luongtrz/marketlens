from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock

import numpy as np

from stockmem.src.config import SearchWeights
from stockmem.src.models import CandleData, MarketSnapshot, StockMemRecord
from stockmem.src.search.embedder import RecordEmbedder
from stockmem.src.search.event_memory import build_daily_event_state
from stockmem.src.search.index import MemoryVectorIndex
from stockmem.src.search.index import ScoredId
from stockmem.src.search.learned_metric import LearnedDiagonalMetric
from stockmem.src.search.searcher import RecordSearcher
from stockmem.src.service import StockMemService


def _candles_trending(slope: float, n: int = 25, start: float = 100.0) -> list[CandleData]:
    out = []
    price = start
    for i in range(n):
        price *= (1.0 + slope)
        out.append(
            CandleData(
                open=price * (1.0 - abs(slope) * 0.5),
                high=price * (1.0 + abs(slope)),
                low=price * (1.0 - abs(slope)),
                close=price,
                volume=1_000_000.0 * (1.0 + 0.01 * i),
            )
        )
    return out


def _build_searcher(
    records: list[StockMemRecord],
    weights: SearchWeights,
    learned_metric: LearnedDiagonalMetric | None = None,
) -> RecordSearcher:
    cache = {r.id: r for r in records if r.id is not None}
    embedder = RecordEmbedder()
    embedder.rebuild_corpus(cache.values())
    index = MemoryVectorIndex()
    index.rebuild([(rid, embedder.embed(rec)) for rid, rec in cache.items()])
    return RecordSearcher(
        embedder=embedder,
        index=index,
        record_cache=cache,
        weights=weights,
        learned_metric=learned_metric,
    )


def test_search_returns_top_k_with_similarity_bounds() -> None:
    rec_a = StockMemRecord(
        id="a",
        date=date(2026, 4, 10),
        symbol="BTC",
        sentiment_score=0.7,
        factors=["Record ETF inflows", "Fed holds interest rate steady"],
        market_snapshot=MarketSnapshot(
            rsi=60.0,
            macd_hist=0.03,
            msi=65.0,
            fear_greed_index=70.0,
            price_change_pct=1.2,
            candles=_candles_trending(0.01),
        ),
    )
    rec_b = StockMemRecord(
        id="b",
        date=date(2026, 3, 8),
        symbol="BTC",
        sentiment_score=-0.2,
        factors=["Major exchange hack"],
        market_snapshot=MarketSnapshot(
            rsi=39.0,
            macd_hist=-0.02,
            msi=35.0,
            fear_greed_index=30.0,
            price_change_pct=-1.5,
            candles=_candles_trending(-0.01),
        ),
    )

    searcher = _build_searcher(
        [rec_a, rec_b],
        SearchWeights(w1_factor=0.35, w2_indicator=0.20, w3_price=0.45),
    )

    query = StockMemRecord(
        date=date(2026, 4, 14),
        symbol="BTC",
        sentiment_score=0.68,
        factors=["Record ETF inflows", "Fed holds interest rate steady"],
        market_snapshot=MarketSnapshot(
            rsi=61.0,
            macd_hist=0.031,
            msi=66.0,
            fear_greed_index=71.0,
            price_change_pct=1.1,
            candles=_candles_trending(0.01),
        ),
    )

    results = searcher.search(query, k=2)
    assert len(results) == 2
    assert all(0.0 <= r.similarity <= 1.0 for r in results)
    assert results[0].record.id == "a"
    assert results[0].retriever_version == "fixed_knn_v1"
    assert "event_vector_cosine" in results[0].event_match


def test_weighted_ranking_changes_with_weights() -> None:
    # rec_a: factor matches query, price opposite.
    # rec_b: factor mismatches, price matches query.
    rec_a = StockMemRecord(
        id="a",
        date=date(2026, 4, 10),
        symbol="BTC",
        sentiment_score=0.0,
        factors=["Record ETF inflows", "Fed holds interest rate steady"],
        market_snapshot=MarketSnapshot(
            rsi=50.0,
            candles=_candles_trending(-0.01),
        ),
    )
    rec_b = StockMemRecord(
        id="b",
        date=date(2026, 4, 11),
        symbol="BTC",
        sentiment_score=0.0,
        factors=["Major exchange hack"],
        market_snapshot=MarketSnapshot(
            rsi=50.0,
            candles=_candles_trending(0.01),
        ),
    )

    query = StockMemRecord(
        date=date(2026, 4, 14),
        symbol="BTC",
        sentiment_score=0.0,
        factors=["Record ETF inflows", "Fed holds interest rate steady"],
        market_snapshot=MarketSnapshot(
            rsi=50.0,
            candles=_candles_trending(0.01),
        ),
    )

    # Factor-dominant → rec_a wins.
    s_factor = _build_searcher(
        [rec_a, rec_b],
        SearchWeights(w1_factor=0.90, w2_indicator=0.05, w3_price=0.05),
    )
    assert s_factor.search(query, k=1)[0].record.id == "a"

    # Price-dominant → rec_b wins.
    s_price = _build_searcher(
        [rec_a, rec_b],
        SearchWeights(w1_factor=0.05, w2_indicator=0.05, w3_price=0.90),
    )
    assert s_price.search(query, k=1)[0].record.id == "b"


def test_search_on_empty_cache_returns_empty() -> None:
    searcher = _build_searcher([], SearchWeights(0.35, 0.20, 0.45))
    query = StockMemRecord(
        date=date(2026, 4, 14),
        symbol="BTC",
        sentiment_score=0.0,
        factors=["Record ETF inflows"],
        market_snapshot=MarketSnapshot(rsi=50.0),
    )
    assert searcher.search(query, k=5) == []


def test_service_search_preserves_same_date_caller_query() -> None:
    stored = StockMemRecord(
        id="stored",
        date=date(2026, 4, 14),
        symbol="BTC",
        sentiment_score=-1.0,
        factors=["Major exchange hack"],
        market_snapshot=MarketSnapshot(rsi=10.0),
    )
    query = StockMemRecord(
        date=stored.date,
        symbol=stored.symbol,
        sentiment_score=1.0,
        factors=["Record ETF inflows"],
        market_snapshot=MarketSnapshot(rsi=90.0),
    )
    service = StockMemService(
        "postgresql+asyncpg://postgres:pass@localhost:5432/postgres",
        "memory",
        SearchWeights(0.35, 0.20, 0.45),
    )
    service.records_by_id[stored.id] = stored
    service.searcher = MagicMock()
    service.searcher.search.return_value = []

    asyncio.run(service.search(query))

    effective_query = service.searcher.search.call_args.args[0]
    assert effective_query.factors == query.factors
    assert effective_query.market_snapshot.rsi == query.market_snapshot.rsi
    assert effective_query.event_state is not None


def test_learned_retriever_falls_back_when_artifact_is_missing() -> None:
    rec = StockMemRecord(
        id="a",
        date=date(2026, 4, 10),
        symbol="BTC",
        sentiment_score=0.0,
        factors=["Record ETF inflows"],
        market_snapshot=MarketSnapshot(rsi=50.0, candles=_candles_trending(0.01)),
    )
    query = rec.model_copy(update={"id": None, "date": date(2026, 4, 14)})
    searcher = _build_searcher([rec], SearchWeights(0.35, 0.20, 0.45))
    fixed = searcher.search(query, k=1, retriever_type="fixed_knn")
    learned = searcher.search(query, k=1, retriever_type="learned_linear")
    assert learned == fixed


def test_learned_retriever_full_scans_beyond_fixed_ann_prefilter() -> None:
    rec_a = StockMemRecord(
        id="ann-only",
        date=date(2026, 4, 10),
        symbol="BTC",
        sentiment_score=-1.0,
        factors=["Major exchange hack"],
        market_snapshot=MarketSnapshot(rsi=20.0, candles=_candles_trending(-0.01)),
    )
    rec_b = StockMemRecord(
        id="learned-best",
        date=date(2026, 4, 11),
        symbol="BTC",
        sentiment_score=1.0,
        factors=["Record ETF inflows"],
        market_snapshot=MarketSnapshot(rsi=80.0, candles=_candles_trending(0.01)),
    )
    metric = LearnedDiagonalMetric(
        block_dims=(75, 5, 60),
        diagonal=np.ones(140),
        block_scales=np.array([0.0, 1.0, 0.0]),
    )
    searcher = _build_searcher(
        [rec_a, rec_b],
        SearchWeights(0.35, 0.20, 0.45),
        learned_metric=metric,
    )
    searcher._index.search = lambda _query, _k: [ScoredId(record_id="ann-only", score=1.0)]
    query = rec_b.model_copy(update={"id": None, "date": date(2026, 4, 14)})

    result = searcher.search(query, k=1, retriever_type="learned_linear")

    assert result[0].record.id == "learned-best"


def test_four_block_learned_retriever_ranks_and_traces_event_similarity() -> None:
    same_event = StockMemRecord(
        id="same-event",
        date=date(2026, 4, 10),
        symbol="BTC",
        sentiment_score=0.0,
        factors=["Record ETF inflows"],
        market_snapshot=MarketSnapshot(rsi=50.0),
    )
    different_event = StockMemRecord(
        id="different-event",
        date=date(2026, 4, 11),
        symbol="BTC",
        sentiment_score=0.0,
        factors=["Major exchange hack"],
        market_snapshot=MarketSnapshot(rsi=50.0),
    )
    query = same_event.model_copy(
        update={
            "id": None,
            "date": date(2026, 4, 14),
        }
    )
    same_event = same_event.model_copy(
        update={"event_state": build_daily_event_state(same_event)}
    )
    different_event = different_event.model_copy(
        update={"event_state": build_daily_event_state(different_event)}
    )
    query = query.model_copy(update={"event_state": build_daily_event_state(query)})
    metric = LearnedDiagonalMetric(
        block_dims=(85, 75, 5, 60),
        diagonal=np.ones(225),
        block_scales=np.array([1.0, 0.0, 0.0, 0.0]),
        version="learned_cem_test",
    )
    searcher = _build_searcher(
        [same_event, different_event],
        SearchWeights(0.35, 0.20, 0.45),
        learned_metric=metric,
    )

    result = searcher.search(query, k=1, retriever_type="learned_linear")

    assert result[0].record.id == "same-event"
    assert result[0].retriever_version == "learned_cem_test"
    assert result[0].event_match["event_vector_cosine"] > 0.9
