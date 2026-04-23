from __future__ import annotations

from datetime import date

from stockmem.src.config import SearchWeights
from stockmem.src.models import CandleData, MarketSnapshot, StockMemRecord
from stockmem.src.search.embedder import RecordEmbedder
from stockmem.src.search.index import MemoryVectorIndex
from stockmem.src.search.searcher import RecordSearcher


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


def _build_searcher(records: list[StockMemRecord], weights: SearchWeights) -> RecordSearcher:
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
