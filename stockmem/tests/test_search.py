from __future__ import annotations

from datetime import date

from stockmem.src.config import SearchWeights
from stockmem.src.models import MarketSnapshot, StockMemRecord
from stockmem.src.search.embedder import RecordEmbedder
from stockmem.src.search.index import MemoryVectorIndex
from stockmem.src.search.searcher import RecordSearcher


def test_search_returns_top_k_with_similarity_bounds() -> None:
    rec_a = StockMemRecord(
        id="a",
        date=date(2026, 4, 10),
        symbol="BTC",
        sentiment_score=0.7,
        factors=["macro", "etf_flow"],
        market_snapshot=MarketSnapshot(rsi=60.0, macd_hist=0.03),
    )
    rec_b = StockMemRecord(
        id="b",
        date=date(2026, 3, 8),
        symbol="BTC",
        sentiment_score=-0.2,
        factors=["risk_off"],
        market_snapshot=MarketSnapshot(rsi=39.0, macd_hist=-0.02),
    )

    cache = {"a": rec_a, "b": rec_b}
    embedder = RecordEmbedder()
    embedder.rebuild_corpus(cache.values())

    index = MemoryVectorIndex()
    index.rebuild([(rid, embedder.embed(rec)) for rid, rec in cache.items()])

    searcher = RecordSearcher(
        embedder=embedder,
        index=index,
        record_cache=cache,
        weights=SearchWeights(w1_factor=0.35, w2_indicator=0.20, w3_price=0.45),
    )

    query = StockMemRecord(
        date=date(2026, 4, 14),
        symbol="BTC",
        sentiment_score=0.68,
        factors=["macro", "etf_flow"],
        market_snapshot=MarketSnapshot(rsi=61.0, macd_hist=0.031),
    )

    results = searcher.search(query, k=2)

    assert len(results) == 2
    assert all(0.0 <= x.similarity <= 1.0 for x in results)
    assert results[0].record.id == "a"
