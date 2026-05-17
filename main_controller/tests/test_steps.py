"""Tests for individual pipeline step functions."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from main_controller.src.orchestrator.context import PipelineContext
from main_controller.src.orchestrator.exceptions import PipelineError
from main_controller.src.orchestrator.steps import ModuleClients, step_collect, step_ai_score, step_stockmem, step_predict
from shared.models.article import IngestionRecord
from shared.models.prediction import SignalType


def _ctx(symbol: str = "BTCUSDT") -> PipelineContext:
    return PipelineContext(symbol=symbol, run_id=uuid4())


def _clients(**overrides) -> ModuleClients:
    """Build a ModuleClients with all methods as AsyncMocks unless overridden."""
    clients = ModuleClients(
        crawler=AsyncMock(),
        aihub=AsyncMock(),
        market=AsyncMock(),
        stockmem=AsyncMock(),
        factorledge=AsyncMock(),
    )
    for attr, val in overrides.items():
        setattr(clients, attr, val)
    return clients


# --- step_collect ---

async def test_step_collect_parallel_success(sample_article, sample_market_snapshot):
    ctx = _ctx()
    crawler = AsyncMock()
    crawler.get_latest = AsyncMock(return_value=[sample_article])
    market = AsyncMock()
    market.get_snapshot = AsyncMock(return_value=sample_market_snapshot)
    clients = _clients(crawler=crawler, market=market)

    await step_collect(ctx, clients)

    assert ctx.latest_articles == [sample_article]
    assert ctx.market_snapshot == sample_market_snapshot
    assert ctx.errors == []


async def test_step_collect_crawler_fail_raises(sample_market_snapshot):
    ctx = _ctx()
    crawler = AsyncMock()
    crawler.get_latest = AsyncMock(side_effect=Exception("connection refused"))
    market = AsyncMock()
    market.get_snapshot = AsyncMock(return_value=sample_market_snapshot)
    clients = _clients(crawler=crawler, market=market)

    with pytest.raises(PipelineError, match="Crawler"):
        await step_collect(ctx, clients)


async def test_step_collect_market_fail_raises(sample_article):
    ctx = _ctx()
    crawler = AsyncMock()
    crawler.get_latest = AsyncMock(return_value=[sample_article])
    market = AsyncMock()
    market.get_snapshot = AsyncMock(side_effect=Exception("timeout"))
    clients = _clients(crawler=crawler, market=market)

    with pytest.raises(PipelineError, match="MarketData"):
        await step_collect(ctx, clients)


# --- step_ai_score ---

async def test_step_ai_score_happy(sample_article, sample_factor, sample_normalized_factor):
    ctx = _ctx()
    ctx.latest_articles = [sample_article]

    aihub = AsyncMock()
    aihub.factors = AsyncMock(return_value=[sample_factor])
    factorledge = AsyncMock()
    factorledge.update_ledger = AsyncMock(return_value=[sample_normalized_factor])
    factorledge.classify_vector = AsyncMock(return_value=[1.0] * 5 + [0.0] * 70)
    clients = _clients(aihub=aihub, factorledge=factorledge)

    await step_ai_score(ctx, clients)

    assert ctx.sentiment_score == 0.7
    assert ctx.sentiment_label == "bullish"
    assert ctx.factors == [sample_normalized_factor]
    assert ctx.raw_factors == [sample_factor]
    assert ctx.factor_vector == [1.0] * 5 + [0.0] * 70
    assert ctx.errors == []


async def test_step_ai_score_factor_ledge_fail_uses_fallback(sample_article, sample_factor):
    ctx = _ctx()
    ctx.latest_articles = [sample_article]

    aihub = AsyncMock()
    aihub.factors = AsyncMock(return_value=[sample_factor])
    factorledge = AsyncMock()
    factorledge.update_ledger = AsyncMock(side_effect=Exception("FactorLedge down"))
    factorledge.classify_vector = AsyncMock(return_value=[])
    clients = _clients(aihub=aihub, factorledge=factorledge)

    await step_ai_score(ctx, clients)

    assert ctx.factors == []
    assert any("FactorLedge" in e for e in ctx.errors)


async def test_step_ai_score_sentiment_fail_defaults_zero():
    ctx = _ctx()
    ctx.latest_articles = [
        IngestionRecord(
            id="art-no-sent",
            article_name="No Sentiment Article",
            source="test",
            url="https://example.com",
            date_published=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            date_crawled=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            sentiment_score=0.0,  # No meaningful sentiment — defaults to neutral
            sentiment_label="neutral",
            factors=[],
        )
    ]

    aihub = AsyncMock()
    aihub.factors = AsyncMock(return_value=[])
    factorledge = AsyncMock()
    factorledge.update_ledger = AsyncMock(return_value=[])
    factorledge.classify_vector = AsyncMock(return_value=[])
    clients = _clients(aihub=aihub, factorledge=factorledge)

    await step_ai_score(ctx, clients)

    assert ctx.sentiment_score == 0.0
    assert ctx.sentiment_label == "neutral"


async def test_step_ai_score_factors_fail_defaults_empty(sample_article):
    ctx = _ctx()
    ctx.latest_articles = [sample_article]

    aihub = AsyncMock()
    aihub.factors = AsyncMock(side_effect=Exception("AIHub factors 500"))
    factorledge = AsyncMock()
    factorledge.update_ledger = AsyncMock(return_value=[])
    factorledge.classify_vector = AsyncMock(return_value=[])
    clients = _clients(aihub=aihub, factorledge=factorledge)

    await step_ai_score(ctx, clients)

    assert any("factors" in e for e in ctx.errors)


# --- step_stockmem ---

async def test_step_stockmem_saves_and_searches(sample_market_snapshot, sample_similar_record, sample_factor):
    ctx = _ctx()
    ctx.market_snapshot = sample_market_snapshot
    ctx.sentiment_score = 0.7
    ctx.sentiment_label = "bullish"
    ctx.raw_factors = [sample_factor]
    ctx.factors = []
    ctx.factor_vector = []
    ctx.latest_articles = []

    stockmem = AsyncMock()
    stockmem.save = AsyncMock(return_value="rec-999")
    stockmem.search = AsyncMock(return_value=[sample_similar_record])
    clients = _clients(stockmem=stockmem)

    await step_stockmem(ctx, clients)

    stockmem.search.assert_awaited_once()
    assert stockmem.search.await_args.kwargs.get("k") == 4  # k_similar=3 + 1
    assert ctx.current_record_id == "rec-999"
    assert ctx.similar_records == [sample_similar_record]
    assert ctx.current_record is not None


async def test_step_stockmem_no_market_snapshot_raises():
    ctx = _ctx()
    ctx.market_snapshot = None
    ctx.raw_factors = []
    clients = _clients()

    with pytest.raises(PipelineError, match="market_snapshot"):
        await step_stockmem(ctx, clients)


async def test_step_stockmem_save_failure_raises_pipeline_error(sample_market_snapshot):
    ctx = _ctx()
    ctx.market_snapshot = sample_market_snapshot
    ctx.sentiment_score = 0.0
    ctx.sentiment_label = "neutral"
    ctx.raw_factors = []
    ctx.factors = []
    ctx.latest_articles = []

    stockmem = AsyncMock()
    stockmem.save = AsyncMock(side_effect=Exception("StockMem down"))
    clients = _clients(stockmem=stockmem)

    with pytest.raises(PipelineError, match="StockMem"):
        await step_stockmem(ctx, clients)


# --- step_predict ---

async def test_step_predict_happy(sample_market_snapshot, sample_stockmem_record, sample_predict_response):
    ctx = _ctx()
    ctx.market_snapshot = sample_market_snapshot
    ctx.current_record = sample_stockmem_record
    ctx.similar_records = []

    aihub = AsyncMock()
    aihub.predict = AsyncMock(return_value=sample_predict_response)
    clients = _clients(aihub=aihub)

    await step_predict(ctx, clients)

    assert ctx.prediction == sample_predict_response
    assert ctx.prediction.signal == SignalType.BUY


async def test_step_predict_aihub_fail_raises_pipeline_error(sample_market_snapshot, sample_stockmem_record):
    ctx = _ctx()
    ctx.market_snapshot = sample_market_snapshot
    ctx.current_record = sample_stockmem_record
    ctx.similar_records = []

    aihub = AsyncMock()
    aihub.predict = AsyncMock(side_effect=Exception("AIHub predict 500"))
    clients = _clients(aihub=aihub)

    with pytest.raises(PipelineError, match="predict"):
        await step_predict(ctx, clients)


async def test_step_predict_llm_gateway_happy(sample_market_snapshot, sample_stockmem_record, sample_predict_response):
    ctx = _ctx()
    ctx.market_snapshot = sample_market_snapshot
    ctx.current_record = sample_stockmem_record
    ctx.similar_records = []

    llm_gateway = AsyncMock()
    llm_gateway.predict = AsyncMock(return_value=sample_predict_response)
    clients = _clients(llm_gateway=llm_gateway)

    await step_predict(ctx, clients, predict_provider="llm_gateway")

    assert ctx.prediction == sample_predict_response
    llm_gateway.predict.assert_awaited_once_with(
        current=sample_stockmem_record,
        similar=[],
        model=None,
    )


async def test_step_predict_llm_gateway_model_override(sample_market_snapshot, sample_stockmem_record, sample_predict_response):
    ctx = _ctx()
    ctx.market_snapshot = sample_market_snapshot
    ctx.current_record = sample_stockmem_record
    ctx.similar_records = []

    llm_gateway = AsyncMock()
    llm_gateway.predict = AsyncMock(return_value=sample_predict_response)
    clients = _clients(llm_gateway=llm_gateway)

    await step_predict(ctx, clients, predict_provider="llm_gateway", llm_model="qwen3.5-plus")

    assert ctx.prediction == sample_predict_response
    llm_gateway.predict.assert_awaited_once_with(
        current=sample_stockmem_record,
        similar=[],
        model="qwen3.5-plus",
    )
