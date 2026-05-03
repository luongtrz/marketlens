"""Integration tests for the full Pipeline orchestrator with mocked clients."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from main_controller.src.orchestrator.pipeline import Pipeline, PipelineConfig
from main_controller.src.orchestrator.steps import ModuleClients
from shared.models.prediction import SignalType


def _make_clients(
    sample_article, sample_market_snapshot, sample_normalized_factor,
    sample_similar_record, sample_predict_response
) -> ModuleClients:
    crawler = AsyncMock()
    crawler.get_latest = AsyncMock(return_value=[sample_article])

    market = AsyncMock()
    market.get_snapshot = AsyncMock(return_value=sample_market_snapshot)

    aihub = AsyncMock()
    aihub.sentiment = AsyncMock(return_value={"score": 0.7, "label": "bullish"})
    aihub.factors = AsyncMock(return_value=[])
    aihub.predict = AsyncMock(return_value=sample_predict_response)

    factorledge = AsyncMock()
    factorledge.ingest = AsyncMock(return_value=[sample_normalized_factor])

    stockmem = AsyncMock()
    stockmem.save = AsyncMock(return_value="rec-001")
    stockmem.search = AsyncMock(return_value=[sample_similar_record])

    return ModuleClients(
        crawler=crawler,
        aihub=aihub,
        market=market,
        stockmem=stockmem,
        factorledge=factorledge,
    )


async def test_pipeline_full_run_with_mocks(
    sample_article, sample_market_snapshot, sample_normalized_factor,
    sample_similar_record, sample_predict_response
):
    clients = _make_clients(
        sample_article, sample_market_snapshot, sample_normalized_factor,
        sample_similar_record, sample_predict_response,
    )
    pipeline = Pipeline(clients, PipelineConfig(k_similar=5))
    result = await pipeline.run("BTCUSDT")

    assert result.signal == SignalType.BUY
    assert result.confidence == 0.85
    assert result.symbol == "BTCUSDT"
    assert len(result.similar_cases) == 1


async def test_pipeline_market_fail_returns_hold(
    sample_article, sample_normalized_factor,
    sample_similar_record, sample_predict_response
):
    crawler = AsyncMock()
    crawler.get_latest = AsyncMock(return_value=[sample_article])

    market = AsyncMock()
    market.get_snapshot = AsyncMock(side_effect=Exception("MarketData down"))

    aihub = AsyncMock()
    aihub.sentiment = AsyncMock(return_value={"score": 0.0, "label": "neutral"})
    aihub.factors = AsyncMock(return_value=[])

    factorledge = AsyncMock()
    factorledge.ingest = AsyncMock(return_value=[sample_normalized_factor])

    stockmem = AsyncMock()

    clients = ModuleClients(
        crawler=crawler, aihub=aihub, market=market,
        stockmem=stockmem, factorledge=factorledge,
    )
    pipeline = Pipeline(clients)
    result = await pipeline.run("BTCUSDT")

    assert result.signal == SignalType.HOLD
    assert result.confidence == 0.0
    assert any("market_snapshot" in e.lower() or "MarketData" in e for e in result.errors)


async def test_pipeline_run_id_propagated(
    sample_article, sample_market_snapshot, sample_normalized_factor,
    sample_similar_record, sample_predict_response
):
    clients = _make_clients(
        sample_article, sample_market_snapshot, sample_normalized_factor,
        sample_similar_record, sample_predict_response,
    )
    pipeline = Pipeline(clients)
    run_id = uuid4()
    result = await pipeline.run("BTCUSDT", run_id=run_id)

    assert result.run_id == str(run_id)
