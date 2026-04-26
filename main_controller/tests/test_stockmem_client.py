"""Tests for StockMemClient."""

import pytest

from main_controller.src.clients.stockmem_client import StockMemClient
from main_controller.src.clients.exceptions import StockMemClientError
from main_controller.tests.conftest import make_get_client, make_connect_error_client
from shared.models.memory import SimilarRecord


async def test_save_returns_id(monkeypatch, sample_stockmem_record):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({"id": "rec-123"}),
    )
    client = StockMemClient()
    record_id = await client.save(sample_stockmem_record)
    assert record_id == "rec-123"


async def test_search_parses_similar_records(monkeypatch, sample_similar_record, sample_stockmem_record):
    payload = {"results": [sample_similar_record.model_dump(mode="json")]}
    monkeypatch.setattr("main_controller.src.clients.base.get_client", make_get_client(payload))
    client = StockMemClient()
    results = await client.search(sample_stockmem_record, k=5)
    assert len(results) == 1
    assert isinstance(results[0], SimilarRecord)
    assert results[0].similarity == 0.92


async def test_http_error_raises(monkeypatch, sample_stockmem_record):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({}, status_code=500),
    )
    client = StockMemClient()
    with pytest.raises(StockMemClientError):
        await client.save(sample_stockmem_record)
