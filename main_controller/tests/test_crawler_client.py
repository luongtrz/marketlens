"""Tests for CrawlerClient."""

import pytest

from main_controller.src.clients.crawler_client import CrawlerClient
from main_controller.src.clients.exceptions import CrawlerClientError
from main_controller.tests.conftest import make_get_client, make_connect_error_client
from shared.models.article import IngestionRecord


async def test_get_latest_happy(monkeypatch, sample_article):
    payload = [sample_article.model_dump(mode="json")]
    monkeypatch.setattr("main_controller.src.clients.base.get_client", make_get_client(payload))
    client = CrawlerClient()
    results = await client.get_latest("BTCUSDT")
    assert len(results) == 1
    assert isinstance(results[0], IngestionRecord)
    assert results[0].id == "art-001"


async def test_connect_error_raises_crawler_error(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_connect_error_client(),
    )
    client = CrawlerClient()
    with pytest.raises(CrawlerClientError):
        await client.get_latest("BTCUSDT")
