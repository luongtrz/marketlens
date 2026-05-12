"""Tests for FactorLedgeClient."""

import pytest

from main_controller.src.clients.factorledge_client import FactorLedgeClient
from main_controller.src.clients.exceptions import FactorLedgeClientError
from main_controller.tests.conftest import make_get_client, make_connect_error_client
from shared.models.factor import NormalizedFactor


async def test_update_ledger_happy(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({"message": "ledger updated"}),
    )
    client = FactorLedgeClient()
    results = await client.update_ledger(
        records=[{"date": "2026-01-01", "factors": ["institutional flows"]}],
        window_days=7,
    )
    assert results == []


async def test_classify_vector_happy(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({"factorVector": [1.0] * 3 + [0.0] * 72}),
    )
    client = FactorLedgeClient()
    result = await client.classify_vector(["institutional flows"])
    assert len(result) == 75
    assert result[:3] == [1.0, 1.0, 1.0]


async def test_update_ledger_http_error_raises(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({}, status_code=503),
    )
    client = FactorLedgeClient()
    with pytest.raises(FactorLedgeClientError):
        await client.update_ledger(
            records=[{"date": "2026-01-01", "factors": ["factor"]}],
        )
