"""Tests for FactorLedgeClient."""

import pytest

from main_controller.src.clients.factorledge_client import FactorLedgeClient
from main_controller.src.clients.exceptions import FactorLedgeClientError
from main_controller.tests.conftest import make_get_client, make_connect_error_client
from shared.models.factor import NormalizedFactor


async def test_ingest_happy(monkeypatch, sample_normalized_factor):
    payload = {"factors": [sample_normalized_factor.model_dump(mode="json")]}
    monkeypatch.setattr("main_controller.src.clients.base.get_client", make_get_client(payload))
    client = FactorLedgeClient()
    results = await client.ingest("art-001", ["institutional flows"], "aihub")
    assert len(results) == 1
    assert isinstance(results[0], NormalizedFactor)


async def test_http_error_raises(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({}, status_code=503),
    )
    client = FactorLedgeClient()
    with pytest.raises(FactorLedgeClientError):
        await client.ingest("art-001", ["factor"], "aihub")
