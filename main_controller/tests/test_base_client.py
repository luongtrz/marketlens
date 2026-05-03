"""Tests for BaseHTTPClient."""

import pytest
import httpx
from unittest.mock import MagicMock

from main_controller.src.clients.base import BaseHTTPClient
from main_controller.src.clients.exceptions import ClientError
from main_controller.tests.conftest import make_get_client, make_connect_error_client


class _TestClient(BaseHTTPClient):
    def __init__(self):
        super().__init__("http://test.local", ClientError)


async def test_get_happy_path(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({"key": "value"}),
    )
    client = _TestClient()
    result = await client._get("/test")
    assert result == {"key": "value"}


async def test_post_happy_path(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({"ok": True}),
    )
    client = _TestClient()
    result = await client._post("/test", {"data": 1})
    assert result == {"ok": True}


async def test_http_4xx_raises_client_error(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({}, status_code=404),
    )
    client = _TestClient()
    with pytest.raises(ClientError, match="GET"):
        await client._get("/not-found")


async def test_http_5xx_raises_client_error(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_get_client({}, status_code=500),
    )
    client = _TestClient()
    with pytest.raises(ClientError, match="GET"):
        await client._get("/error")


async def test_connect_error_raises_client_error(monkeypatch):
    monkeypatch.setattr(
        "main_controller.src.clients.base.get_client",
        make_connect_error_client(),
    )
    client = _TestClient()
    with pytest.raises(ClientError, match="Connection refused"):
        await client._get("/down")
