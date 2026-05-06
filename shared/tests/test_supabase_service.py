"""Tests for SupabaseReadService."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.supabase_service import SupabaseReadService


def test_from_env_returns_none_without_credentials() -> None:
    with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=True):
        assert SupabaseReadService.from_env() is None


def test_from_env_builds_client() -> None:
    env = {
        "SUPABASE_URL": "https://abc.supabase.co",
        "SUPABASE_ANON_KEY": "anon-test",
        "SUPABASE_TABLE": "my_table",
    }
    with patch.dict(os.environ, env, clear=True):
        svc = SupabaseReadService.from_env()
    assert svc is not None
    assert svc._base == "https://abc.supabase.co"
    assert svc._default_table == "my_table"


@pytest.mark.asyncio
async def test_select_rows_returns_parsed_list() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"id": "a"}, "skip", {"id": "b"}]

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("shared.supabase_service.httpx.AsyncClient", return_value=mock_client):
        svc = SupabaseReadService("https://x.supabase.co", "k", default_table="t")
        rows = await svc.select_rows(order="id.desc", limit=5)

    assert rows == [{"id": "a"}, {"id": "b"}]
    mock_client.get.assert_awaited_once()
    call_kwargs = mock_client.get.await_args.kwargs
    params_kv = call_kwargs["params"]
    assert isinstance(params_kv, list)
    assert ("order", "id.desc") in params_kv
    assert ("limit", "5") in params_kv


@pytest.mark.asyncio
async def test_select_rows_includes_offset_when_nonzero() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = []

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("shared.supabase_service.httpx.AsyncClient", return_value=mock_client):
        svc = SupabaseReadService("https://x.supabase.co", "k", default_table="t")
        await svc.select_rows(order="id.desc", limit=10, offset=40)

    params_kv = mock_client.get.await_args.kwargs["params"]
    assert ("offset", "40") in params_kv


@pytest.mark.asyncio
async def test_ping_false_on_http_error() -> None:
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("network", request=MagicMock()))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("shared.supabase_service.httpx.AsyncClient", return_value=mock_client):
        svc = SupabaseReadService("https://x.supabase.co", "k", default_table="t")
        assert await svc.ping() is False
