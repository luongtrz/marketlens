"""Base HTTP client — shared request logic for all module clients."""

import json
from typing import Type

import httpx

from shared.http_client import get_client


class BaseHTTPClient:
    """Async HTTP client base with uniform error wrapping.

    Subclasses pass their error class to __init__; _get/_post wrap
    all httpx and parse errors into that class automatically.
    """

    def __init__(self, base_url: str, error_class: Type[Exception]) -> None:
        self._base_url = base_url.rstrip("/")
        self._error_class = error_class

    async def _get(self, path: str, **params) -> dict | list:
        return await self._request("GET", path, params=params)

    async def _post(self, path: str, json_body: dict) -> dict | list:
        return await self._request("POST", path, json=json_body)

    async def _patch(self, path: str, json_body: dict) -> dict | list:
        return await self._request("PATCH", path, json=json_body)

    async def _request(self, method: str, path: str, **kwargs) -> dict | list:
        url = f"{self._base_url}{path}"
        try:
            async with get_client() as client:
                resp = await client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            raise self._error_class(f"{method} {url} failed: {exc}") from exc
