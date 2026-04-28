"""Generic read access to Supabase via PostgREST (REST API).

Environment variables (when using :meth:`SupabaseReadService.from_env`)::

    SUPABASE_URL — project URL, e.g. https://xxxx.supabase.co
    SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY — API key with read access
    SUPABASE_TABLE — optional default table name (default: news_articles)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SupabaseReadService:
    """Async client for ``GET`` queries against ``/rest/v1/{table}``."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        default_table: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._default_table = default_table
        self._timeout = timeout

    @classmethod
    def from_env(
        cls,
        *,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        default_table: str | None = None,
        timeout: float = 20.0,
    ) -> SupabaseReadService | None:
        base = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        key = (
            supabase_key
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
            or ""
        )
        tbl: str | None
        if default_table is not None:
            tbl = default_table
        else:
            tbl = os.getenv("SUPABASE_TABLE", "news_articles")
        if not base or not key:
            return None
        return cls(base, key, default_table=tbl, timeout=timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
        }

    def _resolve_table(self, table: str | None) -> str:
        resolved = table or self._default_table
        if not resolved:
            raise ValueError("table is required when default_table is not set")
        return resolved

    async def ping(self, table: str | None = None) -> bool:
        """Return True if PostgREST responds 2xx for a minimal select on the table."""
        tbl = self._resolve_table(table)
        endpoint = f"{self._base}/rest/v1/{tbl}"
        params = {"select": "*", "limit": "1"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(endpoint, headers=self._headers(), params=params)
        except httpx.HTTPError:
            return False
        return 200 <= resp.status_code < 300

    async def select_rows(
        self,
        table: str | None = None,
        *,
        columns: str = "*",
        order: str | None = None,
        limit: int = 100,
        extra_params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read query; returns decoded JSON rows or an empty list on failure.

        ``extra_params`` are merged into the query string (PostgREST filters, etc.).
        """
        tbl = self._resolve_table(table)
        endpoint = f"{self._base}/rest/v1/{tbl}"
        params: dict[str, str] = {"select": columns, "limit": str(max(1, limit))}
        if order:
            params["order"] = order
        if extra_params:
            params.update(extra_params)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(endpoint, headers=self._headers(), params=params)
        except httpx.HTTPError as exc:
            logger.warning("Supabase select HTTP error: %s", str(exc)[:200])
            return []

        if not (200 <= resp.status_code < 300):
            logger.warning(
                "Supabase select failed status=%s body=%s",
                resp.status_code,
                (resp.text or "")[:200],
            )
            return []

        data = resp.json()
        if not isinstance(data, list):
            return []

        out: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                out.append(item)
        return out
