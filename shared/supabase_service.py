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


def _parse_content_range_total(header_val: str) -> int | None:
    """Extract total row count from PostgREST ``Content-Range`` (e.g. ``0-0/357``)."""
    if not header_val or "/" not in header_val:
        return None
    total_part = header_val.split("/", 1)[1].strip()
    if total_part == "*":
        return None
    try:
        return int(total_part)
    except ValueError:
        return None


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
        self._http: httpx.AsyncClient | None = None

    def _ensure_http(self) -> httpx.AsyncClient:
        """Reuse one client per instance so connections stay warm (lower latency vs new client/request)."""
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=12, max_keepalive_connections=6),
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

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
            client = self._ensure_http()
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
        offset: int = 0,
        extra_params: dict[str, str] | None = None,
        extra_duplicate_params: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read query; returns decoded JSON rows or an empty list on failure.

        ``extra_params`` are merged into the query string (PostgREST filters, etc.).
        ``extra_duplicate_params`` appends extra ``(key, value)`` pairs so the same key
        can appear twice (e.g. two ``publish_at`` filters).
        ``offset`` is PostgREST row offset (paired with ``limit`` for paging).
        """
        tbl = self._resolve_table(table)
        endpoint = f"{self._base}/rest/v1/{tbl}"
        params_kv: list[tuple[str, str]] = [
            ("select", columns),
            ("limit", str(max(1, limit))),
        ]
        if order:
            params_kv.append(("order", order))
        if extra_params:
            params_kv.extend((k, str(v)) for k, v in extra_params.items())
        if extra_duplicate_params:
            params_kv.extend(extra_duplicate_params)
        if offset > 0:
            params_kv.append(("offset", str(int(offset))))

        try:
            client = self._ensure_http()
            resp = await client.get(endpoint, headers=self._headers(), params=params_kv)
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

    async def count_rows(
        self,
        table: str | None = None,
        *,
        columns: str = "id",
        order: str | None = None,
        extra_params: dict[str, str] | None = None,
        extra_duplicate_params: list[tuple[str, str]] | None = None,
    ) -> int | None:
        """Return total matching rows (``Prefer: count=exact``); None if unavailable."""
        tbl = self._resolve_table(table)
        endpoint = f"{self._base}/rest/v1/{tbl}"
        params_kv: list[tuple[str, str]] = [
            ("select", columns),
            ("limit", "1"),
        ]
        if order:
            params_kv.append(("order", order))
        if extra_params:
            params_kv.extend((k, str(v)) for k, v in extra_params.items())
        if extra_duplicate_params:
            params_kv.extend(extra_duplicate_params)
        headers = {
            **self._headers(),
            "Prefer": "count=exact",
        }
        try:
            client = self._ensure_http()
            resp = await client.get(endpoint, headers=headers, params=params_kv)
        except httpx.HTTPError as exc:
            logger.warning("Supabase count HTTP error: %s", str(exc)[:200])
            return None

        if not (200 <= resp.status_code < 300):
            logger.warning(
                "Supabase count failed status=%s body=%s",
                resp.status_code,
                (resp.text or "")[:200],
            )
            return None
        return _parse_content_range_total(resp.headers.get("content-range") or "")
