"""Read news rows from Supabase via PostgREST (same API the crawler uses for inserts)."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from shared.models.article import IngestionRecord
from shared.supabase_service import SupabaseReadService

logger = logging.getLogger(__name__)


def _news_service(
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    table: str | None = None,
    timeout: float = 20.0,
) -> SupabaseReadService | None:
    return SupabaseReadService.from_env(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        default_table=table,
        timeout=timeout,
    )


def _source_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc or ""
        return host.replace("www.", "") or "unknown"
    except Exception:
        return "unknown"


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    return datetime.now(timezone.utc)


def _stable_id(source_url: str) -> str:
    return hashlib.sha256(source_url.encode()).hexdigest()[:32]


def supabase_row_to_ingestion(row: dict[str, Any]) -> IngestionRecord:
    """Map a PostgREST row from ``news_articles`` to :class:`IngestionRecord`."""
    source_url = str(row.get("source_url") or "")
    header = str(row.get("header") or "Untitled")
    content = str(row.get("content") or "")
    rid = str(row["id"]) if row.get("id") is not None else _stable_id(source_url)
    pub = _parse_dt(row.get("publish_at") or datetime.now(timezone.utc).isoformat())
    crawled_raw = row.get("crawled_at")
    crawled = _parse_dt(crawled_raw) if crawled_raw is not None else pub
    return IngestionRecord(
        id=rid,
        article_name=header[:500],
        source=_source_from_url(source_url),
        url=source_url or "#",
        date_published=pub,
        date_crawled=crawled,
        summary=(content[:2000] if content else None),
        sentiment_score=0.0,
        sentiment_label="neutral",
        factors=[],
        raw_text=content or None,
        metadata={},
    )


def _row_text_matches_symbol(header: str, content: str, symbol: str) -> bool:
    sym = symbol.upper()
    text = f"{header} {content}".lower()
    if "BTC" in sym:
        return "btc" in text or "bitcoin" in text
    if "ETH" in sym:
        return "eth" in text or "ethereum" in text
    return True


async def check_supabase_rest_reachable(timeout: float = 10.0) -> bool:
    """Return True if PostgREST returns 2xx for a minimal ``select`` on the news table."""
    svc = _news_service(timeout=timeout)
    if svc is None:
        return False
    return await svc.ping()


async def fetch_news_articles_from_supabase(
    *,
    limit: int = 50,
    symbol: str = "",
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    table: str | None = None,
    timeout: float = 20.0,
) -> list[IngestionRecord]:
    """Fetch recent rows from the Supabase news table, newest first.

    Uses ``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY`` (or ``SUPABASE_ANON_KEY``)
    when arguments are omitted.
    """
    svc = _news_service(supabase_url, supabase_key, table, timeout=timeout)
    if svc is None:
        logger.warning("Supabase fetch skipped: missing URL or API key in environment")
        return []

    rows = await svc.select_rows(
        order="publish_at.desc",
        limit=limit,
    )

    records: list[IngestionRecord] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            rec = supabase_row_to_ingestion(raw)
        except Exception as exc:
            logger.warning("Skipping malformed Supabase row: %s", str(exc)[:120])
            continue
        if symbol and not _row_text_matches_symbol(
            rec.article_name, rec.raw_text or "", symbol
        ):
            continue
        records.append(rec)
        if len(records) >= limit:
            break

    return records
