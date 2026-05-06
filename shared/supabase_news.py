"""Read news rows from Supabase via PostgREST (same API the crawler uses for inserts)."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from shared.models.article import IngestionRecord
from shared.supabase_service import SupabaseReadService

logger = logging.getLogger(__name__)

# List views: omit ``content`` → much smaller payloads than ``select=*`` (major latency win).
_NEWS_LITE_COLUMNS = "id,header,source_url,publish_at,crawled_at"

# When a ``symbol`` text filter applies, pagination is bounded by scanning newest rows server-side,
# since PostgREST cannot express the whitelist token rules in ``_row_text_matches_symbol``.
_SYM_FILTER_SCAN_CAP = 2500


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
    """Match articles to a trading pair using whole tokens, not naive substrings.

    Substrings like ``"eth" in "tether"`` falsely tag stablecoin/podcast headlines as ETH.
    """
    sym = symbol.upper().strip()
    text = f"{header} {content}".strip()
    if not sym:
        return True
    if "BTC" in sym:
        return bool(re.search(r"\b(btc|bitcoin)\b", text, re.IGNORECASE))
    if "ETH" in sym:
        return bool(re.search(r"\b(eth|ethereum)\b", text, re.IGNORECASE))
    # Other pairs e.g. SOLUSDT → require the base ticker as a word.
    base = sym.replace("USDT", "").replace("USD", "").replace("BUSD", "")
    if base.isalpha() and 2 <= len(base) <= 8:
        return bool(re.search(rf"\b{re.escape(base)}\b", text, re.IGNORECASE))
    return False


def _fmt_postgrest_dt(dt: datetime) -> str:
    """RFC3339 UTC with ``Z`` suffix for PostgREST ``publish_at`` filters."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    s = dt.isoformat(timespec="seconds")
    return s.replace("+00:00", "Z")


async def check_supabase_rest_reachable(timeout: float = 10.0) -> bool:
    """Return True if PostgREST returns 2xx for a minimal ``select`` on the news table."""
    svc = _news_service(timeout=timeout)
    if svc is None:
        return False
    return await svc.ping()


def _publish_dup_filters(
    publish_gte: datetime | None,
    publish_lte: datetime | None,
) -> list[tuple[str, str]]:
    dup_filters: list[tuple[str, str]] = []
    if publish_gte is not None:
        dup_filters.append(("publish_at", f"gte.{_fmt_postgrest_dt(publish_gte)}"))
    if publish_lte is not None:
        dup_filters.append(("publish_at", f"lte.{_fmt_postgrest_dt(publish_lte)}"))
    return dup_filters


async def fetch_news_articles_from_supabase(
    *,
    limit: int = 50,
    offset: int = 0,
    symbol: str = "",
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    table: str | None = None,
    timeout: float = 20.0,
    lite: bool = False,
    publish_gte: datetime | None = None,
    publish_lte: datetime | None = None,
) -> list[IngestionRecord]:
    """Fetch rows from Supabase newest first via PostgREST.

    Uses ``SUPABASE_URL`` / ``SUPABASE_SERVICE_ROLE_KEY`` (or anon key).

    Without ``symbol``, ``limit`` + ``offset`` map to SQL ``LIMIT`` / ``OFFSET`` — full-table pagination.

    With ``symbol``, the token filter runs in Python: up to ``_SYM_FILTER_SCAN_CAP`` newest matching
    dates are scanned, filtered, then windowed ``[offset:offset + limit]`` (cheap for dashboards;
    totals may reflect that cap alone).
    """
    svc = _news_service(supabase_url, supabase_key, table, timeout=timeout)
    if svc is None:
        logger.warning("Supabase fetch skipped: missing URL or API key in environment")
        return []

    dup_filters = _publish_dup_filters(publish_gte, publish_lte)

    if symbol:
        # Need enough scanned rows before Python symbol filter — worst case (all scanned rows match pair).
        scan_floor = offset + limit + 50
        scan_target = min(_SYM_FILTER_SCAN_CAP, max(500, scan_floor))
        if publish_gte is not None or publish_lte is not None:
            scan_target = min(_SYM_FILTER_SCAN_CAP, max(scan_target, 900))

        rows = await svc.select_rows(
            order="publish_at.desc",
            limit=scan_target,
            offset=0,
            columns=_NEWS_LITE_COLUMNS if lite else "*",
            extra_duplicate_params=dup_filters or None,
        )
        filtered: list[IngestionRecord] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            try:
                rec = supabase_row_to_ingestion(raw)
            except Exception as exc:
                logger.warning("Skipping malformed Supabase row: %s", str(exc)[:120])
                continue
            if not _row_text_matches_symbol(rec.article_name, rec.raw_text or "", symbol):
                continue
            filtered.append(rec)
        chunk = filtered[offset : offset + limit]
        return chunk

    # No symbol → server-side paging.
    rows = await svc.select_rows(
        order="publish_at.desc",
        limit=max(1, limit),
        offset=max(0, offset),
        columns=_NEWS_LITE_COLUMNS if lite else "*",
        extra_duplicate_params=dup_filters or None,
    )

    out: list[IngestionRecord] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(supabase_row_to_ingestion(raw))
        except Exception as exc:
            logger.warning("Skipping malformed Supabase row: %s", str(exc)[:120])
            continue

    return out


async def count_news_articles_from_supabase(
    *,
    publish_gte: datetime | None = None,
    publish_lte: datetime | None = None,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    table: str | None = None,
    timeout: float = 20.0,
) -> int | None:
    """Total rows matching publish window (exact via PostgREST count). Missing symbol filter."""

    svc = _news_service(supabase_url, supabase_key, table, timeout=timeout)
    if svc is None:
        return None

    dup_filters = _publish_dup_filters(publish_gte, publish_lte)
    return await svc.count_rows(
        columns="id",
        order="publish_at.desc",
        extra_duplicate_params=dup_filters or None,
    )


async def count_news_articles_matching_symbol_from_supabase(
    *,
    symbol: str,
    publish_gte: datetime | None = None,
    publish_lte: datetime | None = None,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    table: str | None = None,
    timeout: float = 20.0,
    lite: bool = True,
) -> int:
    """Count articles matching Python symbol filter among up to ``_SYM_FILTER_SCAN_CAP`` newest rows."""

    svc = _news_service(supabase_url, supabase_key, table, timeout=timeout)
    if svc is None or not symbol.strip():
        return 0

    dup_filters = _publish_dup_filters(publish_gte, publish_lte)
    rows = await svc.select_rows(
        order="publish_at.desc",
        limit=_SYM_FILTER_SCAN_CAP,
        offset=0,
        columns=_NEWS_LITE_COLUMNS if lite else "*",
        extra_duplicate_params=dup_filters or None,
    )
    matched = 0
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            rec = supabase_row_to_ingestion(raw)
        except Exception:
            continue
        if _row_text_matches_symbol(rec.article_name, rec.raw_text or "", symbol):
            matched += 1

    return matched
