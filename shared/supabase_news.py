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
# ``news_articles`` in Supabase stores only ``sentiment_score`` (numeric); label is derived in code.
_NEWS_LITE_COLUMNS = "id,header,source_url,publish_at,crawled_at,sentiment_score,summary,coin"


def _coerce_sentiment_float(value: object) -> float:
    """Interpret Supabase sentiment: prefers ``-1..1`` (model scale); accepts ``0..100`` UI scale."""

    if value is None:
        return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if -1.001 <= f <= 1.001:
        return max(-1.0, min(1.0, f))
    if 0.0 <= f <= 100.0:
        return max(-1.0, min(1.0, f / 50.0 - 1.0))
    return max(-1.0, min(1.0, f))


def _label_from_sentiment_float(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


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


def _normalize_coin_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        coin = str(item).strip()
        if not coin:
            continue
        if coin.upper() == "GENERAL":
            out.append("General")
        else:
            out.append(coin.upper())
    return sorted(set(out))


def supabase_row_to_ingestion(row: dict[str, Any]) -> IngestionRecord:
    """Map a PostgREST row from ``news_articles`` to :class:`IngestionRecord`."""
    source_url = str(row.get("source_url") or "")
    header = str(row.get("header") or "Untitled")
    content = str(row.get("content") or "")
    rid = str(row["id"]) if row.get("id") is not None else _stable_id(source_url)
    pub = _parse_dt(row.get("publish_at") or datetime.now(timezone.utc).isoformat())
    crawled_raw = row.get("crawled_at")
    crawled = _parse_dt(crawled_raw) if crawled_raw is not None else pub
    raw_sent = row.get("sentiment_score")
    if raw_sent is None and "sentiment" in row:
        raw_sent = row.get("sentiment")
    sentiment_score = _coerce_sentiment_float(raw_sent)
    sentiment_label = _label_from_sentiment_float(sentiment_score)
    db_summary = row.get("summary")
    if isinstance(db_summary, str) and db_summary.strip():
        summary_val = db_summary.strip()[:8000]
    else:
        summary_val = (content[:2000] if content else None)
    coin_tags = _normalize_coin_tags(row.get("coin"))
    return IngestionRecord(
        id=rid,
        article_name=header[:500],
        source=_source_from_url(source_url),
        url=source_url or "#",
        date_published=pub,
        date_crawled=crawled,
        summary=summary_val,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        factors=[],
        raw_text=content or None,
        metadata={"coin": coin_tags},
    )


def _row_text_matches_symbol(
    header: str,
    content: str,
    symbol: str,
    coin: object | None = None,
) -> bool:
    """Match articles to a trading pair, preferring Supabase ``coin`` tags when present."""

    sym = (symbol or "").upper().strip()
    coin_tags = _normalize_coin_tags(coin)
    if coin_tags:
        if "BTC" in sym:
            return "BTC" in coin_tags
        if "ETH" in sym:
            return "ETH" in coin_tags
        if "GENERAL" in sym:
            return "General" in coin_tags

    from shared.asset_tags import text_matches_symbol

    return text_matches_symbol(header, content, symbol)


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


def normalize_news_source_host(raw: str | None) -> str | None:
    """Return a lowercase hostname safe for ``source_url`` filters, or None if invalid."""

    if raw is None:
        return None
    s = raw.strip().lower()
    if not s:
        return None
    # PostgREST ilike payload: host segment only (no protocol/path).
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", s):
        return None
    return s


def _source_url_filter_params(source_host: str | None) -> dict[str, str]:
    hn = normalize_news_source_host(source_host or "")
    if not hn:
        return {}
    return {"source_url": f"ilike.%{hn}%"}


def _coin_filter_params(symbol: str) -> dict[str, str]:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {}
    if "BTC" in sym:
        return {"coin": "cs.{BTC}"}
    if "ETH" in sym:
        return {"coin": "cs.{ETH}"}
    if "GENERAL" in sym:
        return {"coin": "cs.{General}"}
    return {}


async def fetch_recent_news_source_hosts(
    *,
    scan_limit: int = 800,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    table: str | None = None,
    timeout: float = 20.0,
) -> list[str]:
    """Distinct crawler ``source`` hostnames derived from newest ``source_url`` rows.

    Not a global DISTINCT (PostgREST has no cheap generic); sufficiently diverse for UI filters.
    """

    svc = _news_service(supabase_url, supabase_key, table, timeout=timeout)
    if svc is None:
        return []

    rows = await svc.select_rows(
        order="publish_at.desc",
        limit=min(max(scan_limit, 1), _SYM_FILTER_SCAN_CAP),
        offset=0,
        columns="source_url",
    )
    hosts: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        url = raw.get("source_url")
        if isinstance(url, str) and url.strip():
            hosts.add(_source_from_url(url.strip()))
    hosts.discard("unknown")
    return sorted(hosts)


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
    source_host: str | None = None,
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
    url_host_extra = _source_url_filter_params(source_host)

    coin_extra = _coin_filter_params(symbol)
    if symbol and coin_extra:
        rows = await svc.select_rows(
            order="publish_at.desc",
            limit=max(1, limit),
            offset=max(0, offset),
            columns=_NEWS_LITE_COLUMNS if lite else "*",
            extra_params={**url_host_extra, **coin_extra} or None,
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
        return out

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
            extra_params=url_host_extra or None,
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
            if not _row_text_matches_symbol(
                rec.article_name,
                rec.raw_text or "",
                symbol,
                rec.metadata.get("coin"),
            ):
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
        extra_params=url_host_extra or None,
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
    source_host: str | None = None,
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
    url_host_extra = _source_url_filter_params(source_host)
    return await svc.count_rows(
        columns="id",
        order="publish_at.desc",
        extra_params=url_host_extra or None,
        extra_duplicate_params=dup_filters or None,
    )


async def count_news_articles_matching_symbol_from_supabase(
    *,
    symbol: str,
    publish_gte: datetime | None = None,
    publish_lte: datetime | None = None,
    source_host: str | None = None,
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
    url_host_extra = _source_url_filter_params(source_host)
    coin_extra = _coin_filter_params(symbol)

    if coin_extra:
        total = await svc.count_rows(
            columns="id",
            order="publish_at.desc",
            extra_params={**url_host_extra, **coin_extra} or None,
            extra_duplicate_params=dup_filters or None,
        )
        return int(total or 0)

    rows = await svc.select_rows(
        order="publish_at.desc",
        limit=_SYM_FILTER_SCAN_CAP,
        offset=0,
        columns=_NEWS_LITE_COLUMNS if lite else "*",
        extra_params=url_host_extra or None,
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
        if _row_text_matches_symbol(
            rec.article_name,
            rec.raw_text or "",
            symbol,
            rec.metadata.get("coin"),
        ):
            matched += 1

    return matched
