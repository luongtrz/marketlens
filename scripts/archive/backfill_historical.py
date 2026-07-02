"""
Backfill StockMem with historical BTC records from Supabase (2018-2021).

Concurrency model:
- Fetch Binance + Supabase per month (sequential, one batch each)
- Process all days in the month concurrently via asyncio.Semaphore(CONCURRENCY)
- Groq retry on 429 with exponential backoff
- Compact prompt (~320 tokens/call) → ~18 calls/min vs ~6/min with full prompt
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import asyncpg
import httpx

# ── Config ──────────────────────────────────────────────────────────────────
PG_DSN = "postgresql://postgres:pass@localhost:5432/postgres"
SUPABASE_URL = "https://esctepjpgpjgrcymnabx.supabase.co"
SUPABASE_KEY = ""
GROQ_KEY = ""
GROQ_MODEL = "llama-3.1-8b-instant"
BINANCE_URL = "https://api.binance.com/api/v3/klines"

SYMBOL = "BTC"
BINANCE_SYMBOL = "BTCUSDT"
START_DATE = date(2018, 1, 1)
END_DATE = date(2021, 12, 31)

# Concurrency: 5 parallel Groq calls — TPM-limited at ~18 calls/min with compact prompt
CONCURRENCY = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class Candle:
    ts: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def _rsi(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 50.0
    closes = [c.close for c in candles[-(period + 1):]]
    deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return round(100.0 - 100.0 / (1.0 + avg_gain / avg_loss), 2)


def _macd_hist(candles: list[Candle]) -> float:
    if len(candles) < 27:
        return 0.0
    closes = [c.close for c in candles]

    def _ema(data: list[float], p: int) -> float:
        k = 2.0 / (p + 1)
        v = data[0]
        for x in data[1:]:
            v = x * k + v * (1 - k)
        return v

    tail = closes[-26:]
    e12 = _ema(tail[-12:], 12)
    e26 = _ema(tail, 26)
    return round((e12 - e26) / closes[-1] * 100.0, 4) if closes[-1] else 0.0


# ── Supabase fetch ────────────────────────────────────────────────────────────

async def fetch_articles_for_month(
    client: httpx.AsyncClient,
    start: date,
    end: date,
) -> dict[date, list[dict]]:
    by_date: dict[date, list[dict]] = defaultdict(list)
    offset = 0
    start_str = f"{start.isoformat()}T00:00:00Z"
    end_str = f"{end.isoformat()}T23:59:59Z"

    while True:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/news_articles",
            params={
                "select": "id,header,publish_at,sentiment_score",
                "publish_at": f"gte.{start_str}",
                "order": "publish_at.asc",
                "limit": 1000,
                "offset": offset,
            },
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
        )
        resp.raise_for_status()
        rows = resp.json()
        filtered = [r for r in rows if r["publish_at"] <= end_str]
        for row in filtered:
            d = datetime.fromisoformat(row["publish_at"].replace("Z", "+00:00")).date()
            by_date[d].append(row)
        if len(rows) < 1000 or not rows or rows[-1]["publish_at"] > end_str:
            break
        offset += 1000

    return dict(by_date)


# ── Binance OHLCV ─────────────────────────────────────────────────────────────

async def fetch_binance_window(
    client: httpx.AsyncClient,
    start: date,
    end: date,
) -> dict[date, Candle]:
    fetch_start = start - timedelta(days=30)
    start_ts = int(datetime(fetch_start.year, fetch_start.month, fetch_start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int((datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000)
    all_candles: dict[date, Candle] = {}
    current = start_ts

    while True:
        resp = await client.get(
            BINANCE_URL,
            params={"symbol": BINANCE_SYMBOL, "interval": "1d", "limit": 1000, "startTime": current, "endTime": end_ts},
            timeout=30,
        )
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break
        for k in klines:
            d = datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).date()
            all_candles[d] = Candle(ts=d, open=float(k[1]), high=float(k[2]), low=float(k[3]), close=float(k[4]), volume=float(k[5]))
        if len(klines) < 1000:
            break
        current = int(klines[-1][0]) + 86_400_000

    return all_candles


# ── Groq factor extraction (compact prompt) ───────────────────────────────────

_SYSTEM = "You are a crypto analyst. Respond only with valid JSON, no markdown."
_VALID_TYPES = {"macro", "regulatory", "technical", "sentiment", "on_chain", "exchange"}


async def extract_factors_groq(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    target_date: date,
    articles: list[dict],
) -> list[dict]:
    # Headers only — compact prompt ~170 tokens input, ~150 tokens output
    headers = " | ".join((a.get("header") or "").strip() for a in articles[:8] if a.get("header"))
    prompt = (
        f'BTC news {target_date}: {headers}\n\n'
        f'List top 5 market factors as JSON: {{"factors":[{{"name":"...","type":"macro|regulatory|technical|sentiment|on_chain|exchange","polarity":-1.0}}]}}'
    )

    for attempt in range(5):
        async with sem:
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                    json={"model": GROQ_MODEL, "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}], "max_tokens": 300, "temperature": 0.1},
                    timeout=30,
                )
                if resp.status_code == 429:
                    # Groq token bucket resets in ~500ms; short backoff is correct here
                    wait = min(1.0 + attempt * 0.5, 3.0) + random.uniform(0, 0.5)
                    log.warning("  %s: rate limited, retry in %.1fs (attempt %d)", target_date, wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Extract JSON even if model adds extra text
                start = content.find("{")
                end = content.rfind("}") + 1
                if start == -1:
                    return []
                data = json.loads(content[start:end])
                factors = []
                for f in data.get("factors", []):
                    raw_type = str(f.get("type", "macro")).lower().replace("-", "_").replace(" ", "_")
                    if raw_type not in _VALID_TYPES:
                        raw_type = "macro"
                    name = str(f.get("name", "")).strip()
                    if name:
                        factors.append({
                            "name": name,
                            "type": raw_type,
                            "polarity": float(f.get("polarity", 0.0)),
                            "confidence": 0.8,
                        })
                return factors
            except asyncio.TimeoutError:
                log.warning("  %s: timeout attempt %d", target_date, attempt + 1)
                await asyncio.sleep(2 ** attempt)
            except Exception as exc:
                log.warning("  %s: error attempt %d: %s", target_date, attempt + 1, exc)
                await asyncio.sleep(1)
    return []


# ── Postgres upsert ───────────────────────────────────────────────────────────

async def upsert_record(pool: asyncpg.Pool, record_date: date, payload: dict) -> None:
    import hashlib
    date_str = record_date.isoformat()
    rid = hashlib.sha1(f"BTC:{date_str}".encode()).hexdigest()[:16]
    payload["id"] = rid
    payload["date"] = date_str
    payload["symbol"] = "BTC"

    async with pool.acquire() as conn:
        existing_raw = await conn.fetchval(
            "SELECT payload FROM stockmem_records WHERE record_date=$1 AND symbol='BTC'", date_str
        )
        if existing_raw:
            old = json.loads(existing_raw)
            for key in ("future_return_1d", "future_return_3d", "future_return_7d", "future_return_15d", "future_return_30d"):
                if old.get(key) is not None:
                    payload[key] = old[key]
            await conn.execute(
                "UPDATE stockmem_records SET payload=$1 WHERE record_date=$2 AND symbol='BTC'",
                json.dumps(payload, ensure_ascii=True), date_str,
            )
        else:
            await conn.execute(
                "INSERT INTO stockmem_records (id, record_date, symbol, payload) VALUES ($1,$2,$3,$4)",
                rid, date_str, "BTC", json.dumps(payload, ensure_ascii=True),
            )


# ── Process one day ────────────────────────────────────────────────────────────

async def process_day(
    d: date,
    articles: list[dict],
    candle: Candle,
    candle_list: list[Candle],
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    counters: dict,
) -> None:
    preceding = [c for c in candle_list if c.ts < d][-20:]
    all_prec = [c for c in candle_list if c.ts < d]

    rsi_val = _rsi(preceding + [candle])
    macd_val = _macd_hist(all_prec + [candle])
    price_chg = round((candle.close - preceding[-1].close) / preceding[-1].close * 100.0, 4) if preceding else 0.0
    scores = [float(a["sentiment_score"]) for a in articles if a.get("sentiment_score") and float(a["sentiment_score"]) != 0.0]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    label = "bullish" if avg_score > 0.15 else ("bearish" if avg_score < -0.15 else "neutral")
    msi_val = round(max(0.0, min(100.0, 50.0 + (rsi_val - 50.0) * 0.6 + avg_score * 15.0)), 2)

    factors = await extract_factors_groq(client, sem, d, articles)

    def c2d(c: Candle) -> dict:
        return {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}

    summary = " ".join((a.get("header") or "")[:120] for a in articles[:3] if a.get("header"))

    payload: dict[str, Any] = {
        "date": d.isoformat(),
        "symbol": "BTC",
        "sentiment_score": avg_score,
        "sentiment_label": label,
        "factors": [f["name"] for f in factors],
        "normalized_factors": [
            {"name": f["name"], "type": f["type"], "weight": f["confidence"], "polarity": f["polarity"],
             "source_article_id": str(articles[0]["id"]) if articles else "backfill",
             "observed_at": f"{d.isoformat()}T00:00:00+00:00"}
            for f in factors
        ],
        "factor_vector": [],
        "market_snapshot": {
            "symbol": "BTC",
            "timestamp": f"{d.isoformat()}T00:00:00+00:00",
            "ohlcv": c2d(candle),
            "recent_candles": [c2d(c) for c in preceding[-21:]],
            "indicators": {"rsi": rsi_val, "macd_hist": macd_val, "price_change_pct": price_chg, "msi": msi_val},
            "source": "binance",
        },
        "summary": summary,
        "article_ids": [str(a["id"]) for a in articles],
        "future_return_1d": None, "future_return_3d": None, "future_return_7d": None,
        "future_return_15d": None, "future_return_30d": None,
    }

    await upsert_record(pool, d, payload)
    counters["saved"] += 1
    log.info("  %s: %d articles → %d factors | RSI=%.1f price_chg=%.2f%%",
             d, len(articles), len(factors), rsi_val, price_chg)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    pool = await asyncpg.create_pool(PG_DSN, min_size=3, max_size=10)

    existing = await pool.fetch(
        "SELECT record_date FROM stockmem_records WHERE symbol='BTC' AND record_date >= $1 AND record_date <= $2",
        START_DATE.isoformat(), END_DATE.isoformat(),
    )
    existing_dates = {r["record_date"] for r in existing}
    log.info("Already have %d records in range, skipping those", len(existing_dates))

    sem = asyncio.Semaphore(CONCURRENCY)
    counters = {"saved": 0, "skipped": 0, "errors": 0}

    async with httpx.AsyncClient(timeout=30) as client:
        current = START_DATE.replace(day=1)

        while current <= END_DATE:
            # Month window
            month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            if month_end > END_DATE:
                month_end = END_DATE

            log.info("── %s → %s ──", current, month_end)

            try:
                all_candles = await fetch_binance_window(client, current, month_end)
                candle_list = sorted(all_candles.values(), key=lambda c: c.ts)
            except Exception as exc:
                log.error("Binance failed: %s", exc)
                current = month_end + timedelta(days=1)
                continue

            try:
                articles_by_date = await fetch_articles_for_month(client, current, month_end)
                log.info("  Supabase: %d days with articles", len(articles_by_date))
            except Exception as exc:
                log.error("Supabase failed: %s", exc)
                current = month_end + timedelta(days=1)
                continue

            # Build tasks for all days in month — run concurrently
            tasks = []
            d = current
            while d <= month_end:
                date_str = d.isoformat()
                if date_str in existing_dates:
                    d += timedelta(days=1)
                    continue
                candle = all_candles.get(d)
                articles = articles_by_date.get(d, [])
                if candle is None or not articles:
                    counters["skipped"] += 1
                    d += timedelta(days=1)
                    continue
                tasks.append(process_day(d, articles, candle, candle_list, pool, client, sem, counters))
                d += timedelta(days=1)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        log.error("Day failed: %s", r)
                        counters["errors"] += 1

            current = month_end + timedelta(days=1)

        log.info("Done — saved=%d skipped=%d errors=%d", counters["saved"], counters["skipped"], counters["errors"])

        row = await pool.fetchrow(
            "SELECT MIN(record_date) as mn, MAX(record_date) as mx, COUNT(*) as n FROM stockmem_records WHERE symbol='BTC'"
        )
        log.info("Stockmem BTC: %s → %s (%s total)", row["mn"], row["mx"], row["n"])

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
