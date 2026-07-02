"""Backfill 2022 BTC bear market data into StockMem.

No articles / Supabase needed — only Binance OHLCV (via market_data service).
Builds factor_vec from known 2022 macro events + price-action signals using
the local taxonomy (FACTOR_TYPE_MAP phrases → 75d binary vector).

kNN weights: factor(35%) + indicator(20%) + price(45%).
Re-saving existing records is safe — upsert preserves future_return_* values.

Usage:
    python scripts/backfill_2022_market.py [--start 2022-01-01] [--end 2022-12-31] [--dry-run]
    python scripts/backfill_2022_market.py --start 2021-11-01 --end 2022-12-31  # include Nov-Dec 2021
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from stockmem.src.search.taxonomy import build_type_vector, build_group_vector

# ---------------------------------------------------------------------------
# Config — adjust ports if needed
# ---------------------------------------------------------------------------
MARKET_URL = "http://localhost:8002"
STOCKMEM_URL = "http://localhost:8003"
FILL_RETURNS_URL = "http://localhost:8005/fill-returns?symbol=BTC"
SYMBOL = "BTC"
# Binance max per request is 1000 candles. We fetch in one shot for ≤1000d windows.
MAX_CANDLES = 1000


# ---------------------------------------------------------------------------
# Indicator helpers (same logic as main_controller/src/api.py)
# ---------------------------------------------------------------------------

def _rsi(candles: list[dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 50.0
    closes = [c["close"] for c in candles[-(period + 1):]]
    deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return round(100.0 - 100.0 / (1.0 + avg_gain / avg_loss), 2)


def _macd_hist(candles: list[dict]) -> float:
    if len(candles) < 27:
        return 0.0
    closes = [c["close"] for c in candles]

    def _ema(data: list[float], p: int) -> float:
        k = 2.0 / (p + 1)
        v = data[0]
        for x in data[1:]:
            v = x * k + v * (1 - k)
        return v

    tail = closes[-26:]
    e12 = _ema(tail[-12:], 12)
    e26 = _ema(tail, 26)
    last_price = closes[-1] or 1.0
    return round((e12 - e26) / last_price * 100.0, 4)


# ---------------------------------------------------------------------------
# Factor vector builder — known 2022 macro events + price-action signals
# ---------------------------------------------------------------------------

# Key 2022 BTC macro events (exact phrases from BEARISH_FACTORS in taxonomy.py)
_FED_HIKE_DATES = [
    date(2022, 3, 16), date(2022, 5, 4), date(2022, 6, 15),
    date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
]
_LUNA_WINDOW    = (date(2022, 5, 7),  date(2022, 5, 20))
_CELSIUS_WINDOW = (date(2022, 6, 10), date(2022, 7, 10))
_3AC_WINDOW     = (date(2022, 6, 14), date(2022, 7, 20))
_FTX_WINDOW     = (date(2022, 11, 6), date(2022, 11, 25))


def _get_2022_factors(td: date, ohlcv: dict, preceding: list[dict]) -> list[str]:
    """Return FACTOR_TYPE_MAP phrases appropriate for a 2022 bear market date."""
    factors: list[str] = []

    # ── Always present in 2022 bear market ──────────────────────────────────
    factors += [
        "Dollar index surging",         # DXY at 20-year high all year
        "CPI higher than expected",     # Inflation ran hot Jan-Dec 2022
        "Bond yield rising - risk for crypto",
    ]

    # ── Fed rate hikes (within ±7 days of each decision) ────────────────────
    if any(abs((td - fhd).days) <= 7 for fhd in _FED_HIKE_DATES):
        factors.append("Fed raises interest rate")

    # ── LUNA / Terra collapse ────────────────────────────────────────────────
    if _LUNA_WINDOW[0] <= td <= _LUNA_WINDOW[1]:
        factors += ["Terra Luna collapse impact", "Systemic risk concerns", "Large market liquidations"]

    # ── Celsius freeze ───────────────────────────────────────────────────────
    if _CELSIUS_WINDOW[0] <= td <= _CELSIUS_WINDOW[1]:
        factors.append("Celsius Network freezes withdrawals")

    # ── Three Arrows Capital (3AC) bankruptcy ────────────────────────────────
    if _3AC_WINDOW[0] <= td <= _3AC_WINDOW[1]:
        factors += ["Three Arrows Capital bankruptcy", "Systemic risk concerns"]

    # ── FTX collapse ─────────────────────────────────────────────────────────
    if _FTX_WINDOW[0] <= td <= _FTX_WINDOW[1]:
        factors += ["FTX exchange collapse event", "Exchange insolvency concerns", "Large market liquidations"]

    # ── Price-action signals ─────────────────────────────────────────────────
    if len(preceding) >= 7:
        close_7d = preceding[-7]["close"]
        ret_7d = (ohlcv["close"] - close_7d) / close_7d * 100 if close_7d else 0.0
        if ret_7d <= -5:
            factors += ["Strong whale selling", "Large market liquidations"]
        elif ret_7d >= 5:
            factors.append("Strong whale accumulation")

    rsi = _rsi(preceding + [ohlcv])
    if rsi < 30:
        factors.append("Miner selling pressure increasing")
    elif rsi > 70:
        factors.append("Significant volume surge")

    return factors


def _build_factor_vector(td: date, ohlcv: dict, preceding: list[dict]) -> list[float]:
    """Return 75d factor_vector for a 2022 date based on macro events + price action."""
    phrases = _get_2022_factors(td, ohlcv, preceding)
    tv = build_type_vector(phrases)   # 62d type bits
    gv = build_group_vector(phrases)  # 13d group bits
    return [float(x) for x in tv + gv]


# ---------------------------------------------------------------------------
# Fetch candles from market_data service
# ---------------------------------------------------------------------------

async def fetch_candles(
    client: httpx.AsyncClient,
    symbol: str,
    end_date: date,
    limit: int = MAX_CANDLES,
) -> list[dict]:
    """Fetch daily candles ending on end_date from market_data /history endpoint."""
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)
    end_ts = int(end_dt.timestamp() * 1000) - 1
    resp = await client.get(
        f"{MARKET_URL}/history",
        params={"symbol": symbol, "interval": "1d", "limit": limit, "end_time": str(end_ts)},
        timeout=30.0,
    )
    resp.raise_for_status()
    raw = resp.json()
    # market_data returns list of candle dicts with timestamp/open/high/low/close/volume
    return raw if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# Build and save one StockMemRecord
# ---------------------------------------------------------------------------

def _build_record(td: date, ohlcv: dict, preceding: list[dict], avg_score: float = 0.0) -> dict:
    """Build the JSON payload for POST /record on StockMem."""
    rsi_val = _rsi(preceding + [ohlcv])
    macd_val = _macd_hist(preceding + [ohlcv])
    price_chg = round(
        (ohlcv["close"] - preceding[-1]["close"]) / preceding[-1]["close"] * 100.0, 4
    ) if preceding else 0.0
    msi_val = round(max(0.0, min(100.0, 50.0 + (rsi_val - 50.0) * 0.6 + avg_score * 15.0)), 2)

    day_start = datetime(td.year, td.month, td.day, tzinfo=timezone.utc).isoformat()

    # Build recent_candles list (include the target candle last)
    window = (preceding[-20:] + [ohlcv]) if len(preceding) >= 20 else (preceding + [ohlcv])
    recent_candles = [
        {"open": c["open"], "high": c["high"], "low": c["low"],
         "close": c["close"], "volume": c["volume"], "timestamp": c["timestamp"],
         "interval": "1d"}
        for c in window
    ]

    factor_vec = _build_factor_vector(td, ohlcv, preceding)
    factor_phrases = _get_2022_factors(td, ohlcv, preceding)

    return {
        "record": {
            "date": td.isoformat(),
            "symbol": SYMBOL,
            "sentiment_score": -0.3,   # Mild bearish prior for all 2022 records
            "sentiment_label": "bearish",
            "factors": factor_phrases,
            "normalized_factors": [],
            "factor_vector": factor_vec,
            "market_snapshot": {
                "symbol": SYMBOL,
                "timestamp": day_start,
                "ohlcv": {
                    "open": ohlcv["open"],
                    "high": ohlcv["high"],
                    "low": ohlcv["low"],
                    "close": ohlcv["close"],
                    "volume": ohlcv["volume"],
                    "timestamp": ohlcv["timestamp"],
                    "interval": "1d",
                },
                "recent_candles": recent_candles,
                "indicators": {
                    "rsi": rsi_val,
                    "macd_hist": macd_val,
                    "price_change_pct": price_chg,
                    "msi": msi_val,
                    "fear_greed_index": 0.0,
                },
                "source": "binance",
            },
            "summary": f"BTC 2022 bear market {td}. Factors: {'; '.join(factor_phrases[:3])}",
            "article_ids": [],
        }
    }


async def save_record(client: httpx.AsyncClient, payload: dict) -> str:
    resp = await client.post(f"{STOCKMEM_URL}/record", json=payload, timeout=15.0)
    resp.raise_for_status()
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(start: date, end: date, dry_run: bool) -> None:
    print(f"Backfilling {SYMBOL} from {start} to {end} ({(end - start).days + 1} days)")
    if dry_run:
        print("[DRY RUN] No data will be saved.")

    async with httpx.AsyncClient() as client:
        # Fetch candles in chunks of MAX_CANDLES if window > 970 days
        # Always include 30 preceding days for the start of each window
        total_needed = (end - start).days + 1 + 30
        all_candles: list[dict] = []

        if total_needed <= MAX_CANDLES:
            print(f"Fetching {total_needed} daily candles ending {end} from market_data...")
            try:
                batch = await fetch_candles(client, SYMBOL, end, limit=total_needed)
                all_candles = batch
            except Exception as exc:
                print(f"ERROR fetching candles: {exc}")
                return
        else:
            # Fetch in two chunks: older window then newer window
            mid = start + timedelta(days=MAX_CANDLES // 2 - 30)
            print(f"Window > {MAX_CANDLES} days, fetching in 2 chunks...")
            for chunk_end in [mid, end]:
                try:
                    batch = await fetch_candles(client, SYMBOL, chunk_end, limit=MAX_CANDLES)
                    print(f"  chunk ending {chunk_end}: {len(batch)} candles")
                    all_candles.extend(batch)
                except Exception as exc:
                    print(f"  ERROR fetching chunk ending {chunk_end}: {exc}")

        candles = all_candles

        if not candles:
            print("ERROR: No candles returned — is market_data running?")
            return

        # Normalise candle dicts (market_data may return nested or flat format)
        def _norm(c: Any) -> dict:
            if isinstance(c, dict):
                # handle nested ohlcv
                ohlcv = c.get("ohlcv") or c
                return {
                    "open": float(ohlcv.get("open", 0)),
                    "high": float(ohlcv.get("high", 0)),
                    "low": float(ohlcv.get("low", 0)),
                    "close": float(ohlcv.get("close", 0)),
                    "volume": float(ohlcv.get("volume", 0)),
                    "timestamp": c.get("timestamp") or ohlcv.get("timestamp", ""),
                }
            return {}

        candles = [_norm(c) for c in candles if _norm(c)]
        print(f"  Got {len(candles)} candles, range: {candles[0]['timestamp'][:10]} → {candles[-1]['timestamp'][:10]}")

        # Index candles by date
        candle_by_date: dict[date, dict] = {}
        for c in candles:
            ts = c["timestamp"]
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00")).date() if isinstance(ts, str) else date.today()
                candle_by_date[d] = c
            except Exception:
                continue

        # Iterate each target date
        saved = 0
        skipped = 0
        errors = 0
        cur = start
        while cur <= end:
            ohlcv = candle_by_date.get(cur)
            if ohlcv is None:
                skipped += 1
                cur += timedelta(days=1)
                continue

            # Collect preceding candles (all candles before cur date, up to 26 for MACD)
            preceding = [c for d, c in sorted(candle_by_date.items()) if d < cur]

            if len(preceding) < 2:
                print(f"  SKIP {cur}: not enough preceding candles ({len(preceding)})")
                skipped += 1
                cur += timedelta(days=1)
                continue

            payload = _build_record(cur, ohlcv, preceding)

            if dry_run:
                print(f"  [DRY] {cur}  close={ohlcv['close']:.0f}  rsi={payload['record']['market_snapshot']['indicators']['rsi']}")
            else:
                try:
                    rid = await save_record(client, payload)
                    print(f"  SAVED {cur}  id={rid[:8]}  close={ohlcv['close']:.0f}", flush=True)
                    saved += 1
                except httpx.HTTPStatusError as exc:
                    body = exc.response.text[:200]
                    print(f"  ERROR {cur}: HTTP {exc.response.status_code} — {body}")
                    errors += 1
                except Exception as exc:
                    print(f"  ERROR {cur}: {exc}")
                    errors += 1

            cur += timedelta(days=1)

    print(f"\nDone. Saved={saved}  Skipped={skipped}  Errors={errors}")
    if saved > 0 and not dry_run:
        print(f"\nNext step: fill future returns")
        print(f"  curl -X POST '{FILL_RETURNS_URL}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill 2022 BTC bear market data into StockMem")
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2022-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Print records without saving")
    args = parser.parse_args()

    asyncio.run(main(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        dry_run=args.dry_run,
    ))
