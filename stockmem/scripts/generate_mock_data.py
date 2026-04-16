"""
Generate synthetic 3-year BTC data for StockMem weight optimization.

Produces two JSON outputs:
  - records JSON: StockMemRecord payloads (for DB ingest / API smoke tests)
  - optimizer JSON: pre-vectorized rows (factor_vec 75d, indicator_vec 5d,
    price_vec 60d, future_return_{1,7,30}d) consumable by optimize_weights.py

Price model: geometric Brownian motion with a 3-state regime switch
(bull / bear / sideways). Intraday OHLC drawn from a micro random walk so
high/low/range reflect realistic intraday volatility. Volume log-normal
with autocorrelation.

Factors per day: 1–5 sampled from the StockMem taxonomy,
biased 40 % bullish / 40 % bearish / 20 % neutral by the requested baseline.
Bias shifts with regime: bull days favour bullish factors, bear days bearish.

Derived indicators:
  - rsi  (14)                 : Wilder-smoothed gains/losses
  - msi  (Market Sentiment)   : regime-anchored, 0–100
  - fear_greed_index          : 0–100, inversely correlated with realized vol
  - sentiment_score           : -1..1, biased by factor mix + regime
  - price_change_pct          : from close series

Future returns: forward-looking %-change over 1 / 7 / 30 trading days.

Usage:
  python scripts/generate_mock_data.py --days 1095 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from stockmem.src.models import CandleData, MarketSnapshot, StockMemRecord
from stockmem.src.search.embedder import RecordEmbedder, RETURNS_WINDOW
from stockmem.src.search.taxonomy import (
    BULLISH_FACTORS,
    BEARISH_FACTORS,
    NEUTRAL_FACTORS,
)


REGIME_PARAMS = {
    # (daily drift, daily vol). Calibrated to crypto-like regimes.
    "bull": (0.0035, 0.030),
    "bear": (-0.0030, 0.045),
    "side": (0.0000, 0.020),
}
REGIME_TRANSITION = {
    # mean dwell ≈ 60 trading days per regime
    "bull": {"bull": 0.98, "side": 0.015, "bear": 0.005},
    "side": {"bull": 0.01, "side": 0.98, "bear": 0.01},
    "bear": {"bull": 0.005, "side": 0.015, "bear": 0.98},
}

FACTOR_BIAS_BY_REGIME = {
    "bull": {"bullish": 0.65, "bearish": 0.15, "neutral": 0.20},
    "side": {"bullish": 0.40, "bearish": 0.40, "neutral": 0.20},
    "bear": {"bullish": 0.15, "bearish": 0.65, "neutral": 0.20},
}

BULLISH_KEYS = list(BULLISH_FACTORS.keys())
BEARISH_KEYS = list(BEARISH_FACTORS.keys())
NEUTRAL_KEYS = list(NEUTRAL_FACTORS.keys())


def _next_regime(current: str, rng: random.Random) -> str:
    choices = list(REGIME_TRANSITION[current].items())
    names, probs = zip(*choices)
    return rng.choices(names, weights=probs, k=1)[0]


def _sample_factors(regime: str, rng: random.Random) -> list[str]:
    n = rng.choices([1, 2, 3, 4, 5], weights=[25, 35, 25, 10, 5], k=1)[0]
    bias = FACTOR_BIAS_BY_REGIME[regime]
    buckets = rng.choices(
        ["bullish", "bearish", "neutral"],
        weights=[bias["bullish"], bias["bearish"], bias["neutral"]],
        k=n,
    )
    chosen: list[str] = []
    for b in buckets:
        pool = {
            "bullish": BULLISH_KEYS,
            "bearish": BEARISH_KEYS,
            "neutral": NEUTRAL_KEYS,
        }[b]
        pick = rng.choice(pool)
        if pick not in chosen:
            chosen.append(pick)
    return chosen


def _simulate_prices(days: int, seed: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    rng_regime = random.Random(seed ^ 0xA5A5)

    regime = "side"
    regimes: list[str] = []
    opens = np.zeros(days)
    highs = np.zeros(days)
    lows = np.zeros(days)
    closes = np.zeros(days)
    volumes = np.zeros(days)

    close_prev = 30_000.0
    log_vol_prev = math.log(1.5e10)

    for t in range(days):
        regime = _next_regime(regime, rng_regime)
        regimes.append(regime)
        mu, sigma = REGIME_PARAMS[regime]

        # Daily log return
        daily_logret = rng.normal(mu - 0.5 * sigma * sigma, sigma)
        close = close_prev * math.exp(daily_logret)

        # Intraday micro-walk: 24 hourly ticks with sigma/sqrt(24)
        intraday_sigma = sigma / math.sqrt(24)
        path = [close_prev]
        for _ in range(24):
            step = rng.normal(0.0, intraday_sigma)
            path.append(path[-1] * math.exp(step))
        # Final tick snaps to `close` to keep close-to-close consistent.
        path[-1] = close

        open_px = path[0]
        high_px = max(path)
        low_px = min(path)

        # Volume: log-normal with regime offset + autocorrelation.
        vol_drift = {"bull": 0.02, "bear": 0.03, "side": 0.0}[regime]
        log_vol = 0.85 * log_vol_prev + 0.15 * (math.log(1.5e10) + vol_drift) + rng.normal(0.0, 0.15)

        opens[t] = open_px
        highs[t] = high_px
        lows[t] = low_px
        closes[t] = close
        volumes[t] = math.exp(log_vol)

        close_prev = close
        log_vol_prev = log_vol

    ohlcv = np.vstack([opens, highs, lows, closes, volumes]).T  # (days, 5)
    return ohlcv, regimes, closes


def _rsi_wilder(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss <= 1e-12:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _fgi(realized_vol: np.ndarray, rsi: np.ndarray) -> np.ndarray:
    # Low vol + high RSI → greed; high vol + low RSI → fear.
    vol_scaled = np.clip((realized_vol - 0.01) / 0.05, 0.0, 1.0)
    rsi_scaled = np.clip(rsi / 100.0, 0.0, 1.0)
    raw = 0.6 * rsi_scaled + 0.4 * (1.0 - vol_scaled)
    return np.clip(raw * 100.0, 0.0, 100.0)


def _realized_vol(closes: np.ndarray, window: int = 20) -> np.ndarray:
    n = len(closes)
    rv = np.zeros(n)
    logret = np.diff(np.log(closes + 1e-12))
    for i in range(n):
        start = max(0, i - window)
        seg = logret[start:i]
        rv[i] = float(seg.std()) if seg.size >= 2 else 0.02
    return rv


def _msi(regime: str, rsi_val: float, rng: random.Random) -> float:
    anchor = {"bull": 65.0, "side": 50.0, "bear": 35.0}[regime]
    rsi_delta = (rsi_val - 50.0) * 0.3
    noise = rng.gauss(0.0, 5.0)
    return float(max(0.0, min(100.0, anchor + rsi_delta + noise)))


def _sentiment_score(factors: list[str], regime: str, rng: random.Random) -> float:
    bull_ct = sum(1 for f in factors if f in BULLISH_FACTORS)
    bear_ct = sum(1 for f in factors if f in BEARISH_FACTORS)
    if factors:
        factor_signal = (bull_ct - bear_ct) / len(factors)
    else:
        factor_signal = 0.0
    regime_signal = {"bull": 0.35, "side": 0.0, "bear": -0.35}[regime]
    noise = rng.gauss(0.0, 0.15)
    return float(max(-1.0, min(1.0, 0.6 * factor_signal + 0.3 * regime_signal + noise)))


def build_records(days: int, seed: int, symbol: str = "BTC") -> list[StockMemRecord]:
    ohlcv, regimes, closes = _simulate_prices(days, seed=seed)
    rsi_arr = _rsi_wilder(closes, period=14)
    rv = _realized_vol(closes, window=20)
    fgi_arr = _fgi(rv, rsi_arr)
    rng_py = random.Random(seed ^ 0x5A5A)

    start_day = date.today() - timedelta(days=days)

    records: list[StockMemRecord] = []
    for t in range(days):
        regime = regimes[t]
        factors = _sample_factors(regime, rng_py)

        snap_start = max(0, t - (RETURNS_WINDOW + 1) + 1)
        candles: list[CandleData] = []
        for i in range(snap_start, t + 1):
            candles.append(
                CandleData(
                    open=float(ohlcv[i, 0]),
                    high=float(ohlcv[i, 1]),
                    low=float(ohlcv[i, 2]),
                    close=float(ohlcv[i, 3]),
                    volume=float(ohlcv[i, 4]),
                )
            )

        if t == 0:
            pct_change = 0.0
        else:
            pct_change = (closes[t] - closes[t - 1]) / closes[t - 1] * 100.0

        future_1d = _forward_return(closes, t, 1)
        future_7d = _forward_return(closes, t, 7)
        future_30d = _forward_return(closes, t, 30)

        record = StockMemRecord(
            id=f"mock-{t:04d}",
            date=start_day + timedelta(days=t),
            symbol=symbol,
            sentiment_score=_sentiment_score(factors, regime, rng_py),
            factors=factors,
            market_snapshot=MarketSnapshot(
                rsi=float(rsi_arr[t]),
                macd_hist=0.0,
                msi=_msi(regime, float(rsi_arr[t]), rng_py),
                fear_greed_index=float(fgi_arr[t]),
                price_change_pct=float(pct_change),
                candles=candles,
            ),
            summary=f"regime={regime}",
            article_ids=[],
            future_return_1d=future_1d,
            future_return_7d=future_7d,
            future_return_30d=future_30d,
        )
        records.append(record)

    return records


def _forward_return(closes: np.ndarray, t: int, horizon: int) -> float:
    target = t + horizon
    if target >= len(closes):
        return 0.0
    return float((closes[target] - closes[t]) / closes[t] * 100.0)


def records_to_optimizer_rows(records: list[StockMemRecord]) -> list[dict]:
    embedder = RecordEmbedder()
    embedder.rebuild_corpus(records)

    rows: list[dict] = []
    for rec in records:
        split = embedder.embed_split(rec)
        rows.append(
            {
                "date": rec.date.isoformat(),
                "factor_vec": split.factor_vec.tolist(),
                "indicator_vec": split.indicator_vec.tolist(),
                "price_vec": split.price_vec.tolist(),
                "future_return_1d": rec.future_return_1d or 0.0,
                "future_return_7d": rec.future_return_7d or 0.0,
                "future_return_30d": rec.future_return_30d or 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic StockMem records + optimizer rows")
    parser.add_argument("--days", type=int, default=1095, help="Trading days (default: 3 years)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument(
        "--records-output",
        default="stockmem/data/mock_3y_records.json",
        help="Path for raw StockMemRecord JSON",
    )
    parser.add_argument(
        "--optimizer-output",
        default="stockmem/data/mock_3y_optimizer.json",
        help="Path for pre-vectorized optimizer JSON",
    )
    args = parser.parse_args()

    records = build_records(args.days, seed=args.seed, symbol=args.symbol)

    records_path = Path(args.records_output)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        json.dumps(
            [r.model_dump(mode="json") for r in records],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"[generate_mock_data] Wrote {len(records)} records → {records_path}")

    opt_rows = records_to_optimizer_rows(records)
    opt_path = Path(args.optimizer_output)
    opt_path.parent.mkdir(parents=True, exist_ok=True)
    opt_path.write_text(json.dumps(opt_rows, ensure_ascii=True), encoding="utf-8")
    print(f"[generate_mock_data] Wrote {len(opt_rows)} optimizer rows → {opt_path}")


if __name__ == "__main__":
    main()
