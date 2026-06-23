"""ETH kNN-returns evaluation using BTC factor vectors as proxy.

Strategy:
  - price_vec   (60d): ETH OHLCV from Binance  → ETH-specific ✓
  - indicator_vec (5d): ETH RSI + price_change_pct (computed)
                        + msi / fear_greed / sentiment borrowed from BTC records (crypto-wide)
  - factor_vec  (75d): BTC factor_vector for same date (macro proxy)

No DB writes — runs entirely in-memory.

Usage:
    python scripts/eval_eth_proxy.py
    python scripts/eval_eth_proxy.py --buy-thr 2 --sell-thr 2 --k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import httpx
import numpy as np

DB_URL       = "postgresql://postgres:pass@localhost:5432/postgres"
BINANCE_URL  = "https://api.binance.com/api/v3/klines"
WEIGHTS_FILE = Path(__file__).parent.parent / "stockmem" / "config" / "weights.auto.json"
RETURN_W     = {"1d": 0.40, "3d": 0.30, "7d": 0.15, "15d": 0.10, "30d": 0.05}
HORIZONS     = list(RETURN_W.keys())
HORIZON_DAYS = {"1d": 1, "3d": 3, "7d": 7, "15d": 15, "30d": 30}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _load_search_weights() -> tuple[float, float, float]:
    if WEIGHTS_FILE.exists():
        w = json.loads(WEIGHTS_FILE.read_text())["weights"]
        return w["w1_factor"], w["w2_indicator"], w["w3_price"]
    return 0.35, 0.20, 0.45


# ── RSI ───────────────────────────────────────────────────────────────────────

def compute_rsi_series(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Returns RSI array same length as closes (first `period` values = 50.0)."""
    rsi = np.full(len(closes), 50.0)
    if len(closes) <= period:
        return rsi
    deltas = np.diff(closes)
    gains  = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_g  = np.mean(gains[:period])
    avg_l  = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i])  / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / (avg_l + 1e-8)
        rsi[i + 1] = 100 - 100 / (1 + rs)
    return rsi


# ── Binance ───────────────────────────────────────────────────────────────────

async def fetch_eth_ohlcv(start_date: str = "2022-01-01") -> list[dict]:
    """Fetch all ETH daily candles from Binance since start_date."""
    import time as _time
    start_ms = int(
        __import__("datetime").datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=__import__("datetime").timezone.utc)
        .timestamp() * 1000
    )
    candles: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as http:
        while True:
            r = await http.get(BINANCE_URL, params={
                "symbol": "ETHUSDT", "interval": "1d",
                "startTime": start_ms, "limit": 1000,
            })
            data = r.json()
            if not data or not isinstance(data, list):
                break
            for c in data:
                candles.append({
                    "date":   __import__("datetime").datetime.fromtimestamp(
                        c[0] / 1000, tz=__import__("datetime").timezone.utc
                    ).strftime("%Y-%m-%d"),
                    "open":   float(c[1]),
                    "high":   float(c[2]),
                    "low":    float(c[3]),
                    "close":  float(c[4]),
                    "volume": float(c[5]),
                })
            if len(data) < 1000:
                break
            start_ms = data[-1][0] + 86_400_000
    candles.sort(key=lambda c: c["date"])
    return candles


# ── Load BTC proxy data from DB ───────────────────────────────────────────────

async def load_btc_proxy() -> dict[str, dict]:
    """Returns {date: {factor_vector, msi, fear_greed_index, sentiment_score}}."""
    conn = await asyncpg.connect(DB_URL)
    rows = await conn.fetch(
        "SELECT record_date, payload FROM stockmem_records WHERE symbol='BTC' ORDER BY record_date"
    )
    await conn.close()

    out: dict[str, dict] = {}
    for row in rows:
        p  = json.loads(row["payload"])
        ms = p.get("market_snapshot", {})
        out[str(row["record_date"])] = {
            "factor_vector":    p.get("factor_vector", []),
            "msi":              float(ms.get("msi") or 0.0),
            "fear_greed_index": float(ms.get("fear_greed_index") or 50.0),
            "sentiment_score":  float(p.get("sentiment_score") or 0.0),
        }
    return out


# ── Build ETH records ─────────────────────────────────────────────────────────

@dataclass
class EthRecord:
    date: str
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray
    future: dict[str, float | None]
    borrowed_factor: bool  # True = using BTC proxy


def build_eth_records(
    candles: list[dict],
    btc_proxy: dict[str, dict],
) -> list[EthRecord]:
    closes  = np.array([c["close"]  for c in candles], np.float32)
    highs   = np.array([c["high"]   for c in candles], np.float32)
    lows    = np.array([c["low"]    for c in candles], np.float32)
    volumes = np.array([c["volume"] for c in candles], np.float32)
    rsi_series = compute_rsi_series(closes)
    n = len(candles)

    date_to_idx = {c["date"]: i for i, c in enumerate(candles)}

    def _future_ret(i: int, days: int) -> float | None:
        j = i + days
        if j >= n:
            return None
        base = closes[i]
        if base < 1e-8:
            return None
        return round(float((closes[j] - base) / base * 100), 4)

    def _price_vec(i: int) -> np.ndarray:
        start = max(0, i - 20)
        c_window = closes[start : i + 1]
        h_window = highs[start  : i + 1]
        l_window = lows[start   : i + 1]
        v_window = volumes[start: i + 1]
        if len(c_window) < 2:
            return np.zeros(60, np.float32)
        rets   = np.diff(c_window) / (c_window[:-1] + 1e-8)
        ranges = (h_window - l_window) / (c_window + 1e-8)
        vchs   = np.diff(v_window) / (v_window[:-1] + 1e-8)

        def tail20(a: np.ndarray) -> np.ndarray:
            t = a[-20:]
            return np.pad(t, (20 - len(t), 0)).astype(np.float32)

        return _l2(np.concatenate([tail20(rets), tail20(ranges[1:]), tail20(vchs)]))

    records: list[EthRecord] = []
    missing_factor = 0

    for i, c in enumerate(candles):
        date = c["date"]

        # Factor vec: borrow from BTC same date
        btc = btc_proxy.get(date)
        borrowed = btc is not None and len(btc.get("factor_vector", [])) == 75
        if borrowed:
            fv = _l2(np.array(btc["factor_vector"], np.float32))
        else:
            fv = np.zeros(75, np.float32)
            missing_factor += 1

        # Indicator vec: ETH RSI + price_change_pct, BTC msi/fgi/sentiment
        pcp = float((closes[i] - closes[i - 1]) / (closes[i - 1] + 1e-8) * 100) if i > 0 else 0.0
        msi = btc["msi"]              if btc else 0.0
        fgi = btc["fear_greed_index"] if btc else 50.0
        snt = btc["sentiment_score"]  if btc else 0.0
        raw = np.array([msi, float(rsi_series[i]), snt, fgi, pcp], np.float32)
        mean, std = raw.mean(), raw.std() + 1e-8
        iv = _l2((raw - mean) / std)

        # Price vec: ETH-specific
        pv = _price_vec(i)

        # Future returns
        future = {h: _future_ret(i, HORIZON_DAYS[h]) for h in HORIZONS}

        records.append(EthRecord(
            date=date,
            factor_vec=fv,
            indicator_vec=iv,
            price_vec=pv,
            future=future,
            borrowed_factor=borrowed,
        ))

    if missing_factor:
        print(f"  Missing BTC proxy for {missing_factor} dates (factor_vec = zeros)")
    return records


# ── Evaluation (same logic as eval_knn_returns.py) ───────────────────────────

_RETURN_WEIGHTS = np.array([RETURN_W[h] for h in HORIZONS], np.float64)


def _knn_avg(rec: EthRecord, weights: dict[str, float]) -> float | None:
    total_w = total_v = 0.0
    for h, w in weights.items():
        v = rec.future.get(h)
        if v is not None:
            total_v += v * w
            total_w += w
    return total_v / total_w if total_w > 0 else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def evaluate(
    records: list[EthRecord],
    k: int,
    return_weights: dict[str, float],
    buy_thr: float,
    sell_thr: float,
    eval_horizon: str,
    sw1: float, sw2: float, sw3: float,
) -> None:
    total = skip_nn = skip_ret = 0
    results: list[dict] = []

    print(f"\nEvaluating {len(records)} ETH dates  k={k}  thr=±{buy_thr}%  "
          f"search_w=({sw1:.3f}/{sw2:.3f}/{sw3:.3f})", flush=True)

    for i, qry in enumerate(records):
        prior = records[:i]
        if len(prior) < k:
            skip_nn += 1
            continue

        sims = []
        for p in prior:
            score = (sw1 * _cosine(qry.factor_vec,    p.factor_vec) +
                     sw2 * _cosine(qry.indicator_vec,  p.indicator_vec) +
                     sw3 * _cosine(qry.price_vec,      p.price_vec))
            sims.append((score, p))
        sims.sort(key=lambda x: x[0], reverse=True)
        top_k = [r for _, r in sims[:k]]

        avgs = [a for r in top_k if (a := _knn_avg(r, return_weights)) is not None]
        if not avgs:
            skip_nn += 1
            continue

        overall = sum(avgs) / len(avgs)
        sig = "BUY" if overall > buy_thr else "SELL" if overall < -sell_thr else "HOLD"

        actual = qry.future.get(eval_horizon)
        if actual is None:
            skip_ret += 1
            continue

        if sig == "BUY":
            correct = actual > 0
        elif sig == "SELL":
            correct = actual < 0
        else:
            correct = -buy_thr <= actual <= buy_thr

        results.append({"signal": sig, "avg": overall, "actual": actual,
                         "correct": correct, "date": qry.date})
        total += 1

    buy_res  = [r for r in results if r["signal"] == "BUY"]
    sell_res = [r for r in results if r["signal"] == "SELL"]
    hold_res = [r for r in results if r["signal"] == "HOLD"]

    def _da(lst):
        return sum(1 for r in lst if r["correct"]) / len(lst) * 100 if lst else 0
    def _avg(lst):
        return sum(r["actual"] for r in lst) / len(lst) if lst else 0

    da_buy, da_sell, da_hold = _da(buy_res), _da(sell_res), _da(hold_res)
    da_all = (sum(1 for r in results if r["correct"]) / total * 100) if total else 0
    coverage = (len(buy_res) + len(sell_res)) / total * 100 if total else 0

    print(f"\n{'='*62}")
    print(f"  ETH  |  Eval horizon: D+{eval_horizon}  |  Dates: {total}")
    print(f"  Skipped (too early / no data): {skip_nn + skip_ret}")
    print(f"{'='*62}")
    print(f"  Signal   Count    Share   DA                   Avg actual")
    print(f"  -------  -----   ------  -------------------  ----------")
    print(f"  BUY      {len(buy_res):5d}   {len(buy_res)/total*100:5.1f}%  {da_buy:5.1f}% ({sum(1 for r in buy_res if r['correct'])}/{len(buy_res)})     {_avg(buy_res):+.2f}%")
    print(f"  SELL     {len(sell_res):5d}   {len(sell_res)/total*100:5.1f}%  {da_sell:5.1f}% ({sum(1 for r in sell_res if r['correct'])}/{len(sell_res)})     {_avg(sell_res):+.2f}%")
    print(f"  HOLD     {len(hold_res):5d}   {len(hold_res)/total*100:5.1f}%  {da_hold:5.1f}% ({sum(1 for r in hold_res if r['correct'])}/{len(hold_res)})   {_avg(hold_res):+.2f}%")
    print(f"  -------  -----   ------  -------------------  ----------")
    print(f"  ALL      {total:5d}   100.0%  {da_all:5.1f}% overall")
    print(f"{'='*62}")
    print(f"  Coverage (BUY+SELL): {coverage:.1f}%")
    print(f"  Note: HOLD correct = actual in [-{buy_thr}%, +{buy_thr}%]")

    # Compare vs BTC
    print(f"\n  ── BTC reference (same config) ──────────────────────────")
    print(f"  BUY DA 59.7%  SELL DA 57.5%  Coverage 58.2%  (New Bayesian ±2%)")
    delta_buy  = da_buy  - 59.7
    delta_sell = da_sell - 57.5
    delta_cov  = coverage - 58.2
    print(f"  ETH vs BTC:  BUY DA {delta_buy:+.1f}pp  SELL DA {delta_sell:+.1f}pp  Coverage {delta_cov:+.1f}pp")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buy-thr",  type=float, default=2.0)
    ap.add_argument("--sell-thr", type=float, default=2.0)
    ap.add_argument("--k",        type=int,   default=5)
    ap.add_argument("--horizon",  default="7d",
                    choices=["1d", "3d", "7d", "15d", "30d"])
    args = ap.parse_args()

    return_weights = {"1d": 0.40, "3d": 0.30, "7d": 0.15, "15d": 0.10, "30d": 0.05}
    sw1, sw2, sw3 = _load_search_weights()
    print(f"Search weights: ({sw1:.4f}/{sw2:.4f}/{sw3:.4f})")

    print("Loading BTC proxy data from PostgreSQL...", flush=True)
    btc_proxy = await load_btc_proxy()
    print(f"  BTC records: {len(btc_proxy)} dates")

    print("Fetching ETH OHLCV from Binance...", flush=True)
    candles = await fetch_eth_ohlcv("2022-01-01")
    print(f"  ETH candles: {len(candles)} ({candles[0]['date']} → {candles[-1]['date']})")

    print("Building ETH records (hybrid: ETH price/indicators + BTC factors)...", flush=True)
    records = build_eth_records(candles, btc_proxy)
    borrowed = sum(1 for r in records if r.borrowed_factor)
    print(f"  {borrowed}/{len(records)} dates have BTC factor proxy")

    evaluate(records, args.k, return_weights,
             args.buy_thr, args.sell_thr, args.horizon,
             sw1, sw2, sw3)


if __name__ == "__main__":
    asyncio.run(main())
