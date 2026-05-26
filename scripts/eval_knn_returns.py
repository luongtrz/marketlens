"""Offline evaluation of kNN-returns signal accuracy.

Loads all stockmem_records from PostgreSQL, computes cosine similarity
between each date's embedding and all prior records, picks top-k, applies
the kNN-returns weighted average, then measures Directional Accuracy (DA)
against actual future returns.

Usage:
    python scripts/eval_knn_returns.py
    python scripts/eval_knn_returns.py --k 5 --w1d 0.40 --w3d 0.30 --w7d 0.15 --w15d 0.10 --w30d 0.05
    python scripts/eval_knn_returns.py --buy-thr 3 --sell-thr 3 --horizon 7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass

import asyncpg
import numpy as np


DB_URL = "postgresql://postgres:pass@localhost:5432/postgres"


@dataclass
class Record:
    date: str
    factor_vec: np.ndarray       # 75d (stored)
    indicator_vec: np.ndarray    # 5d  (computed)
    price_vec: np.ndarray        # 60d (computed)
    joint: np.ndarray            # 140d combined
    future: dict[str, float | None]  # 1d/3d/7d/15d/30d


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _compute_indicator_vec(ms: dict, sentiment: float) -> np.ndarray:
    msi = float(ms.get("msi") or 0.0)
    rsi = float(ms.get("rsi") or 50.0)
    fgi = float(ms.get("fear_greed_index") or 50.0)
    pcp = float(ms.get("price_change_pct") or 0.0)
    raw = np.array([msi, rsi, sentiment, fgi, pcp], dtype=np.float32)
    mean = raw.mean()
    std = raw.std() + 1e-8
    return _l2((raw - mean) / std)


def _compute_price_vec(candles: list[dict]) -> np.ndarray:
    if len(candles) < 2:
        return np.zeros(60, dtype=np.float32)
    closes  = np.array([float(c.get("close", 0)) for c in candles], dtype=np.float32)
    highs   = np.array([float(c.get("high",  0)) for c in candles], dtype=np.float32)
    lows    = np.array([float(c.get("low",   0)) for c in candles], dtype=np.float32)
    volumes = np.array([float(c.get("volume",0)) for c in candles], dtype=np.float32)

    rets    = np.diff(closes) / (closes[:-1] + 1e-8)
    ranges  = (highs - lows) / (closes + 1e-8)
    vol_chg = np.diff(volumes) / (volumes[:-1] + 1e-8)

    def _tail20(arr: np.ndarray) -> np.ndarray:
        tail = arr[-20:]
        if len(tail) < 20:
            tail = np.pad(tail, (20 - len(tail), 0))
        return tail.astype(np.float32)

    price_features = np.concatenate([_tail20(rets), _tail20(ranges[1:]), _tail20(vol_chg)])
    return _l2(price_features)


def _build_joint(fv: np.ndarray, iv: np.ndarray, pv: np.ndarray) -> np.ndarray:
    W1, W2, W3 = 0.35, 0.20, 0.45
    return np.concatenate([fv, iv * W2, pv * W3]).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _weighted_avg(record: Record, weights: dict[str, float]) -> float | None:
    total_w = total_v = 0.0
    for h, w in weights.items():
        v = record.future.get(h)
        if v is not None:
            total_v += v * w
            total_w += w
    return total_v / total_w if total_w > 0 else None


def _signal(avg: float, buy_thr: float, sell_thr: float) -> str:
    if avg > buy_thr:
        return "BUY"
    if avg < -sell_thr:
        return "SELL"
    return "HOLD"


async def load_records() -> list[Record]:
    conn = await asyncpg.connect(DB_URL)
    rows = await conn.fetch(
        "SELECT record_date, payload FROM stockmem_records WHERE symbol='BTC' ORDER BY record_date"
    )
    await conn.close()

    records: list[Record] = []
    skipped = 0
    for row in rows:
        p = json.loads(row["payload"])
        ms = p.get("market_snapshot", {})

        fv_raw = p.get("factor_vector", [])
        if not fv_raw or len(fv_raw) != 75:
            skipped += 1
            continue
        fv = _l2(np.array(fv_raw, dtype=np.float32))

        candles = ms.get("recent_candles") or ms.get("candles") or []
        iv = _compute_indicator_vec(ms, float(p.get("sentiment_score", 0.0)))
        pv = _compute_price_vec(candles)
        joint = _build_joint(fv, iv, pv)

        records.append(Record(
            date=str(row["record_date"]),
            factor_vec=fv,
            indicator_vec=iv,
            price_vec=pv,
            joint=joint,
            future={
                "1d":  p.get("future_return_1d"),
                "3d":  p.get("future_return_3d"),
                "7d":  p.get("future_return_7d"),
                "15d": p.get("future_return_15d"),
                "30d": p.get("future_return_30d"),
            },
        ))

    if skipped:
        print(f"  Skipped {skipped} records with missing factor_vector", flush=True)
    return records


def evaluate(
    records: list[Record],
    k: int,
    weights: dict[str, float],
    buy_thr: float,
    sell_thr: float,
    eval_horizon: str,
) -> None:
    n = len(records)
    # Use component-wise weighted similarity like the real searcher
    # score = 0.35*sim(factor) + 0.20*sim(indicator) + 0.45*sim(price)
    W_F, W_I, W_P = 0.35, 0.20, 0.45

    total = skipped_no_neighbors = skipped_no_return = 0
    results: list[dict] = []

    print(f"\nEvaluating {n} dates with k={k}, weights={weights}, thr=±{buy_thr}%/{sell_thr}%", flush=True)

    for i, qry in enumerate(records):
        # All prior records only (before_date filter)
        prior = records[:i]
        if len(prior) < k:
            skipped_no_neighbors += 1
            continue

        # Compute component cosine similarities
        sims = []
        for p in prior:
            sim_f = _cosine(qry.factor_vec, p.factor_vec)
            sim_i = _cosine(qry.indicator_vec, p.indicator_vec)
            sim_p = _cosine(qry.price_vec, p.price_vec)
            score = W_F * sim_f + W_I * sim_i + W_P * sim_p
            sims.append((score, p))

        sims.sort(key=lambda x: x[0], reverse=True)
        top_k = [rec for _, rec in sims[:k]]

        # kNN-returns weighted average
        avgs = [a for r in top_k if (a := _weighted_avg(r, weights)) is not None]
        if not avgs:
            skipped_no_neighbors += 1
            continue

        overall_avg = sum(avgs) / len(avgs)
        sig = _signal(overall_avg, buy_thr, sell_thr)

        actual = qry.future.get(eval_horizon)
        if actual is None:
            skipped_no_return += 1
            continue

        # Directional accuracy
        # BUY correct if actual > 0; SELL correct if actual < 0
        # HOLD correct if actual within [-buy_thr, +buy_thr] (market truly flat)
        if sig == "BUY":
            correct = actual > 0
        elif sig == "SELL":
            correct = actual < 0
        else:  # HOLD
            correct = -buy_thr <= actual <= buy_thr

        results.append({
            "date": qry.date,
            "signal": sig,
            "avg": overall_avg,
            "actual": actual,
            "correct": correct,
        })
        total += 1

    # Stats
    buy_res  = [r for r in results if r["signal"] == "BUY"]
    sell_res = [r for r in results if r["signal"] == "SELL"]
    hold_res = [r for r in results if r["signal"] == "HOLD"]

    buy_correct  = [r for r in buy_res  if r["correct"]]
    sell_correct = [r for r in sell_res if r["correct"]]
    hold_correct = [r for r in hold_res if r["correct"]]

    da_buy  = len(buy_correct)  / len(buy_res)  * 100 if buy_res  else 0
    da_sell = len(sell_correct) / len(sell_res) * 100 if sell_res else 0
    da_hold = len(hold_correct) / len(hold_res) * 100 if hold_res else 0
    da_all  = (len(buy_correct) + len(sell_correct) + len(hold_correct)) / total * 100 if total else 0
    coverage = (len(buy_res) + len(sell_res)) / total * 100 if total else 0

    avg_ret_buy  = sum(r["actual"] for r in buy_res)  / len(buy_res)  if buy_res  else 0
    avg_ret_sell = sum(r["actual"] for r in sell_res) / len(sell_res) if sell_res else 0
    avg_ret_hold = sum(r["actual"] for r in hold_res) / len(hold_res) if hold_res else 0

    print(f"\n{'='*62}")
    print(f"  Eval horizon: D+{eval_horizon}  |  Dates evaluated: {total}")
    print(f"  Skipped (too early / no neighbors): {skipped_no_neighbors}")
    print(f"  Skipped (no actual return):         {skipped_no_return}")
    print(f"{'='*62}")
    print(f"  Signal   Count    Share   DA                   Avg actual")
    print(f"  -------  -----   ------  -------------------  ----------")
    print(f"  BUY      {len(buy_res):5d}   {len(buy_res)/total*100:5.1f}%  {da_buy:5.1f}% ({len(buy_correct)}/{len(buy_res)})     {avg_ret_buy:+.2f}%")
    print(f"  SELL     {len(sell_res):5d}   {len(sell_res)/total*100:5.1f}%  {da_sell:5.1f}% ({len(sell_correct)}/{len(sell_res)})     {avg_ret_sell:+.2f}%")
    print(f"  HOLD     {len(hold_res):5d}   {len(hold_res)/total*100:5.1f}%  {da_hold:5.1f}% ({len(hold_correct)}/{len(hold_res)})   {avg_ret_hold:+.2f}%")
    print(f"  -------  -----   ------  -------------------  ----------")
    print(f"  ALL      {total:5d}   100.0%  {da_all:5.1f}% overall")
    print(f"{'='*62}")
    print(f"  Coverage (BUY+SELL): {coverage:.1f}%")
    print(f"  Note: HOLD correct = actual in [-{buy_thr}%, +{buy_thr}%]")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w1d",  type=float, default=0.40)
    ap.add_argument("--w3d",  type=float, default=0.30)
    ap.add_argument("--w7d",  type=float, default=0.15)
    ap.add_argument("--w15d", type=float, default=0.10)
    ap.add_argument("--w30d", type=float, default=0.05)
    ap.add_argument("--buy-thr",  type=float, default=3.0, help="BUY threshold %%")
    ap.add_argument("--sell-thr", type=float, default=3.0, help="SELL threshold %% (symmetric)")
    ap.add_argument("--horizon",  default="7d", choices=["1d","3d","7d","15d","30d"])
    args = ap.parse_args()

    weights = {
        "1d": args.w1d, "3d": args.w3d, "7d": args.w7d,
        "15d": args.w15d, "30d": args.w30d,
    }
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 0.01:
        print(f"Warning: weights sum to {total_w:.2f}, not 1.0")

    print("Loading records from PostgreSQL...", flush=True)
    records = await load_records()
    print(f"Loaded {len(records)} records ({records[0].date} → {records[-1].date})")

    evaluate(records, args.k, weights, args.buy_thr, args.sell_thr, args.horizon)


if __name__ == "__main__":
    asyncio.run(main())
