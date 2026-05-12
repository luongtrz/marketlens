"""Walk-forward backtest of stockmem via the HTTP API.

Loads mock_3y_records.json, pushes records chronologically to the live service,
and evaluates similarity-search prediction quality at each step after warmup.

Metrics reported:
  DA    = Directional Accuracy — did majority of similar cases predict the right sign?
  WADA  = Weighted-Avg DA — similarity-weighted vote accuracy
  MRet  = mean absolute return of retrieved neighbors
  MSim  = mean similarity score of retrieved neighbors

Usage:
    python stockmem/scripts/backtest_api.py
    python stockmem/scripts/backtest_api.py --url http://localhost:8003 --k 5 --warmup 100 --horizon 7d
    python stockmem/scripts/backtest_api.py --skip-load   # skip loading if data already in DB
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
MOCK_DATA = ROOT / "stockmem" / "data" / "mock_3y_records.json"

HORIZONS = {"1d": "future_return_1d", "7d": "future_return_7d", "30d": "future_return_30d"}


@dataclass
class EvalRow:
    date: str
    actual_return: float
    predicted_sign: float       # weighted-vote sign of similar cases
    mean_similarity: float
    n_neighbors: int
    correct: bool               # directional accuracy
    wa_correct: bool            # weighted-average directional accuracy


@dataclass
class Stats:
    rows: list[EvalRow] = field(default_factory=list)

    def add(self, row: EvalRow) -> None:
        self.rows.append(row)

    def report(self) -> dict:
        if not self.rows:
            return {}
        da = sum(r.correct for r in self.rows) / len(self.rows)
        wada = sum(r.wa_correct for r in self.rows) / len(self.rows)
        msim = sum(r.mean_similarity for r in self.rows) / len(self.rows)
        mabs = sum(abs(r.actual_return) for r in self.rows) / len(self.rows)
        return {
            "n_eval": len(self.rows),
            "DA": round(da, 4),
            "WADA": round(wada, 4),
            "mean_similarity": round(msim, 4),
            "mean_abs_return_pct": round(mabs, 4),
        }


def load_records() -> list[dict]:
    with open(MOCK_DATA) as f:
        records = json.load(f)
    records.sort(key=lambda r: r["date"])
    return records


def push_record(client: httpx.Client, base_url: str, record: dict) -> Optional[str]:
    resp = client.post(f"{base_url}/record", json={"record": record}, timeout=10)
    if resp.status_code == 200:
        return resp.json()["id"]
    return None


def search(client: httpx.Client, base_url: str, record: dict, k: int, before_date: str) -> list[dict]:
    payload = {"query": record, "k": k, "before_date": before_date}
    resp = client.post(f"{base_url}/search", json=payload, timeout=10)
    if resp.status_code == 200:
        return resp.json()["results"]
    return []


def evaluate(results: list[dict], actual_return: float, horizon_key: str) -> Optional[EvalRow]:
    if not results:
        return None

    neighbor_returns = []
    similarities = []
    for r in results:
        ret = r["record"].get(horizon_key)
        if ret is None:
            continue
        neighbor_returns.append(ret)
        similarities.append(r["similarity"])

    if not neighbor_returns:
        return None

    # Simple majority vote
    votes = [1 if ret > 0 else -1 for ret in neighbor_returns]
    predicted_sign = sum(votes) / len(votes)
    correct = (predicted_sign >= 0) == (actual_return >= 0)

    # Weighted vote
    total_sim = sum(similarities)
    if total_sim > 0:
        wa_sign = sum(s * (1 if r > 0 else -1) for s, r in zip(similarities, neighbor_returns)) / total_sim
    else:
        wa_sign = predicted_sign
    wa_correct = (wa_sign >= 0) == (actual_return >= 0)

    return EvalRow(
        date="",
        actual_return=actual_return,
        predicted_sign=predicted_sign,
        mean_similarity=sum(similarities) / len(similarities),
        n_neighbors=len(neighbor_returns),
        correct=correct,
        wa_correct=wa_correct,
    )


def run(url: str, k: int, warmup: int, horizon: str, skip_load: bool) -> None:
    horizon_key = HORIZONS[horizon]
    records = load_records()
    print(f"Loaded {len(records)} records ({records[0]['date']} → {records[-1]['date']})")
    print(f"Config: k={k}, warmup={warmup}, horizon={horizon}, url={url}")

    with httpx.Client() as client:
        # Check health
        try:
            resp = client.get(f"{url}/health", timeout=5)
            h = resp.json()
            print(f"Service: {h['status']} | backend={h['vector_backend']}\n")
        except Exception as e:
            print(f"ERROR: Cannot reach {url} — {e}")
            sys.exit(1)

        stats = Stats()
        loaded = 0
        searched = 0
        errors = 0
        t0 = time.time()

        for i, rec in enumerate(records):
            if i >= warmup:
                # Search BEFORE storing the current record (walk-forward guard)
                results = search(client, url, rec, k=k, before_date=rec["date"])
                searched += 1

                actual = rec.get(horizon_key)
                if actual is not None:
                    row = evaluate(results, actual, horizon_key)
                    if row:
                        row.date = rec["date"]
                        stats.add(row)

            if not skip_load:
                rid = push_record(client, url, rec)
                if rid:
                    loaded += 1
                else:
                    errors += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:4d}/{len(records)}] loaded={loaded} searched={searched} "
                      f"errors={errors} ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s — loaded={loaded}, searched={searched}, errors={errors}")
        print()

        metrics = stats.report()
        if metrics:
            print("=== Backtest Results ===")
            print(f"  Eval samples  : {metrics['n_eval']}")
            print(f"  Horizon       : {horizon}")
            print(f"  DA (majority) : {metrics['DA']*100:.1f}%")
            print(f"  WADA (w-avg)  : {metrics['WADA']*100:.1f}%")
            print(f"  Mean sim score: {metrics['mean_similarity']:.4f}")
            print(f"  Mean |return|  : {metrics['mean_abs_return_pct']:.2f}%")
        else:
            print("No eval data collected — check that mock records have future returns.")

        # Per-decile breakdown by similarity
        if stats.rows:
            rows_sorted = sorted(stats.rows, key=lambda r: r.mean_similarity)
            n = len(rows_sorted)
            q1 = rows_sorted[:n//3]
            q2 = rows_sorted[n//3:2*n//3]
            q3 = rows_sorted[2*n//3:]
            print("\n=== DA by Similarity Tertile ===")
            for label, bucket in [("Low sim", q1), ("Mid sim", q2), ("High sim", q3)]:
                if bucket:
                    da = sum(r.correct for r in bucket) / len(bucket)
                    msim = sum(r.mean_similarity for r in bucket) / len(bucket)
                    print(f"  {label} (avg={msim:.3f}): DA={da*100:.1f}%  n={len(bucket)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of stockmem via HTTP API")
    parser.add_argument("--url", default="http://localhost:8003", help="StockMem base URL")
    parser.add_argument("--k", type=int, default=5, help="Neighbors to retrieve per query")
    parser.add_argument("--warmup", type=int, default=60, help="Records to load before evaluating")
    parser.add_argument("--horizon", choices=list(HORIZONS), default="7d", help="Return horizon")
    parser.add_argument("--skip-load", action="store_true", help="Skip loading records (data already in DB)")
    args = parser.parse_args()
    run(args.url, args.k, args.warmup, args.horizon, args.skip_load)


if __name__ == "__main__":
    main()
