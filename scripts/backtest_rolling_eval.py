"""Rolling-window evaluation on stored backtest_results.

Uses existing rows in Supabase (no new /run calls) and compares:
1) Baseline signal in table.
2) Policy-adjusted signal:
   - confidence gate for BUY/SELL -> HOLD when confidence is too low
   - HOLD release using top_similar_ret7d when directional bias is clear
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip():
            os.environ[k.strip()] = v.strip().strip("\"'")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""


@dataclass
class Row:
    backtest_date: str
    signal: str
    confidence: float
    top_similar_ret7d: float | None
    actual_return_1d: float | None
    actual_return_7d: float | None
    actual_return_30d: float | None


@dataclass
class WindowStats:
    d1: float
    d7: float
    d30: float
    coverage: float
    directional_precision_7d: float


def supabase_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _check_correct(signal: str, actual_return: float | None, hold_threshold: float = 1.0) -> bool | None:
    if actual_return is None:
        return None
    if signal == "BUY":
        return actual_return > 0
    if signal == "SELL":
        return actual_return < 0
    if signal == "HOLD":
        return abs(actual_return) < hold_threshold
    return None


def _policy_signal(
    row: Row,
    min_directional_conf: float,
    hold_release_ret7d: float,
    hold_release_conf: float,
) -> str:
    signal = row.signal.upper()
    conf = row.confidence
    sim7 = row.top_similar_ret7d

    if signal in {"BUY", "SELL"} and conf < min_directional_conf:
        return "HOLD"
    if signal == "HOLD" and sim7 is not None and conf >= hold_release_conf and abs(sim7) >= hold_release_ret7d:
        return "BUY" if sim7 > 0 else "SELL"
    return signal


def _eval_window(rows: list[Row], signal_fn) -> WindowStats:
    c1 = n1 = 0
    c7 = n7 = 0
    c30 = n30 = 0
    directional_total_7d = 0
    directional_ok_7d = 0
    directional_count = 0

    for row in rows:
        signal = signal_fn(row)
        if signal in {"BUY", "SELL"}:
            directional_count += 1

        ok1 = _check_correct(signal, row.actual_return_1d)
        ok7 = _check_correct(signal, row.actual_return_7d)
        ok30 = _check_correct(signal, row.actual_return_30d)

        if ok1 is not None:
            n1 += 1
            if ok1:
                c1 += 1
        if ok7 is not None:
            n7 += 1
            if ok7:
                c7 += 1
        if ok30 is not None:
            n30 += 1
            if ok30:
                c30 += 1

        if signal in {"BUY", "SELL"} and row.actual_return_7d is not None:
            directional_total_7d += 1
            if (signal == "BUY" and row.actual_return_7d > 0) or (signal == "SELL" and row.actual_return_7d < 0):
                directional_ok_7d += 1

    return WindowStats(
        d1=(c1 / n1) * 100 if n1 else 0.0,
        d7=(c7 / n7) * 100 if n7 else 0.0,
        d30=(c30 / n30) * 100 if n30 else 0.0,
        coverage=(directional_count / len(rows)) * 100 if rows else 0.0,
        directional_precision_7d=(directional_ok_7d / directional_total_7d) * 100 if directional_total_7d else 0.0,
    )


def _aggregate(label: str, stats: list[WindowStats]) -> None:
    if not stats:
        print(f"{label}: no windows")
        return

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    d1s = [s.d1 for s in stats]
    d7s = [s.d7 for s in stats]
    d30s = [s.d30 for s in stats]
    covs = [s.coverage for s in stats]
    p7s = [s.directional_precision_7d for s in stats]

    print(f"\n[{label}] windows={len(stats)}")
    print(f"  D+1  mean/min/max: {_mean(d1s):.2f}% / {min(d1s):.2f}% / {max(d1s):.2f}%")
    print(f"  D+7  mean/min/max: {_mean(d7s):.2f}% / {min(d7s):.2f}% / {max(d7s):.2f}%")
    print(f"  D+30 mean/min/max: {_mean(d30s):.2f}% / {min(d30s):.2f}% / {max(d30s):.2f}%")
    print(f"  Coverage mean      : {_mean(covs):.2f}%")
    print(f"  DirPrecision7d mean: {_mean(p7s):.2f}%")


async def _fetch_rows(symbol: str, limit: int) -> list[Row]:
    params = [
        ("select", "backtest_date,signal,confidence,top_similar_ret7d,actual_return_1d,actual_return_7d,actual_return_30d"),
        ("symbol", f"eq.{symbol}"),
        ("signal", "not.is.null"),
        ("order", "backtest_date.asc"),
        ("limit", str(limit)),
    ]
    async with httpx.AsyncClient(timeout=60) as http:
        resp = await http.get(f"{SUPABASE_URL}/rest/v1/backtest_results", headers=supabase_headers(), params=params)
        resp.raise_for_status()
        raw = resp.json()
    out: list[Row] = []
    for item in raw:
        out.append(
            Row(
                backtest_date=str(item.get("backtest_date", "")),
                signal=str(item.get("signal", "HOLD")).upper(),
                confidence=_to_float(item.get("confidence")) or 0.0,
                top_similar_ret7d=_to_float(item.get("top_similar_ret7d")),
                actual_return_1d=_to_float(item.get("actual_return_1d")),
                actual_return_7d=_to_float(item.get("actual_return_7d")),
                actual_return_30d=_to_float(item.get("actual_return_30d")),
            )
        )
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling eval on backtest_results")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=60, help="Drop first N oldest rows before rolling eval")
    parser.add_argument("--window", type=int, default=120, help="Rolling window size")
    parser.add_argument("--step", type=int, default=20, help="Rolling step")
    parser.add_argument("--min-directional-conf", type=float, default=0.56)
    parser.add_argument("--hold-release-ret7d", type=float, default=2.5)
    parser.add_argument("--hold-release-conf", type=float, default=0.5)
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")

    rows = await _fetch_rows(args.symbol.upper(), args.limit)
    if len(rows) <= args.warmup:
        raise RuntimeError(f"Not enough rows: total={len(rows)}, warmup={args.warmup}")

    eval_rows = rows[args.warmup:]
    print(
        f"Loaded {len(rows)} rows for {args.symbol.upper()} "
        f"({rows[0].backtest_date} -> {rows[-1].backtest_date}); eval={len(eval_rows)}"
    )

    baseline_stats: list[WindowStats] = []
    policy_stats: list[WindowStats] = []

    for start in range(0, len(eval_rows) - args.window + 1, args.step):
        window_rows = eval_rows[start : start + args.window]
        baseline_stats.append(_eval_window(window_rows, lambda r: r.signal))
        policy_stats.append(
            _eval_window(
                window_rows,
                lambda r: _policy_signal(
                    r,
                    min_directional_conf=args.min_directional_conf,
                    hold_release_ret7d=args.hold_release_ret7d,
                    hold_release_conf=args.hold_release_conf,
                ),
            )
        )

    _aggregate("baseline", baseline_stats)
    _aggregate("policy", policy_stats)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
