from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Iterable


START_DATE = "2025-07-01"
END_DATE = "2026-05-01"


PREDICTION_FILES = {
    "naive_current_ai": "artifacts/current_context_ai_eval/naive_current_ai_test.jsonl",
    "fixed_knn_rolling_stable": "artifacts/learned_strict_test_v3/fixed_knn_rolling_stable.jsonl",
    "fixed_retriever_learned_head": "artifacts/learned_strict_test_v3/fixed_retriever_learned_head.jsonl",
    "learned_retriever_fixed_head": "artifacts/learned_strict_test_v3/learned_retriever_fixed_head.jsonl",
    "learned_finbert_rolling_stable": "artifacts/learned_strict_test_v3/learned_finbert_rolling_stable.jsonl",
}


@dataclass(frozen=True)
class RegimeFeatures:
    date: str
    future_return_7d: float
    prior_20d_return: float
    prior_20d_volatility: float
    rsi: float
    macd_hist: float
    price_change_pct: float


@dataclass(frozen=True)
class SliceRule:
    name: str
    description: str
    predicate: Callable[[RegimeFeatures], bool]


def _load_payload(line: str) -> dict:
    raw = json.loads(line)
    return raw.get("payload", raw)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _daily_return_pct(candles: list[dict]) -> list[float]:
    returns: list[float] = []
    for prev, cur in zip(candles, candles[1:]):
        prev_close = prev.get("close")
        cur_close = cur.get("close")
        if _is_number(prev_close) and _is_number(cur_close) and float(prev_close) != 0.0:
            returns.append((float(cur_close) / float(prev_close) - 1.0) * 100.0)
    return returns


def load_regime_features(dataset: Path, *, start_date: str, end_date: str) -> dict[str, RegimeFeatures]:
    rows: dict[str, RegimeFeatures] = {}
    with dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = _load_payload(line)
            date = str(payload.get("date") or payload.get("record_date"))
            if date < start_date or date > end_date:
                continue

            market = payload.get("market_snapshot") or {}
            indicators = market.get("indicators") or {}
            candles = market.get("candles") or market.get("recent_candles") or []
            if len(candles) < 2:
                continue
            first_close = candles[0].get("close")
            last_close = candles[-1].get("close")
            if not (_is_number(first_close) and _is_number(last_close) and float(first_close) != 0.0):
                continue

            prior_20d_return = (float(last_close) / float(first_close) - 1.0) * 100.0
            daily_returns = _daily_return_pct(candles)
            prior_20d_volatility = pstdev(daily_returns) if len(daily_returns) >= 2 else 0.0

            future_return_7d = payload.get("future_return_7d")
            rsi = indicators.get("rsi", market.get("rsi"))
            macd_hist = indicators.get("macd_hist", market.get("macd_hist"))
            price_change_pct = indicators.get("price_change_pct", market.get("price_change_pct"))
            values = (future_return_7d, rsi, macd_hist, price_change_pct)
            if not all(_is_number(value) for value in values):
                continue

            rows[date] = RegimeFeatures(
                date=date,
                future_return_7d=float(future_return_7d),
                prior_20d_return=prior_20d_return,
                prior_20d_volatility=prior_20d_volatility,
                rsi=float(rsi),
                macd_hist=float(macd_hist),
                price_change_pct=float(price_change_pct),
            )
    return rows


def _quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty values")
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def build_slice_rules(features: dict[str, RegimeFeatures]) -> list[SliceRule]:
    vol_values = [row.prior_20d_volatility for row in features.values()]
    vol_q25 = _quantile(vol_values, 0.25)
    vol_q75 = _quantile(vol_values, 0.75)

    return [
        SliceRule(
            "official_all",
            "Full chronological held-out test window.",
            lambda row: True,
        ),
        SliceRule(
            "deep_downtrend_ret20_le_neg15",
            "Prior 20-day return <= -15%. Small favorable drawdown slice.",
            lambda row: row.prior_20d_return <= -15.0,
        ),
        SliceRule(
            "downtrend_ret20_le_neg5",
            "Prior 20-day return <= -5%. Broader drawdown regime.",
            lambda row: row.prior_20d_return <= -5.0,
        ),
        SliceRule(
            "sideways_abs_ret20_le_3",
            "Absolute prior 20-day return <= 3%. Sideways regime.",
            lambda row: abs(row.prior_20d_return) <= 3.0,
        ),
        SliceRule(
            "uptrend_ret20_ge_5",
            "Prior 20-day return >= +5%. Positive momentum regime.",
            lambda row: row.prior_20d_return >= 5.0,
        ),
        SliceRule(
            "strong_uptrend_ret20_ge_10",
            "Prior 20-day return >= +10%. Strong positive momentum regime.",
            lambda row: row.prior_20d_return >= 10.0,
        ),
        SliceRule(
            "high_volatility_top_quartile",
            f"Prior 20-day daily-return volatility >= test-set Q75 ({vol_q75:.2f}%).",
            lambda row: row.prior_20d_volatility >= vol_q75,
        ),
        SliceRule(
            "low_volatility_bottom_quartile",
            f"Prior 20-day daily-return volatility <= test-set Q25 ({vol_q25:.2f}%).",
            lambda row: row.prior_20d_volatility <= vol_q25,
        ),
        SliceRule(
            "oversold_rsi_le_40",
            "Current RSI <= 40. Technical oversold representative.",
            lambda row: row.rsi <= 40.0,
        ),
        SliceRule(
            "overbought_rsi_ge_65",
            "Current RSI >= 65. Technical overbought representative.",
            lambda row: row.rsi >= 65.0,
        ),
    ]


def load_predictions(path: Path, *, start_date: str, end_date: str) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            date = str(row["date"])
            if start_date <= date <= end_date:
                rows.append(row)
    return rows


def _active_correct(row: dict) -> bool | None:
    signal = row["predicted_signal"]
    actual_return = float(row["actual_return_7d"])
    if signal == "BUY":
        return actual_return > 0.0
    if signal == "SELL":
        return actual_return < 0.0
    return None


def summarize_prediction_rows(rows: list[dict], dates: set[str]) -> dict | None:
    selected = [row for row in rows if row["date"] in dates]
    if not selected:
        return None
    n = len(selected)
    active_rows = [row for row in selected if row["predicted_signal"] in {"BUY", "SELL"}]
    active_n = len(active_rows)
    active_correct = sum(bool(_active_correct(row)) for row in active_rows)
    return {
        "n": n,
        "overall_acc": sum(row["predicted_signal"] == row["actual_signal"] for row in selected) / n,
        "active_acc": active_correct / active_n if active_n else 0.0,
        "coverage": active_n / n,
        "hit_at_5_same_sign": sum(bool(row.get("top5_same_sign")) for row in selected) / n,
        "buy_rate": sum(row["predicted_signal"] == "BUY" for row in selected) / n,
        "hold_rate": sum(row["predicted_signal"] == "HOLD" for row in selected) / n,
        "sell_rate": sum(row["predicted_signal"] == "SELL" for row in selected) / n,
    }


def summarize_actuals(features: dict[str, RegimeFeatures], dates: set[str]) -> dict:
    selected = [features[date] for date in sorted(dates)]
    n = len(selected)
    actual_buy = sum(row.future_return_7d > 2.0 for row in selected)
    actual_sell = sum(row.future_return_7d < -2.0 for row in selected)
    actual_hold = n - actual_buy - actual_sell
    return {
        "n_dates": n,
        "first_date": selected[0].date if selected else None,
        "last_date": selected[-1].date if selected else None,
        "mean_future_return_7d": mean(row.future_return_7d for row in selected) if selected else 0.0,
        "mean_prior_20d_return": mean(row.prior_20d_return for row in selected) if selected else 0.0,
        "mean_prior_20d_volatility": mean(row.prior_20d_volatility for row in selected) if selected else 0.0,
        "actual_buy_rate": actual_buy / n if n else 0.0,
        "actual_hold_rate": actual_hold / n if n else 0.0,
        "actual_sell_rate": actual_sell / n if n else 0.0,
    }


def _fmt(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Exploratory Regime Slice Report",
        "",
        "This is an exploratory analysis, not the primary paper result. The primary",
        "result remains the full chronological held-out test. These slices use only",
        "current or prior market features available at prediction time, then measure",
        "how existing per-row prediction artifacts behave inside each regime.",
        "",
        f"- Dataset: `{result['dataset']}`",
        f"- Test window: `{result['start_date']}` to `{result['end_date']}`",
        f"- Minimum slice size requested: `{result['min_slice_size']}` rows",
        "",
        "## Model Coverage",
        "",
        "| Model | Available prediction rows |",
        "| --- | ---: |",
    ]
    for model, count in result["model_prediction_counts"].items():
        lines.append(f"| `{model}` | {count} |")

    lines.extend(
        [
            "",
            "Note: per-row naive LLM artifacts may cover fewer dates than the structured",
            "StockMem artifacts. Use the official summary table for primary full-test",
            "claims.",
            "",
            "## Slice Summary",
            "",
        ]
    )

    for item in result["slices"]:
        if item["n_dates"] < result["min_slice_size"]:
            continue
        lines.extend(
            [
                f"### {item['name']}",
                "",
                item["description"],
                "",
                f"- dates: `{item['first_date']}` to `{item['last_date']}`",
                f"- n_dates: `{item['n_dates']}`",
                f"- mean prior 20d return: `{item['mean_prior_20d_return']:+.2f}%`",
                f"- mean prior 20d volatility: `{item['mean_prior_20d_volatility']:.2f}%`",
                f"- mean future 7d return: `{item['mean_future_return_7d']:+.2f}%`",
                f"- actual distribution: BUY `{item['actual_buy_rate']:.3f}`, HOLD `{item['actual_hold_rate']:.3f}`, SELL `{item['actual_sell_rate']:.3f}`",
                "",
                "| Model | n | Overall | Active | Coverage | Hit@5 | BUY% | HOLD% | SELL% |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_name, metrics in item["models"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{model_name}`",
                        str(metrics["n"]),
                        f"{metrics['overall_acc']:.4f}",
                        f"{metrics['active_acc']:.4f}",
                        f"{metrics['coverage']:.4f}",
                        f"{metrics['hit_at_5_same_sign']:.4f}",
                        f"{metrics['buy_rate']:.3f}",
                        f"{metrics['hold_rate']:.3f}",
                        f"{metrics['sell_rate']:.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The broad downtrend slice is the cleanest favorable representative because",
            "  it has a larger sample than the deep-drawdown slice.",
            "- Strong uptrend and sideways representatives are weaker for the current",
            "  fixed kNN plus learned-head pipeline.",
            "- This should be written as regime-stratified error analysis, not as a",
            "  replacement for the official chronological test.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(result: dict, path: Path) -> None:
    fieldnames = [
        "slice",
        "description",
        "n_dates",
        "first_date",
        "last_date",
        "model",
        "n",
        "overall_acc",
        "active_acc",
        "coverage",
        "hit_at_5_same_sign",
        "buy_rate",
        "hold_rate",
        "sell_rate",
        "mean_future_return_7d",
        "mean_prior_20d_return",
        "mean_prior_20d_volatility",
        "actual_buy_rate",
        "actual_hold_rate",
        "actual_sell_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in result["slices"]:
            for model_name, metrics in item["models"].items():
                row = {
                    "slice": item["name"],
                    "description": item["description"],
                    "n_dates": item["n_dates"],
                    "first_date": item["first_date"],
                    "last_date": item["last_date"],
                    "model": model_name,
                    "mean_future_return_7d": _fmt(item["mean_future_return_7d"]),
                    "mean_prior_20d_return": _fmt(item["mean_prior_20d_return"]),
                    "mean_prior_20d_volatility": _fmt(item["mean_prior_20d_volatility"]),
                    "actual_buy_rate": _fmt(item["actual_buy_rate"]),
                    "actual_hold_rate": _fmt(item["actual_hold_rate"]),
                    "actual_sell_rate": _fmt(item["actual_sell_rate"]),
                }
                row.update({key: _fmt(value) for key, value in metrics.items()})
                writer.writerow(row)


def analyze(
    *,
    dataset: Path,
    out_dir: Path,
    start_date: str,
    end_date: str,
    min_slice_size: int,
) -> dict:
    features = load_regime_features(dataset, start_date=start_date, end_date=end_date)
    if not features:
        raise SystemExit("no eligible dataset rows found")

    predictions = {
        model_name: load_predictions(Path(path), start_date=start_date, end_date=end_date)
        for model_name, path in PREDICTION_FILES.items()
    }
    rules = build_slice_rules(features)

    slices: list[dict] = []
    for rule in rules:
        dates = {date for date, row in features.items() if rule.predicate(row)}
        if not dates:
            continue
        item = {
            "name": rule.name,
            "description": rule.description,
            **summarize_actuals(features, dates),
            "models": {},
        }
        for model_name, rows in predictions.items():
            metrics = summarize_prediction_rows(rows, dates)
            if metrics is not None:
                item["models"][model_name] = metrics
        slices.append(item)

    result = {
        "dataset": str(dataset),
        "start_date": start_date,
        "end_date": end_date,
        "min_slice_size": min_slice_size,
        "model_prediction_counts": {name: len(rows) for name, rows in predictions.items()},
        "slices": slices,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(result, out_dir / "summary.md")
    write_csv(result, out_dir / "summary.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exploratory StockMem regime slices")
    parser.add_argument("--dataset", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--out-dir", default="artifacts/exploratory_regime_slices")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--min-slice-size", type=int, default=20)
    args = parser.parse_args()

    result = analyze(
        dataset=Path(args.dataset),
        out_dir=Path(args.out_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        min_slice_size=args.min_slice_size,
    )
    print(f"wrote {len(result['slices'])} slices to {args.out_dir}")


if __name__ == "__main__":
    main()
