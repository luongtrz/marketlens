from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from stockmem.src.search.learned_metric import LearnedDiagonalMetric


SEARCH_WEIGHTS_DEFAULT = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)
PRODUCTION_RETURN_WEIGHTS = {"1d": 0.15, "3d": 0.25, "7d": 0.35, "15d": 0.15, "30d": 0.10}
CLASSES = ("BUY", "HOLD", "SELL")
HORIZONS = ("1d", "3d", "7d", "15d", "30d")


@dataclass(frozen=True)
class Record:
    date: str
    day: date
    open: float
    close: float
    factor_vec: np.ndarray
    indicator_vec: np.ndarray
    price_vec: np.ndarray
    event_vec: np.ndarray
    future: dict[str, float | None]

    @property
    def blocks(self) -> tuple[np.ndarray, ...]:
        return (self.event_vec, self.factor_vec, self.indicator_vec, self.price_vec)


@dataclass(frozen=True)
class HeadConfig:
    name: str
    retriever: str
    k: int
    buy_thr: float
    sell_thr: float
    return_weights: dict[str, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    fee_bps_side: float
    slippage_bps_side: float

    @property
    def side_cost_pct(self) -> float:
        return (self.fee_bps_side + self.slippage_bps_side) / 100.0

    @property
    def roundtrip_cost_pct(self) -> float:
        return 2.0 * self.side_cost_pct


def _l2(arr: np.ndarray) -> np.ndarray:
    arr64 = np.asarray(arr, dtype=np.float64)
    norm = float(np.linalg.norm(arr64))
    if norm <= 1e-12:
        return np.zeros_like(arr64, dtype=np.float32)
    return (arr64 / norm).astype(np.float32)


def _load_ndjson(path: Path, *, start_date: str) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            payload = raw["payload"]
            day_str = str(raw.get("record_date") or payload.get("date"))
            if day_str < start_date:
                continue
            factor_raw = payload.get("factor_vec") or payload.get("factor_vector") or []
            indicator_raw = payload.get("indicator_vec") or []
            price_raw = payload.get("price_vec") or []
            event_raw = payload.get("event_vec") or []
            if len(factor_raw) != 75 or len(indicator_raw) != 5 or len(price_raw) != 60:
                continue
            if len(event_raw) != 85:
                event_raw = [0.0] * 85
            ohlcv = payload.get("market_snapshot", {}).get("ohlcv") or {}
            if "open" not in ohlcv or "close" not in ohlcv:
                continue
            records.append(
                Record(
                    date=day_str,
                    day=date.fromisoformat(day_str),
                    open=float(ohlcv["open"]),
                    close=float(ohlcv["close"]),
                    factor_vec=_l2(np.array(factor_raw, dtype=np.float32)),
                    indicator_vec=_l2(np.array(indicator_raw, dtype=np.float32)),
                    price_vec=_l2(np.array(price_raw, dtype=np.float32)),
                    event_vec=_l2(np.array(event_raw, dtype=np.float32)),
                    future={
                        "1d": payload.get("future_return_1d"),
                        "3d": payload.get("future_return_3d"),
                        "7d": payload.get("future_return_7d"),
                        "15d": payload.get("future_return_15d"),
                        "30d": payload.get("future_return_30d"),
                    },
                )
            )
    return records


def _load_search_weights(path: Path) -> tuple[float, float, float]:
    if not path.exists():
        return SEARCH_WEIGHTS_DEFAULT
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("weights", payload)
    return (
        float(source["w1_factor"]),
        float(source["w2_indicator"]),
        float(source["w3_price"]),
    )


def _load_head_config(path: Path) -> HeadConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    head = payload.get("head", {})
    return HeadConfig(
        name=str(payload.get("name", path.stem)),
        retriever=str(payload.get("retriever", "fixed_knn")),
        k=int(head["k"]),
        buy_thr=float(head["buy_threshold"]),
        sell_thr=float(head["sell_threshold"]),
        return_weights={k: float(v) for k, v in head["return_weights"].items()},
    )


def _fixed_score(q: Record, c: Record, weights: tuple[float, float, float]) -> float:
    return (
        weights[0] * float(np.dot(q.factor_vec, c.factor_vec))
        + weights[1] * float(np.dot(q.indicator_vec, c.indicator_vec))
        + weights[2] * float(np.dot(q.price_vec, c.price_vec))
    )


def _weighted_avg(record: Record, weights: dict[str, float]) -> float | None:
    total_w = 0.0
    total_v = 0.0
    for horizon, weight in weights.items():
        value = record.future.get(horizon)
        if value is None:
            continue
        total_v += float(value) * weight
        total_w += weight
    if total_w <= 1e-12:
        return None
    return total_v / total_w


def _signal(avg: float, buy_thr: float, sell_thr: float) -> str:
    if avg > buy_thr:
        return "BUY"
    if avg < -sell_thr:
        return "SELL"
    return "HOLD"


def _actual_class(actual_ret_7d: float, buy_thr: float, sell_thr: float) -> str:
    if actual_ret_7d > buy_thr:
        return "BUY"
    if actual_ret_7d < -sell_thr:
        return "SELL"
    return "HOLD"


def _annualized_sharpe(trade_returns_pct: list[float], trades_per_year: float) -> float:
    if not trade_returns_pct:
        return 0.0
    arr = np.asarray(trade_returns_pct, dtype=np.float64) / 100.0
    std = arr.std()
    if std <= 1e-12:
        return 0.0
    return float(arr.mean() / std * np.sqrt(max(trades_per_year, 1e-12)))


def _annualized_sortino(trade_returns_pct: list[float], trades_per_year: float) -> float:
    if not trade_returns_pct:
        return 0.0
    arr = np.asarray(trade_returns_pct, dtype=np.float64) / 100.0
    downside = arr[arr < 0]
    if downside.size == 0:
        return float("inf")
    dd = downside.std()
    if dd <= 1e-12:
        return 0.0
    return float(arr.mean() / dd * np.sqrt(max(trades_per_year, 1e-12)))


def _max_drawdown(trade_returns_pct: list[float]) -> float:
    if not trade_returns_pct:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(trade_returns_pct, dtype=np.float64) / 100.0)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(running_max, 1e-12) - 1.0
    return float(drawdowns.min()) * 100.0


def _profit_factor(trade_returns_pct: list[float]) -> float:
    gains = sum(v for v in trade_returns_pct if v > 0)
    losses = -sum(v for v in trade_returns_pct if v < 0)
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _top_evidence(scored_neighbors: list[tuple[float, Record]], *, limit: int = 5) -> list[dict[str, float | str | None]]:
    evidence: list[dict[str, float | str | None]] = []
    for score, rec in scored_neighbors[:limit]:
        evidence.append(
            {
                "date": rec.date,
                "score": round(float(score), 6),
                "future_return_7d": None if rec.future.get("7d") is None else round(float(rec.future["7d"]), 4),
                "future_return_30d": None if rec.future.get("30d") is None else round(float(rec.future["30d"]), 4),
            }
        )
    return evidence


def _classification_summary(rows: list[dict[str, object]], buy_thr: float, sell_thr: float) -> dict[str, float | int]:
    buy = sell = hold = buy_correct = sell_correct = hold_correct = 0
    for row in rows:
        pred = str(row["signal"])
        actual_ret = float(row["actual_return_7d"])
        if pred == "BUY":
            buy += 1
            buy_correct += int(actual_ret > 0.0)
        elif pred == "SELL":
            sell += 1
            sell_correct += int(actual_ret < 0.0)
        else:
            hold += 1
            hold_correct += int(-sell_thr <= actual_ret <= buy_thr)
    n = len(rows)
    active = buy + sell
    total_correct = buy_correct + sell_correct + hold_correct
    return {
        "n": n,
        "buy": buy,
        "sell": sell,
        "hold": hold,
        "coverage_pct": (100.0 * active / n) if n else 0.0,
        "buy_da_pct": (100.0 * buy_correct / buy) if buy else 0.0,
        "sell_da_pct": (100.0 * sell_correct / sell) if sell else 0.0,
        "hold_da_pct": (100.0 * hold_correct / hold) if hold else 0.0,
        "overall_da_pct": (100.0 * total_correct / n) if n else 0.0,
        "active_acc_pct": (100.0 * (buy_correct + sell_correct) / active) if active else 0.0,
    }


def _simulate_strategy(
    name: str,
    records: list[Record],
    score_fn,
    config: HeadConfig,
    scenario: Scenario,
    *,
    hold_days: int,
) -> dict[str, object]:
    decision_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    next_available_idx = 0

    for idx, query in enumerate(records):
        actual_ret_7d = query.future.get("7d")
        if actual_ret_7d is None:
            continue
        pool = records[:idx]
        if len(pool) < config.k:
            continue

        scored = sorted(
            ((score_fn(query, candidate), candidate) for candidate in pool),
            key=lambda item: item[0],
            reverse=True,
        )[: config.k]
        avgs = [avg for _, row in scored if (avg := _weighted_avg(row, config.return_weights)) is not None]
        if not avgs:
            continue
        weighted_avg = float(np.mean(avgs))
        signal = _signal(weighted_avg, config.buy_thr, config.sell_thr)
        actual_ret_7d = float(actual_ret_7d)
        actual_cls = _actual_class(actual_ret_7d, config.buy_thr, config.sell_thr)
        evidence = _top_evidence(scored, limit=config.k)

        eligible = idx >= next_available_idx and signal != "HOLD"
        decision_rows.append(
            {
                "date": query.date,
                "signal": signal,
                "weighted_avg_return": round(weighted_avg, 4),
                "actual_return_7d": round(actual_ret_7d, 4),
                "actual_class_7d": actual_cls,
                "eligible_to_trade": eligible,
                "eligibility_reason": "eligible" if eligible else ("cooldown" if idx < next_available_idx else "hold_signal"),
                "evidence": evidence,
            }
        )

        if not eligible:
            continue
        entry_idx = idx + 1
        exit_idx = entry_idx + hold_days
        if exit_idx >= len(records):
            continue
        entry_open = records[entry_idx].open
        exit_open = records[exit_idx].open
        if entry_open <= 0 or exit_open <= 0:
            continue
        side = 1.0 if signal == "BUY" else -1.0
        gross_return = side * ((exit_open - entry_open) / entry_open) * 100.0
        net_return = gross_return - scenario.roundtrip_cost_pct
        next_available_idx = exit_idx

        trade_rows.append(
            {
                "signal_date": query.date,
                "entry_date": records[entry_idx].date,
                "exit_date": records[exit_idx].date,
                "signal": signal,
                "entry_open": round(entry_open, 4),
                "exit_open": round(exit_open, 4),
                "gross_return_pct": round(gross_return, 4),
                "net_return_pct": round(net_return, 4),
                "label_return_7d_pct": round(actual_ret_7d, 4),
                "cost_pct": round(scenario.roundtrip_cost_pct, 4),
                "evidence": evidence,
            }
        )

    cls_summary = _classification_summary(decision_rows, config.buy_thr, config.sell_thr)
    trade_returns = [float(row["net_return_pct"]) for row in trade_rows]
    wins = [v for v in trade_returns if v > 0.0]
    losses = [v for v in trade_returns if v <= 0.0]
    total_days = (records[-1].day - records[0].day).days + 1 if records else 0
    trades_per_year = (len(trade_rows) / total_days * 365.0) if total_days > 0 else 0.0
    exposure_days = len(trade_rows) * hold_days
    exposure_pct = 100.0 * exposure_days / total_days if total_days > 0 else 0.0
    total_return_pct = (np.prod([1.0 + ret / 100.0 for ret in trade_returns]) - 1.0) * 100.0 if trade_returns else 0.0

    trading_summary = {
        "trade_count": len(trade_rows),
        "trades_per_year": trades_per_year,
        "exposure_pct": exposure_pct,
        "avg_trade_return_pct": float(np.mean(trade_returns)) if trade_returns else 0.0,
        "median_trade_return_pct": float(np.median(trade_returns)) if trade_returns else 0.0,
        "win_rate_pct": 100.0 * len(wins) / len(trade_rows) if trade_rows else 0.0,
        "profit_factor": _profit_factor(trade_returns),
        "total_return_pct": total_return_pct,
        "sharpe": _annualized_sharpe(trade_returns, trades_per_year),
        "sortino": _annualized_sortino(trade_returns, trades_per_year),
        "max_drawdown_pct": _max_drawdown(trade_returns),
        "turnover_pct_per_trade": scenario.roundtrip_cost_pct,
        "avg_winner_pct": float(np.mean(wins)) if wins else 0.0,
        "avg_loser_pct": float(np.mean(losses)) if losses else 0.0,
    }

    return {
        "name": name,
        "scenario": {
            "name": scenario.name,
            "fee_bps_side": scenario.fee_bps_side,
            "slippage_bps_side": scenario.slippage_bps_side,
            "roundtrip_cost_pct": scenario.roundtrip_cost_pct,
            "hold_days": hold_days,
            "entry_rule": "next_day_open",
            "exit_rule": f"+{hold_days}d_open",
            "position_model": "long_short_flat",
            "overlap": "disallow",
        },
        "config": {
            "retriever": config.retriever,
            "k": config.k,
            "buy_threshold": config.buy_thr,
            "sell_threshold": config.sell_thr,
            "return_weights": config.return_weights,
        },
        "classification_summary": cls_summary,
        "trading_summary": trading_summary,
        "decision_rows": decision_rows,
        "trade_rows": trade_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Market-path backtest using next-day open entry, +7d open exit, "
            "long/short/flat, costs, and evidence rows."
        )
    )
    parser.add_argument("--input-ndjson", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--artifact", default="stockmem/config/learned_retriever_finbert.json")
    parser.add_argument("--weights", default="stockmem/config/weights.auto.json")
    parser.add_argument("--learned-head", default="stockmem/config/knn_head.learned_finbert_rolling_stable.json")
    parser.add_argument("--fixed-head", default="stockmem/config/knn_head.fixed_knn_rolling_stable.json")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--hold-days", type=int, default=7)
    parser.add_argument("--output", default="artifacts/backtests/knn_head_market_path_backtest.json")
    args = parser.parse_args()

    records = _load_ndjson(Path(args.input_ndjson), start_date=args.start_date)
    learned_metric = LearnedDiagonalMetric.load(args.artifact)
    search_weights = _load_search_weights(Path(args.weights))

    learned_candidate = _load_head_config(Path(args.learned_head))
    fixed_candidate = _load_head_config(Path(args.fixed_head))
    production = HeadConfig(
        name="production_knn_returns_baseline",
        retriever="fixed_knn",
        k=5,
        buy_thr=2.0,
        sell_thr=2.0,
        return_weights=PRODUCTION_RETURN_WEIGHTS,
    )

    scenarios = [
        Scenario("no_cost", fee_bps_side=0.0, slippage_bps_side=0.0),
        Scenario("fee_10bps_side", fee_bps_side=10.0, slippage_bps_side=0.0),
        Scenario("fee_10bps_plus_slippage_5bps_side", fee_bps_side=10.0, slippage_bps_side=5.0),
    ]

    fixed_score = lambda q, c: _fixed_score(q, c, search_weights)
    learned_score = lambda q, c: learned_metric.score(q.blocks, c.blocks)
    strategy_defs = [
        ("learned_finbert_rolling_stable", learned_score, learned_candidate),
        ("fixed_knn_rolling_stable", fixed_score, fixed_candidate),
        ("production_knn_returns_baseline", fixed_score, production),
    ]

    results: list[dict[str, object]] = []
    for scenario in scenarios:
        for name, score_fn, config in strategy_defs:
            results.append(
                _simulate_strategy(
                    name,
                    records,
                    score_fn,
                    config,
                    scenario,
                    hold_days=args.hold_days,
                )
            )

    payload = {
        "start_date": args.start_date,
        "record_count": len(records),
        "search_weights": {
            "w_factor": search_weights[0],
            "w_indicator": search_weights[1],
            "w_price": search_weights[2],
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "record_count": len(records)}, indent=2))
    for result in results:
        cls = result["classification_summary"]
        trd = result["trading_summary"]
        scen = result["scenario"]
        print(
            f"{result['name']} [{scen['name']}]: "
            f"overall_DA={cls['overall_da_pct']:.1f}% "
            f"active_acc={cls['active_acc_pct']:.1f}% "
            f"coverage={cls['coverage_pct']:.1f}% "
            f"trades={trd['trade_count']} "
            f"total_ret={trd['total_return_pct']:.1f}% "
            f"sharpe={trd['sharpe']:.3f} "
            f"sortino={trd['sortino']:.3f} "
            f"mdd={trd['max_drawdown_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
