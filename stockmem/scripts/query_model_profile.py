from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from stockmem.scripts.experimental.evaluate_consensus_retriever_heads import (
    _candidate_heads,
    _load_config,
)
from stockmem.scripts.experimental.train_majority_consensus_retriever import (
    QueryCache,
    _fixed_scores,
    _learned_scores,
    _load_rows,
    _matured_pool,
    _minmax,
    _rank_scores,
    _regime_scores,
    _top_indices,
)
from stockmem.scripts.ndjson_eval_common import actual_signal, load_knn_weights
from stockmem.src.search.learned_metric import LearnedDiagonalMetric


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = ROOT / "stockmem" / "config" / "model_profiles.json"
DEFAULT_BTC_WEIGHTS = ROOT / "stockmem" / "config" / "weights.auto.json"


def _resolve(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def _load_profile(path: Path, symbol: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    key = symbol.upper()
    if key not in profiles:
        raise SystemExit(f"No model profile for symbol={key!r} in {path}")
    return profiles[key]


def _select_query(rows: list[Any], symbol: str, query_date: date | None) -> Any:
    symbol = symbol.upper()
    candidates = [row for row in rows if row.record.symbol.upper() == symbol]
    if not candidates:
        raise SystemExit(f"No rows found for symbol={symbol!r}")
    if query_date is None:
        return candidates[-1]
    matches = [row for row in candidates if row.date == query_date]
    if not matches:
        available = f"{candidates[0].date}..{candidates[-1].date}"
        raise SystemExit(f"No row found for {symbol} on {query_date}; available={available}")
    return matches[0]


def _head_by_name(name: str, label_threshold: float):
    for head in _candidate_heads(label_threshold):
        if head.name == name:
            return head
    raise SystemExit(f"Unsupported decision head in profile: {name}")


def _record_horizon_returns(record: Any) -> dict[str, float | None]:
    return {
        "1d": record.future_return_1d,
        "3d": record.future_return_3d,
        "7d": record.future_return_7d,
        "15d": record.future_return_15d,
        "30d": record.future_return_30d,
    }


def _evidence_row(
    *,
    candidate: Any,
    rank: int,
    final_score: float,
    fixed_score: float,
    learned_score: float,
    recency_score: float,
    regime_score: float,
    label_threshold: float,
) -> dict[str, Any]:
    record = candidate.record
    return {
        "rank": rank,
        "date": record.date.isoformat(),
        "symbol": record.symbol,
        "signal_7d": actual_signal(record.future_return_7d, label_threshold),
        "future_return_7d": record.future_return_7d,
        "future_returns": _record_horizon_returns(record),
        "scores": {
            "final": round(float(final_score), 8),
            "fixed": round(float(fixed_score), 8),
            "learned": round(float(learned_score), 8),
            "recency": round(float(recency_score), 8),
            "regime": round(float(regime_score), 8),
        },
        "summary": record.summary,
        "factors": record.factors,
        "sentiment_label": record.sentiment_label,
        "sentiment_score": record.sentiment_score,
        "finbert_sentiment_score": record.finbert_sentiment_score,
    }


def query_profile(
    *,
    symbol: str,
    query_date: date | None,
    profiles_path: Path,
    top_k: int | None,
) -> dict[str, Any]:
    symbol = symbol.upper()
    profile = _load_profile(profiles_path, symbol)
    label_threshold = float(profile.get("label_threshold", 2.0))
    evidence_top_k = int(top_k or profile.get("evidence_top_k", 10))

    data_path = _resolve(profile["dataset"])
    artifact_path = _resolve(profile["learned_retriever_artifact"])
    config_path = _resolve(profile["consensus_retriever_config"])
    weights_path = _resolve(profile.get("fixed_weights", DEFAULT_BTC_WEIGHTS))

    rows = _load_rows(data_path, label_threshold=label_threshold)
    query = _select_query(rows, symbol, query_date)
    pool = _matured_pool(rows, query)
    if not pool:
        raise SystemExit(f"No matured historical candidates for {symbol} on {query.date}")

    fixed_weights = load_knn_weights(weights_path)
    learned_metric = LearnedDiagonalMetric.load(artifact_path)
    retriever_config = _load_config(config_path)

    cache = QueryCache(
        query_date=query.date,
        actual_id=query.label_id,
        candidate_labels=np.asarray([candidate.label_id for candidate in pool], dtype=np.int8),
        fixed=_minmax(_fixed_scores(query, pool, fixed_weights)),
        learned=_minmax(_learned_scores(query, pool, learned_metric)),
        age_days=np.asarray([(query.date - candidate.date).days for candidate in pool], dtype=np.float64),
        regime=_regime_scores(query, pool),
    )
    scores = _rank_scores(cache, retriever_config)
    top_indices = _top_indices(scores, min(evidence_top_k, scores.size))
    ranked_records = [pool[int(index)].record for index in top_indices]

    head_name = str(profile["decision_head"])
    head = _head_by_name(head_name, label_threshold)
    predicted_signal, confidence = head.head(ranked_records)
    actual = actual_signal(query.record.future_return_7d, label_threshold)
    same_count = int(np.sum(cache.candidate_labels[top_indices] == query.label_id))
    recency_scores = np.exp(-cache.age_days / retriever_config.recency_half_life_days)

    evidence = [
        _evidence_row(
            candidate=pool[int(index)],
            rank=rank,
            final_score=float(scores[int(index)]),
            fixed_score=float(cache.fixed[int(index)]),
            learned_score=float(cache.learned[int(index)]),
            recency_score=float(recency_scores[int(index)]),
            regime_score=float(cache.regime[int(index)]),
            label_threshold=label_threshold,
        )
        for rank, index in enumerate(top_indices, start=1)
    ]

    return {
        "symbol": symbol,
        "query_date": query.date.isoformat(),
        "profile": {
            "dataset": str(data_path),
            "learned_retriever_artifact": str(artifact_path),
            "consensus_retriever_config": str(config_path),
            "fixed_weights": str(weights_path),
            "decision_head": head_name,
            "evidence_top_k": evidence_top_k,
            "label_threshold": label_threshold,
        },
        "query": {
            "actual_signal_7d": actual,
            "actual_return_7d": query.record.future_return_7d,
            "future_returns": _record_horizon_returns(query.record),
            "summary": query.record.summary,
            "factors": query.record.factors,
            "sentiment_label": query.record.sentiment_label,
            "sentiment_score": query.record.sentiment_score,
            "finbert_sentiment_score": query.record.finbert_sentiment_score,
        },
        "prediction": {
            "signal": predicted_signal,
            "confidence": confidence,
            "same_d7_count_at_k": same_count,
            "majority_same_d7": same_count >= ((len(top_indices) + 1) // 2),
        },
        "retriever_config": retriever_config.as_dict(),
        "pool_size": len(pool),
        "evidence": evidence,
    }


def _print_human(payload: dict[str, Any]) -> None:
    profile = payload["profile"]
    prediction = payload["prediction"]
    query = payload["query"]
    print(f"StockMem query {payload['symbol']} {payload['query_date']}")
    print(f"artifact: {profile['learned_retriever_artifact']}")
    print(f"retriever_config: {profile['consensus_retriever_config']}")
    print(f"head: {profile['decision_head']}")
    print(
        "prediction: "
        f"{prediction['signal']} confidence={prediction['confidence']} "
        f"same_d7@{profile['evidence_top_k']}={prediction['same_d7_count_at_k']} "
        f"majority={prediction['majority_same_d7']}"
    )
    print(f"actual: {query['actual_signal_7d']} return_7d={query['actual_return_7d']}")
    print("")
    print("rank | date       | signal | ret7d    | final    | learned  | recency  | regime")
    print("-----+------------+--------+----------+----------+----------+----------+-------")
    for row in payload["evidence"]:
        scores = row["scores"]
        ret7 = row["future_return_7d"]
        ret7_text = "None" if ret7 is None else f"{float(ret7): .4f}"
        print(
            f"{row['rank']:>4} | {row['date']} | {row['signal_7d']:<6} | "
            f"{ret7_text:>8} | {scores['final']:>8.4f} | "
            f"{scores['learned']:>8.4f} | {scores['recency']:>8.4f} | "
            f"{scores['regime']:>6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the maintained StockMem model profile from local NDJSON + artifacts."
    )
    parser.add_argument("--symbol", default=None, help="Asset symbol, e.g. BTC or ETH. Defaults to model_profiles default_symbol.")
    parser.add_argument("--date", default=None, help="Query date in YYYY-MM-DD. Defaults to latest labeled row.")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES), help="Path to stockmem/config/model_profiles.json.")
    parser.add_argument("--top-k", type=int, default=None, help="Override evidence top-k from the model profile.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload instead of the compact table.")
    parser.add_argument("--out", default=None, help="Optional path to write the full JSON payload.")
    args = parser.parse_args()

    profiles_path = _resolve(args.profiles)
    profiles_payload = json.loads(profiles_path.read_text(encoding="utf-8"))
    symbol = (args.symbol or profiles_payload.get("default_symbol") or "BTC").upper()
    query_date = date.fromisoformat(args.date) if args.date else None

    payload = query_profile(
        symbol=symbol,
        query_date=query_date,
        profiles_path=profiles_path,
        top_k=args.top_k,
    )
    if args.out:
        out_path = _resolve(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_human(payload)


if __name__ == "__main__":
    main()
