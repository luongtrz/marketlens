from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stockmem.src.config import SearchWeights
from stockmem.src.models import StockMemRecord
from stockmem.src.search.embedder import RecordEmbedder


def _get_optimize_module():
    try:
        from stockmem.scripts import optimize_weights as ow  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Cannot import stockmem/scripts/optimize_weights.py. "
            "Ensure stockmem scripts are available in PYTHONPATH."
        ) from exc
    return ow


def retrain_weights(
    records: list[StockMemRecord],
    *,
    horizon: str,
    k: int,
    warmup: int,
    trials: int,
    stable_top_k: int = 10,
    seed: int = 42,
) -> tuple[SearchWeights, dict[str, Any]]:
    """Bayesian (Optuna/TPE) retraining of search weights from in-memory records."""
    ow = _get_optimize_module()
    o = ow._require_optuna()

    filtered = [
        r for r in records
        if r.future_return_1d is not None
        and r.future_return_7d is not None
        and r.future_return_30d is not None
    ]
    if len(filtered) <= warmup:
        raise ValueError(
            f"Not enough labeled records for retraining: {len(filtered)} <= warmup({warmup})"
        )

    filtered.sort(key=lambda x: str(x.date))
    embedder = RecordEmbedder()
    embedder.rebuild_corpus(filtered)

    rows = []
    for rec in filtered:
        split = embedder.embed_split(rec)
        rows.append(
            ow.Row(
                date=str(rec.date),
                factor_vec=split.factor_vec,
                indicator_vec=split.indicator_vec,
                price_vec=split.price_vec,
                future_return_1d=float(rec.future_return_1d or 0.0),
                future_return_7d=float(rec.future_return_7d or 0.0),
                future_return_30d=float(rec.future_return_30d or 0.0),
            )
        )

    study = o.create_study(
        direction="maximize",
        sampler=o.samplers.TPESampler(seed=seed),
        study_name="stockmem_auto_retrain",
    )
    objective = ow.make_objective(
        rows=rows,
        horizon=horizon,
        k=k,
        warmup=warmup,
        sharpe_mode="nonoverlap",
        cv_folds=1,
        cv_holdout_ratio=0.2,
        cv_min_holdout=120,
        maturity_guard=True,
    )
    study.optimize(objective, n_trials=max(1, trials), show_progress_bar=False)

    w1, w2, w3, stable_meta = ow.select_stable_median_weights(
        study=study,
        top_k=stable_top_k,
    )
    weights = SearchWeights(w1_factor=w1, w2_indicator=w2, w3_price=w3)
    metrics = ow.walk_forward_evaluate(
        rows=rows,
        w1=w1,
        w2=w2,
        w3=w3,
        k=k,
        warmup=warmup,
        horizon=horizon,
        sharpe_mode="nonoverlap",
        maturity_guard=True,
    )
    payload = {
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "horizon": horizon,
        "n_records": len(rows),
        "warmup": warmup,
        "k": k,
        "trials": trials,
        "evaluation_protocol": {
            "version": ow.EVALUATION_PROTOCOL_VERSION,
            "maturity_guard": True,
        },
        "stable_selection": stable_meta,
        "weights": asdict(weights),
        "metrics": metrics,
        "best_score": float(study.best_value),
    }
    return weights, payload


def write_weights_snapshot(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
