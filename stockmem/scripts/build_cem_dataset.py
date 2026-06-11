from __future__ import annotations

import argparse
import json
from pathlib import Path

from stockmem.scripts.cem_dataset import label_rows, matured_pool, mine_candidates
from stockmem.scripts.optimize_weights import load_rows, validate_rows


DEFAULT_WEIGHTS = (0.544392055430515, 0.30908053253948164, 0.14156627274414413)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-clean CEM retriever dataset")
    parser.add_argument("--data", default="stockmem/data/real_optimizer.json")
    parser.add_argument("--output", default="stockmem/data/cem_dataset.json")
    parser.add_argument("--band", choices=["0.5sigma", "fixed"], default="0.5sigma")
    parser.add_argument("--fixed-band", type=float, default=0.01)
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    validate_rows(rows)
    labeled = label_rows(rows, band=args.band, fixed_band=args.fixed_band)
    records = [
        {
            "date": item.row.date,
            "split": item.split,
            "direction": item.direction,
            "band_value": item.band_value,
            "causal_volatility": item.causal_volatility,
            "factor_vec": item.row.factor_vec.tolist(),
            "indicator_vec": item.row.indicator_vec.tolist(),
            "price_vec": item.row.price_vec.tolist(),
            "event_vec": item.row.event_vec.tolist(),
            "future_return_7d": item.row.future_return_7d,
        }
        for item in labeled
    ]
    counts: dict[str, int] = {}
    for item in labeled:
        key = f"{item.split}:{item.direction}"
        counts[key] = counts.get(key, 0) + 1
    train_rows = [item for item in labeled if item.split == "train"]
    mined = [
        pair
        for anchor in train_rows
        if anchor.direction != 0
        if (
            pair := mine_candidates(
                anchor,
                matured_pool(train_rows, anchor),
                weights=DEFAULT_WEIGHTS,
                hard_negs=8,
            )
        )
        is not None
    ]
    pair_diagnostics = {
        "mineable_anchors": len(mined),
        "mean_positives": (
            sum(len(pair.positives) for pair in mined) / len(mined) if mined else 0.0
        ),
        "mean_negatives": (
            sum(len(pair.negatives) for pair in mined) / len(mined) if mined else 0.0
        ),
    }
    payload = {
        "metadata": {
            "source": args.data,
            "horizon_days": 7,
            "band": args.band,
            "fixed_band": args.fixed_band,
            "block_dims": [block.size for block in labeled[0].blocks],
            "counts": counts,
            "pair_diagnostics": pair_diagnostics,
            "mining_protocol": "outcome_regime_distillation_v1",
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
