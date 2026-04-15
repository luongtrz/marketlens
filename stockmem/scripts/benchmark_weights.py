from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimize_weights import evaluate, load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs candidate StockMem weights")
    parser.add_argument("--data", required=True, help="Path to vectorized dataset JSON")
    parser.add_argument("--weights", required=True, help="Path to candidate weights JSON")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=250)
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    baseline = (0.35, 0.2, 0.45)
    base_metrics = evaluate(rows, baseline[0], baseline[1], baseline[2], args.k, args.warmup)

    payload = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    cand = (
        float(payload["w1_factor"]),
        float(payload["w2_indicator"]),
        float(payload["w3_price"]),
    )
    cand_metrics = evaluate(rows, cand[0], cand[1], cand[2], args.k, args.warmup)

    out = {
        "baseline": {"weights": {"w1": baseline[0], "w2": baseline[1], "w3": baseline[2]}, "metrics": base_metrics},
        "candidate": {"weights": {"w1": cand[0], "w2": cand[1], "w3": cand[2]}, "metrics": cand_metrics},
        "delta_combined": cand_metrics["combined"] - base_metrics["combined"],
        "delta_da": cand_metrics["da"] - base_metrics["da"],
        "delta_sharpe": cand_metrics["sharpe"] - base_metrics["sharpe"],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
