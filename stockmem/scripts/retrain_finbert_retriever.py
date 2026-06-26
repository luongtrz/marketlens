from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NDJSON = ROOT / "data" / "exports" / "stockmem_records.ndjson"
DEFAULT_DATASET = ROOT / "stockmem" / "data" / "real_optimizer_finbert.json"
DEFAULT_ARTIFACT = ROOT / "stockmem" / "config" / "learned_retriever_finbert.json"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    env = dict(**__import__("os").environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild optimizer data from stockmem_records using FinBERT sentiment, "
            "then retrain the learned retriever with the standard train/val/test split."
        )
    )
    parser.add_argument("--input-ndjson", default=str(DEFAULT_NDJSON))
    parser.add_argument("--dataset-output", default=str(DEFAULT_DATASET))
    parser.add_argument("--artifact-output", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-metric",
        choices=["hit", "combined", "ndcg", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--outcome-weight", type=float, default=0.45)
    parser.add_argument("--regime-weight", type=float, default=0.35)
    parser.add_argument("--surface-weight", type=float, default=0.20)
    parser.add_argument("--init-artifact", default=None)
    parser.add_argument("--skip-optuna", action="store_true")
    args = parser.parse_args()

    python = sys.executable

    _run(
        [
            python,
            "stockmem/scripts/regen_optimizer_data.py",
            "--input-ndjson",
            args.input_ndjson,
            "--output",
            args.dataset_output,
            "--sentiment-source",
            "finbert",
        ]
    )
    _run(
        [
            python,
            "stockmem/scripts/train_learned_retriever.py",
            "--data",
            args.dataset_output,
            "--output",
            args.artifact_output,
            "--trials",
            str(args.trials),
            "--epochs",
            str(args.epochs),
            "--seeds",
            str(args.seeds),
            "--seed",
            str(args.seed),
            "--selection-metric",
            args.selection_metric,
            "--outcome-weight",
            str(args.outcome_weight),
            "--regime-weight",
            str(args.regime_weight),
            "--surface-weight",
            str(args.surface_weight),
        ]
        + (["--init-artifact", args.init_artifact] if args.init_artifact else [])
        + (["--skip-optuna"] if args.skip_optuna else [])
    )


if __name__ == "__main__":
    main()
