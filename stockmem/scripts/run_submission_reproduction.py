from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, check=True)


def _write_manifest(out_dir: Path, dataset: Path, *, skip_llm: bool) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "test_split": ["2025-07-01", "2026-05-01"],
        "label_threshold": 2.0,
        "skip_llm": skip_llm,
        "outputs": {
            "naive_llm": "current_context_ai_eval/summary.json",
            "structured": "learned_strict_test/summary.json",
            "feature_ablation": "fixed_knn_component_ablation/summary.json",
            "tables": "tables/",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce StockMem submission metrics")
    parser.add_argument("--dataset", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--out-dir", default="submission/stockmem_2026_07")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Groq-backed naive LLM run")
    parser.add_argument("--llm-prompt-style", default="legacy", choices=["compact", "legacy"])
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    naive_dir = out_dir / "current_context_ai_eval"
    strict_dir = out_dir / "learned_strict_test"
    ablation_dir = out_dir / "fixed_knn_component_ablation"
    tables_dir = out_dir / "tables"

    py = sys.executable
    if not args.skip_llm:
        _run(
            [
                py,
                "aihub/scripts/evaluate_naive_llm_baseline.py",
                "--data",
                str(dataset),
                "--weights",
                "stockmem/config/weights.auto.json",
                "--fixed-head",
                "stockmem/config/knn_head.fixed_knn_rolling_stable.json",
                "--out-dir",
                str(naive_dir),
                "--prompt-style",
                args.llm_prompt_style,
                "--prediction-max-attempts",
                "6",
                "--prediction-retry-delay-seconds",
                "15",
            ]
        )

    _run(
        [
            py,
            "stockmem/scripts/evaluate_stockmem_strict_models.py",
            "--data",
            str(dataset),
            "--weights",
            "stockmem/config/weights.auto.json",
            "--artifact",
            "stockmem/config/learned_retriever_finbert.json",
            "--fixed-head",
            "stockmem/config/knn_head.fixed_knn_rolling_stable.json",
            "--learned-head",
            "stockmem/config/knn_head.learned_finbert_rolling_stable.json",
            "--out-dir",
            str(strict_dir),
        ]
    )
    _run(
        [
            py,
            "stockmem/scripts/evaluate_stockmem_feature_ablation.py",
            "--data",
            str(dataset),
            "--weights",
            "stockmem/config/weights.auto.json",
            "--fixed-head",
            "stockmem/config/knn_head.fixed_knn_rolling_stable.json",
            "--out-dir",
            str(ablation_dir),
        ]
    )

    naive_summary = naive_dir / "summary.json"
    if args.skip_llm:
        naive_summary = Path("artifacts/current_context_ai_eval/summary.json")
    _run(
        [
            py,
            "stockmem/scripts/export_stockmem_report_tables.py",
            "--strict-summary",
            str(strict_dir / "summary.json"),
            "--naive-summary",
            str(naive_summary),
            "--ablation-summary",
            str(ablation_dir / "summary.json"),
            "--out-dir",
            str(tables_dir),
        ]
    )
    _write_manifest(out_dir, dataset, skip_llm=args.skip_llm)
    print(f"wrote submission bundle to {out_dir}")


if __name__ == "__main__":
    main()
