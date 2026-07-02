from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_COLUMNS = (
    "n",
    "overall_acc",
    "active_acc",
    "coverage",
    "hit_at_5_same_sign",
    "buy_rate",
    "hold_rate",
    "sell_rate",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(row: dict[str, Any], column: str) -> str:
    value = row.get(column)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return ""


def _write_metric_table(
    rows: list[dict[str, Any]],
    *,
    name_key: str,
    title: str,
    md_path: Path,
    csv_path: Path,
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", *METRIC_COLUMNS]
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(header) + " |",
        "| --- | " + " | ".join("---:" for _ in METRIC_COLUMNS) + " |",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            out = {"model": row[name_key]}
            out.update({column: _metric_value(row, column) for column in METRIC_COLUMNS})
            writer.writerow(out)
            lines.append("| " + " | ".join(out[column] for column in header) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paired_table(summary: dict[str, Any], out_dir: Path) -> None:
    paired = summary.get("paired_stats", {})
    rows: list[dict[str, str]] = []
    for pair_name, payload in paired.items():
        for metric in ("overall_acc", "active_acc", "coverage", "hit_at_5_same_sign"):
            item = payload.get(metric, {})
            rows.append(
                {
                    "pair": pair_name,
                    "metric": metric,
                    "mean_delta": f"{float(item.get('mean_delta', 0.0)):+.4f}",
                    "ci_low": f"{float(item.get('ci_low', 0.0)):+.4f}",
                    "ci_high": f"{float(item.get('ci_high', 0.0)):+.4f}",
                    "mcnemar_p": f"{float(payload.get('mcnemar', {}).get('p_value', 1.0)):.6f}",
                }
            )
    if not rows:
        return
    header = ["pair", "metric", "mean_delta", "ci_low", "ci_high", "mcnemar_p"]
    md_lines = [
        "# Paired Statistical Tests",
        "",
        "| " + " | ".join(header) + " |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    csv_path = out_dir / "paired_stat_tests.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            md_lines.append("| " + " | ".join(row[column] for column in header) + " |")
    (out_dir / "paired_stat_tests.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )


def export_tables(
    *,
    strict_summary: Path,
    naive_summary: Path,
    ablation_summary: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    strict = _load(strict_summary)
    naive = _load(naive_summary)
    ablation = _load(ablation_summary)

    _write_metric_table(
        strict["models"],
        name_key="name",
        title="Primary Structured StockMem Models",
        md_path=out_dir / "primary_structured_models.md",
        csv_path=out_dir / "primary_structured_models.csv",
    )
    _write_metric_table(
        naive["models"],
        name_key="name",
        title="Naive LLM Baseline vs StockMem",
        md_path=out_dir / "naive_llm_vs_stockmem.md",
        csv_path=out_dir / "naive_llm_vs_stockmem.csv",
    )
    _write_metric_table(
        ablation["variants"],
        name_key="name",
        title="StockMem Feature Block Ablation",
        md_path=out_dir / "feature_ablation.md",
        csv_path=out_dir / "feature_ablation.csv",
    )
    _write_paired_table(strict, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact report tables from StockMem summaries")
    parser.add_argument("--strict-summary", default="artifacts/learned_strict_test_v3/summary.json")
    parser.add_argument("--naive-summary", default="artifacts/current_context_ai_eval/summary.json")
    parser.add_argument("--ablation-summary", default="artifacts/fixed_knn_component_ablation/summary.json")
    parser.add_argument("--out-dir", default="results_tables")
    args = parser.parse_args()

    export_tables(
        strict_summary=Path(args.strict_summary),
        naive_summary=Path(args.naive_summary),
        ablation_summary=Path(args.ablation_summary),
        out_dir=Path(args.out_dir),
    )
    print(f"wrote compact tables to {args.out_dir}")


if __name__ == "__main__":
    main()
