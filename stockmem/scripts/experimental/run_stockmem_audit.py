from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_step(
    *,
    name: str,
    args: list[str],
    out_dir: Path,
    steps: list[dict[str, object]],
) -> None:
    log_path = out_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    print(f"[audit] START {name}", flush=True)
    print("+ " + " ".join(args), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {name}\n")
        handle.write(f"started_at={started.isoformat()}\n")
        handle.write("+ " + " ".join(args) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - t0
    status = {
        "name": name,
        "args": args,
        "log": str(log_path),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "returncode": proc.returncode,
    }
    steps.append(status)
    print(f"[audit] END {name} rc={proc.returncode} elapsed={elapsed:.1f}s log={log_path}", flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metric_audit(out_dir: Path, *, strict_summary: Path, ablation_summary: Path, policy_json: Path) -> None:
    strict = _load_json(strict_summary)
    ablation = _load_json(ablation_summary)
    policy = _load_json(policy_json)

    findings: list[str] = []
    strict_threshold = strict.get("label_threshold")
    ablation_threshold = ablation.get("label_threshold")
    if strict_threshold != 2.0:
        findings.append(f"strict label_threshold expected 2.0, got {strict_threshold}")
    if ablation_threshold != 2.0:
        findings.append(f"ablation label_threshold expected 2.0, got {ablation_threshold}")

    for summary_name, summary, row_key in (
        ("strict", strict, "models"),
        ("ablation", ablation, "variants"),
    ):
        for row in summary.get(row_key, []):
            total = int(row.get("n", 0))
            predicted = row.get("predicted_counts", {})
            predicted_total = sum(int(v) for v in predicted.values())
            if total != predicted_total:
                findings.append(
                    f"{summary_name}/{row.get('name')} n={total} != predicted_counts sum={predicted_total}"
                )
            coverage = float(row.get("coverage", 0.0))
            buy_rate = float(row.get("buy_rate", 0.0))
            sell_rate = float(row.get("sell_rate", 0.0))
            if abs(coverage - (buy_rate + sell_rate)) > 1e-9:
                findings.append(
                    f"{summary_name}/{row.get('name')} coverage mismatch: {coverage} vs buy+sell {buy_rate + sell_rate}"
                )

    policy_metrics = policy.get("test_metrics", {})
    policy_note = (
        "calibrate_policy corrected: HOLD is correct only when actual labeled direction is HOLD; "
        "probabilities use labeled UP/DOWN/HOLD buckets."
    )
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strict_summary": str(strict_summary),
        "ablation_summary": str(ablation_summary),
        "policy_json": str(policy_json),
        "policy_note": policy_note,
        "policy_test_metrics": policy_metrics,
        "findings": findings,
        "passed": not findings,
    }
    (out_dir / "metric_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Metric Audit",
        "",
        policy_note,
        "",
        f"- strict summary: `{strict_summary}`",
        f"- ablation summary: `{ablation_summary}`",
        f"- corrected policy: `{policy_json}`",
        f"- passed: `{not findings}`",
        "",
        "## Corrected Policy Test Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "da",
        "buy_da",
        "sell_da",
        "coverage",
        "sharpe",
        "sortino",
        "max_drawdown",
        "brier",
        "ece",
        "n",
        "n_buy",
        "n_sell",
        "n_hold",
    ):
        if key in policy_metrics:
            value = policy_metrics[key]
            if isinstance(value, float):
                lines.append(f"| `{key}` | {value:.6f} |")
            else:
                lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(f"- {item}" for item in findings)
    else:
        lines.append("- No metric consistency findings in strict/ablation summaries.")
    (out_dir / "metric_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StockMem audit/reproduction jobs in Docker")
    parser.add_argument("--dataset", default="data/exports/stockmem_records.ndjson")
    parser.add_argument("--policy-data", default="stockmem/data/real_optimizer_finbert.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--min-pool-size", type=int, default=5)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    out_dir = Path(args.out_dir or f"artifacts/audit_runs/stockmem_audit_{started.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Path(args.dataset)
    policy_data = Path(args.policy_data)
    py = sys.executable
    steps: list[dict[str, object]] = []

    manifest: dict[str, object] = {
        "created_at": started.isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset) if dataset.exists() else None,
        "policy_data": str(policy_data),
        "policy_data_sha256": _sha256(policy_data) if policy_data.exists() else None,
        "out_dir": str(out_dir),
        "steps": steps,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        _run_step(
            name="py_compile",
            args=[
                py,
                "-m",
                "py_compile",
                "stockmem/scripts/calibrate_policy.py",
                "stockmem/scripts/evaluate_stockmem_strict_models.py",
                "stockmem/scripts/evaluate_stockmem_feature_ablation.py",
                "stockmem/scripts/experimental/analyze_regime_slices.py",
                "stockmem/scripts/experimental/evaluate_structured_models_full_history.py",
            ],
            out_dir=out_dir,
            steps=steps,
        )
        _run_step(
            name="strict_models",
            args=[
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
                str(out_dir / "learned_strict_test"),
            ],
            out_dir=out_dir,
            steps=steps,
        )
        _run_step(
            name="feature_ablation",
            args=[
                py,
                "stockmem/scripts/evaluate_stockmem_feature_ablation.py",
                "--data",
                str(dataset),
                "--weights",
                "stockmem/config/weights.auto.json",
                "--fixed-head",
                "stockmem/config/knn_head.fixed_knn_rolling_stable.json",
                "--out-dir",
                str(out_dir / "fixed_knn_component_ablation"),
            ],
            out_dir=out_dir,
            steps=steps,
        )
        _run_step(
            name="regime_slices",
            args=[
                py,
                "stockmem/scripts/experimental/analyze_regime_slices.py",
                "--dataset",
                str(dataset),
                "--out-dir",
                str(out_dir / "exploratory_regime_slices"),
                "--min-slice-size",
                "20",
            ],
            out_dir=out_dir,
            steps=steps,
        )
        _run_step(
            name="full_history_structured",
            args=[
                py,
                "stockmem/scripts/experimental/evaluate_structured_models_full_history.py",
                "--data",
                str(dataset),
                "--out-dir",
                str(out_dir / "full_history_structured_models"),
                "--start-date",
                "2018-01-01",
                "--min-pool-size",
                str(args.min_pool_size),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
            ],
            out_dir=out_dir,
            steps=steps,
        )
        _run_step(
            name="calibrate_policy_corrected",
            args=[
                py,
                "stockmem/scripts/calibrate_policy.py",
                "--data",
                str(policy_data),
                "--artifact",
                "stockmem/config/learned_retriever_finbert.json",
                "--output",
                str(out_dir / "policy_corrected.json"),
            ],
            out_dir=out_dir,
            steps=steps,
        )
        _write_metric_audit(
            out_dir,
            strict_summary=out_dir / "learned_strict_test" / "summary.json",
            ablation_summary=out_dir / "fixed_knn_component_ablation" / "summary.json",
            policy_json=out_dir / "policy_corrected.json",
        )
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["steps"] = steps
        manifest["completed"] = bool(steps) and all(int(step["returncode"]) == 0 for step in steps)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[audit] manifest: {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
