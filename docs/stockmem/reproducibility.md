# StockMem Reproducibility

Generated outputs are not committed. Reproduce them into `submission/` or
`results_tables/` when needed for the report.

The official NDJSON dataset is also not committed in the clean submission
branch. Place it at `data/exports/stockmem_records.ndjson`, or pass another
path with `--dataset`.

## Official Docker Command

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /bin/sh \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python stockmem/scripts/run_submission_reproduction.py \
    --dataset data/exports/stockmem_records.ndjson \
    --out-dir submission/stockmem_2026_07"
```

If Groq quota is unavailable:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /bin/sh \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python stockmem/scripts/run_submission_reproduction.py \
    --dataset data/exports/stockmem_records.ndjson \
    --out-dir submission/stockmem_2026_07 \
    --skip-llm \
    --naive-summary artifacts/current_context_ai_eval/summary.json"
```

## Official Scripts

| Script | Purpose |
| --- | --- |
| `aihub/scripts/evaluate_naive_llm_baseline.py` | Current-context LLM baseline with resume/retry behavior. |
| `stockmem/scripts/evaluate_stockmem_strict_models.py` | Strict structured model comparison. |
| `stockmem/scripts/evaluate_stockmem_feature_ablation.py` | Fixed-kNN feature-block ablation. |
| `stockmem/scripts/export_stockmem_report_tables.py` | Compact Markdown/CSV table export. |
| `stockmem/scripts/run_submission_reproduction.py` | End-to-end reproduction orchestrator. |

## Expected Outputs

| Path | Contents |
| --- | --- |
| `submission/stockmem_2026_07/manifest.json` | Dataset checksum, split, output map. |
| `submission/stockmem_2026_07/current_context_ai_eval/summary.json` | Naive LLM vs structured baseline summary. |
| `submission/stockmem_2026_07/learned_strict_test/summary.json` | Primary structured model table and paired tests. |
| `submission/stockmem_2026_07/fixed_knn_component_ablation/summary.json` | Feature-block ablation. |
| `submission/stockmem_2026_07/tables/*.md` | Report-ready compact tables. |
| `submission/stockmem_2026_07/tables/*.csv` | Spreadsheet-friendly tables. |

## Audit Rules

- Do not silently convert failed LLM calls into `HOLD`.
- Use resume mode for LLM runs to avoid paying for completed rows again.
- Keep `artifacts/`, `results_tables/`, and `submission/` out of git.
- Treat per-record JSONL files and logs as audit artifacts, not thesis text.
