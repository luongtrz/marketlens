# StockMem Submission Reproduction

This is the clean submission entrypoint for the StockMem experiments. The repository should submit code, configuration, and documentation only. Generated metrics, JSONL predictions, logs, and table exports are reproducible outputs and should remain outside git.

## Official Claim

The official evaluation uses the held-out split from `2025-07-01` to `2026-05-01` with `305` rows and a `2%` threshold on `future_return_7d`.

The report should emphasize:

- the structured StockMem pipeline beats the naive current-context LLM baseline on the same held-out window;
- fixed kNN retrieval is the robust evidence generator;
- the best strict structured variant is fixed retrieval plus the learned stable head;
- learned retriever, hybrid reranking, and head-aligned training are useful negative or diagnostic experiments, not the recommended production path.

## One-Command Reproduction

Run from the repository root:

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

If Groq quota is unavailable, reproduce the structured metrics and table export without rerunning the LLM:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  --entrypoint /bin/sh \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python stockmem/scripts/run_submission_reproduction.py \
    --dataset data/exports/stockmem_records.ndjson \
    --out-dir submission/stockmem_2026_07 \
    --skip-llm"
```

## Output

The reproduction command writes:

- `submission/stockmem_2026_07/manifest.json`
- `submission/stockmem_2026_07/current_context_ai_eval/summary.json`
- `submission/stockmem_2026_07/learned_strict_test/summary.json`
- `submission/stockmem_2026_07/fixed_knn_component_ablation/summary.json`
- `submission/stockmem_2026_07/tables/*.md`
- `submission/stockmem_2026_07/tables/*.csv`

Only compact tables should be used in the thesis/report. Per-record JSONL files and raw logs are audit artifacts, not submission material.

## Official Scripts

- `aihub/scripts/evaluate_naive_llm_baseline.py`
- `stockmem/scripts/evaluate_stockmem_strict_models.py`
- `stockmem/scripts/evaluate_stockmem_feature_ablation.py`
- `stockmem/scripts/export_stockmem_report_tables.py`
- `stockmem/scripts/run_submission_reproduction.py`

Experimental scripts for hybrid reranking and head-aligned retriever training are retained for appendix discussion.
