# Scripts Directory

The top-level `scripts/archive/` folder contains historical research,
backfill, and one-off experiment utilities. They are kept for auditability,
but they are not the official submission entrypoints.

Use the maintained StockMem reproduction command instead:

```bash
PYTHONPATH=. python stockmem/scripts/run_submission_reproduction.py \
  --dataset data/exports/stockmem_records.ndjson \
  --out-dir submission/stockmem_2026_07
```

For Docker usage and expected outputs, see
`docs/stockmem/reproducibility.md`.
