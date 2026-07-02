# MarketLens

MarketLens is a crypto market intelligence system with news ingestion,
market-data enrichment, StockMem historical retrieval, and AI-assisted
prediction tooling. This branch is organized for a production-ready
graduation-project submission: app code is kept, generated artifacts are
ignored, and the maintained docs are grouped by domain.

## Start Here

- [Documentation map](docs/README.md)
- [System architecture](docs/system/architecture.md)
- [StockMem methodology](docs/stockmem/methodology.md)
- [StockMem experiments](docs/stockmem/experiments.md)
- [Reproducibility](docs/stockmem/reproducibility.md)

## Official StockMem Reproduction

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

If Groq quota is unavailable, use `--skip-llm` to reproduce the structured
StockMem tables and reuse the existing local LLM summary.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The Docker image used during the experiments did not include `pytest`; use a
development environment with the `dev` extras for the full test suite.

Generated folders such as `artifacts/`, `results_tables/`, `submission/`, and
local dataset exports are intentionally ignored. To reproduce StockMem metrics,
place the NDJSON export at `data/exports/stockmem_records.ndjson` or pass a
custom `--dataset` path.
