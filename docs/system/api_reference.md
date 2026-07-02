# MarketLens API Reference

This is the compact API map for the maintained app services. Historical,
VPS-specific examples and hard-coded hostnames were archived under
`docs/archive/system/`.

## MainController

| Endpoint | Purpose |
| --- | --- |
| `POST /run` | Trigger one pipeline run for a symbol. |
| `GET /status/{run_id}` | Read pipeline status. |
| `GET /result/{run_id}` | Read assembled forecast result. |
| `POST /backfill` | Backfill historical records. |

## AIHub

| Endpoint | Purpose |
| --- | --- |
| `POST /sentiment` | Score text sentiment. |
| `POST /factors` | Extract structured market factors. |
| `POST /predict` | Produce a prediction from supplied context. |

The research LLM baseline does not depend on the live endpoint. It is run by
`aihub/scripts/evaluate_naive_llm_baseline.py` against the frozen StockMem
export.

## StockMem

| Endpoint | Purpose |
| --- | --- |
| `POST /record` | Persist a daily memory record. |
| `POST /search` | Retrieve similar historical records. |
| `GET /record/{record_id}` | Fetch a stored record by id. |

StockMem supports fixed weighted kNN and learned diagonal metric retrieval.
The official evaluation scripts use offline NDJSON data for reproducibility.

## Other Services

- Crawler: article collection and news enrichment.
- MarketData: candle/history/indicator snapshots.
- FactorLedge: factor normalization and summaries.
- Frontend: dashboard consuming MainController and forecast endpoints.
