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

StockMem runtime search supports fixed weighted kNN and learned diagonal metric
retrieval. The maintained evidence-retrieval model for the research report is
the offline `learned_recency_50_50` consensus retriever documented in
`docs/stockmem/trend_aware_retrieval.md`; its maintained decision head is
`count_vote_buy3_sell4` over top-10 evidence. These report models are evaluated
from the frozen NDJSON export for reproducibility.

For multi-asset reporting, model profiles are separated by symbol in:

```text
stockmem/config/model_profiles.json
```

The intended deployment shape is one StockMem decision profile per asset:

| Symbol | Profile status | Learned retriever artifact |
| --- | --- | --- |
| `BTC` | maintained | `stockmem/config/learned_retriever_finbert.json` |
| `ETH` | prepared for fine-tune | `stockmem/config/learned_retriever_finbert.eth.json` |

## Other Services

- Crawler: article collection and news enrichment.
- MarketData: candle/history/indicator snapshots.
- FactorLedge: factor normalization and summaries.
- Frontend: dashboard consuming MainController and forecast endpoints.
