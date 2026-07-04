# Strict Test: Learned Retriever vs Fixed-kNN

- Data source: `data/exports/stockmem_records_eth.ndjson`
- Test split: `2025-07-01` to `2026-05-01`
- Label threshold: `±2.00%` on `future_return_7d`

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate | Avg conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_knn_rolling_stable | 305 | 0.3180 | 0.4498 | 0.7508 | 0.8656 | 0.4426 | 0.2492 | 0.3082 | 0.7124 |
| fixed_retriever_learned_head | 305 | 0.3443 | 0.4661 | 0.8230 | 0.8656 | 0.4492 | 0.1770 | 0.3738 | 0.7250 |
| learned_retriever_fixed_head | 305 | 0.3803 | 0.4792 | 0.7869 | 0.8820 | 0.5574 | 0.2131 | 0.2295 | 0.7220 |
| learned_finbert_rolling_stable | 305 | 0.4098 | 0.5020 | 0.8361 | 0.8820 | 0.5148 | 0.1639 | 0.3213 | 0.7337 |

## Paired Comparison

Primary pair: `fixed_knn_rolling_stable` vs `fixed_retriever_learned_head`

| Metric | Delta learned-fixed | 95% bootstrap CI |
| --- | ---: | ---: |
| overall_acc | +0.0264 | [-0.0066, +0.0623] |
| active_acc | +0.0165 | [-0.0098, +0.0436] |
| coverage | +0.0717 | [+0.0328, +0.1115] |
| hit_at_5_same_sign | +0.0000 | [+0.0000, +0.0000] |

- McNemar exact test p-value: `0.168638`
- Discordant pairs: `26`
- Fixed-only correct: `9`
- Learned-only correct: `17`

Secondary pair: `fixed_knn_rolling_stable` vs `learned_finbert_rolling_stable`

- overall_acc delta: `+0.0918` (95% CI [+0.0230, +0.1607])
- McNemar exact p-value: `0.012603`
