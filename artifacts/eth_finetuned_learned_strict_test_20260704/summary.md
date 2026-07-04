# Strict Test: Learned Retriever vs Fixed-kNN

- Data source: `data/exports/stockmem_records_eth.ndjson`
- Test split: `2025-07-01` to `2026-05-01`
- Label threshold: `±2.00%` on `future_return_7d`

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate | Avg conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_knn_rolling_stable | 305 | 0.3180 | 0.4498 | 0.7508 | 0.8656 | 0.4426 | 0.2492 | 0.3082 | 0.7124 |
| fixed_retriever_learned_head | 305 | 0.3443 | 0.4661 | 0.8230 | 0.8656 | 0.4492 | 0.1770 | 0.3738 | 0.7250 |
| learned_retriever_fixed_head | 305 | 0.3738 | 0.4818 | 0.7213 | 0.8689 | 0.4885 | 0.2787 | 0.2328 | 0.7033 |
| learned_finbert_rolling_stable | 305 | 0.3902 | 0.4980 | 0.8164 | 0.8689 | 0.4787 | 0.1836 | 0.3377 | 0.7147 |

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

- overall_acc delta: `+0.0719` (95% CI [+0.0066, +0.1377])
- McNemar exact p-value: `0.038958`
