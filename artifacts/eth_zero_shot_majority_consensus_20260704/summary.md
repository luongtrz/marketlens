# Majority-Consensus Retriever Evaluation

- Data: `data/exports/stockmem_records_eth.ndjson`
- Top-k: `10`
- Label threshold: `±2.00%` on `future_return_7d`
- Min pool size: `10`
- Full-history date range: `2018-01-01` to `2026-07-01`

## val

| Model | n | Hit@10 | Majority@10 | Mean Same | Weighted Same | BUY Maj | HOLD Maj | SELL Maj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 174 | 0.9770 | 0.2586 | 3.5517 | 0.3552 | 0.0566 | 0.0357 | 0.4409 |
| `learned_only` | 174 | 0.9885 | 0.2931 | 3.7184 | 0.3718 | 0.1509 | 0.0357 | 0.4516 |
| `recency_only` | 174 | 0.8218 | 0.5517 | 5.1724 | 0.5172 | 0.3396 | 0.1786 | 0.7849 |
| `learned_recency_50_50` | 174 | 0.8563 | 0.5000 | 5.3851 | 0.5385 | 0.2830 | 0.0714 | 0.7527 |
| `fixed_recency_50_50` | 174 | 0.8276 | 0.5517 | 5.0172 | 0.5017 | 0.3019 | 0.1071 | 0.8280 |

## test

| Model | n | Hit@10 | Majority@10 | Mean Same | Weighted Same | BUY Maj | HOLD Maj | SELL Maj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 305 | 0.9770 | 0.2852 | 3.5934 | 0.3593 | 0.2923 | 0.0000 | 0.3577 |
| `learned_only` | 305 | 0.9902 | 0.3311 | 3.8066 | 0.3807 | 0.3615 | 0.0000 | 0.3942 |
| `recency_only` | 305 | 0.9016 | 0.5410 | 5.4295 | 0.5430 | 0.6308 | 0.1053 | 0.5766 |
| `learned_recency_50_50` | 305 | 0.8721 | 0.4754 | 4.9672 | 0.4967 | 0.5154 | 0.0526 | 0.5547 |
| `fixed_recency_50_50` | 305 | 0.9016 | 0.5410 | 5.3672 | 0.5367 | 0.6462 | 0.0526 | 0.5766 |

## full_2018_now

| Model | n | Hit@10 | Majority@10 | Mean Same | Weighted Same | BUY Maj | HOLD Maj | SELL Maj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 2889 | 0.9571 | 0.4150 | 4.0699 | 0.4070 | 0.4757 | 0.0526 | 0.4840 |
| `learned_only` | 2889 | 0.9585 | 0.4112 | 4.0564 | 0.4056 | 0.5181 | 0.0481 | 0.4287 |
| `recency_only` | 2889 | 0.8567 | 0.5292 | 5.1478 | 0.5148 | 0.6131 | 0.1465 | 0.5799 |
| `learned_recency_50_50` | 2889 | 0.8584 | 0.5112 | 5.1305 | 0.5130 | 0.5776 | 0.1465 | 0.5748 |
| `fixed_recency_50_50` | 2889 | 0.9048 | 0.5196 | 4.9387 | 0.4939 | 0.6023 | 0.1350 | 0.5722 |

## Configs

```json
{
  "fixed_only": {
    "w_fixed": 1.0,
    "w_learned": 0.0,
    "w_recency": 0.0,
    "w_regime": 0.0,
    "recency_half_life_days": 21.0
  },
  "learned_only": {
    "w_fixed": 0.0,
    "w_learned": 1.0,
    "w_recency": 0.0,
    "w_regime": 0.0,
    "recency_half_life_days": 21.0
  },
  "recency_only": {
    "w_fixed": 0.0,
    "w_learned": 0.0,
    "w_recency": 1.0,
    "w_regime": 0.0,
    "recency_half_life_days": 21.0
  },
  "learned_recency_50_50": {
    "w_fixed": 0.0,
    "w_learned": 0.5,
    "w_recency": 0.5,
    "w_regime": 0.0,
    "recency_half_life_days": 21.0
  },
  "fixed_recency_50_50": {
    "w_fixed": 0.5,
    "w_learned": 0.0,
    "w_recency": 0.5,
    "w_regime": 0.0,
    "recency_half_life_days": 21.0
  }
}
```
