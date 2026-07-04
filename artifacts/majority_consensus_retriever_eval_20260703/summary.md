# Majority-Consensus Retriever Evaluation

- Data: `data/exports/stockmem_records.ndjson`
- Top-k: `10`
- Label threshold: `±2.00%` on `future_return_7d`
- Min pool size: `10`
- Full-history date range: `2018-01-01` to `2026-06-08`

## val

| Model | n | Hit@10 | Majority@10 | Mean Same | Weighted Same | BUY Maj | HOLD Maj | SELL Maj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 174 | 0.9483 | 0.4368 | 4.0172 | 0.4017 | 0.7273 | 0.0000 | 0.2105 |
| `learned_only` | 174 | 0.9310 | 0.5057 | 4.2241 | 0.4224 | 0.8068 | 0.0000 | 0.2982 |
| `recency_only` | 174 | 0.8736 | 0.6264 | 6.2644 | 0.6264 | 0.7273 | 0.1379 | 0.7193 |
| `learned_recency_50_50` | 174 | 0.8621 | 0.5920 | 5.7874 | 0.5787 | 0.6818 | 0.0690 | 0.7193 |
| `fixed_recency_50_50` | 174 | 0.8851 | 0.6264 | 6.0230 | 0.6023 | 0.7386 | 0.0690 | 0.7368 |
| `unconstrained` | 174 | 0.8908 | 0.6379 | 6.2644 | 0.6264 | 0.7386 | 0.1379 | 0.7368 |
| `memory_first_learned030` | 174 | 0.8678 | 0.5977 | 5.5977 | 0.5598 | 0.6932 | 0.0345 | 0.7368 |
| `memory_first_learned020_fine` | 174 | 0.9023 | 0.6207 | 5.4943 | 0.5494 | 0.7386 | 0.0345 | 0.7368 |
| `memory_first_min_memory050` | 174 | 0.8851 | 0.6207 | 5.7931 | 0.5793 | 0.7273 | 0.0690 | 0.7368 |

## test

| Model | n | Hit@10 | Majority@10 | Mean Same | Weighted Same | BUY Maj | HOLD Maj | SELL Maj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 305 | 0.9246 | 0.3639 | 3.6852 | 0.3685 | 0.6735 | 0.0172 | 0.2953 |
| `learned_only` | 305 | 0.9508 | 0.3541 | 3.7377 | 0.3738 | 0.5306 | 0.0000 | 0.3758 |
| `recency_only` | 305 | 0.8131 | 0.5180 | 5.3016 | 0.5302 | 0.5510 | 0.1034 | 0.6577 |
| `learned_recency_50_50` | 305 | 0.8361 | 0.5443 | 5.2754 | 0.5275 | 0.5816 | 0.1379 | 0.6779 |
| `fixed_recency_50_50` | 305 | 0.8459 | 0.5180 | 5.1279 | 0.5128 | 0.5408 | 0.0517 | 0.6846 |
| `unconstrained` | 305 | 0.8230 | 0.5246 | 5.2984 | 0.5298 | 0.5408 | 0.0862 | 0.6846 |
| `memory_first_learned030` | 305 | 0.8459 | 0.5279 | 5.0885 | 0.5089 | 0.5918 | 0.0517 | 0.6711 |
| `memory_first_learned020_fine` | 305 | 0.8984 | 0.5180 | 4.8590 | 0.4859 | 0.5612 | 0.0517 | 0.6711 |
| `memory_first_min_memory050` | 305 | 0.8689 | 0.5115 | 4.8918 | 0.4892 | 0.5408 | 0.0345 | 0.6779 |

## full_2018_now

| Model | n | Hit@10 | Majority@10 | Mean Same | Weighted Same | BUY Maj | HOLD Maj | SELL Maj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 2871 | 0.9620 | 0.3817 | 3.9032 | 0.3903 | 0.5389 | 0.0435 | 0.3508 |
| `learned_only` | 2871 | 0.9603 | 0.3755 | 3.8802 | 0.3880 | 0.5019 | 0.0435 | 0.3790 |
| `recency_only` | 2871 | 0.8384 | 0.5075 | 5.0665 | 0.5067 | 0.5966 | 0.2016 | 0.5441 |
| `learned_recency_50_50` | 2871 | 0.8687 | 0.5106 | 4.9721 | 0.4972 | 0.5989 | 0.1917 | 0.5544 |
| `fixed_recency_50_50` | 2871 | 0.8788 | 0.5037 | 4.9220 | 0.4922 | 0.5989 | 0.1621 | 0.5497 |
| `unconstrained` | 2871 | 0.8541 | 0.5092 | 5.0596 | 0.5060 | 0.5989 | 0.1976 | 0.5478 |
| `memory_first_learned030` | 2871 | 0.8868 | 0.4967 | 4.8276 | 0.4828 | 0.5958 | 0.1522 | 0.5394 |
| `memory_first_learned020_fine` | 2871 | 0.9168 | 0.4842 | 4.7029 | 0.4703 | 0.5958 | 0.1166 | 0.5225 |
| `memory_first_min_memory050` | 2871 | 0.9025 | 0.4869 | 4.7600 | 0.4760 | 0.5920 | 0.1206 | 0.5328 |

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
  },
  "unconstrained": {
    "w_fixed": 0.1,
    "w_learned": 0.0,
    "w_recency": 0.8,
    "w_regime": 0.1,
    "recency_half_life_days": 21.0
  },
  "memory_first_learned030": {
    "w_fixed": 0.3,
    "w_learned": 0.3,
    "w_recency": 0.4,
    "w_regime": 0.0,
    "recency_half_life_days": 30.0
  },
  "memory_first_learned020_fine": {
    "w_fixed": 0.4,
    "w_learned": 0.2,
    "w_recency": 0.4,
    "w_regime": 0.0,
    "recency_half_life_days": 14.0
  },
  "memory_first_min_memory050": {
    "w_fixed": 0.6,
    "w_learned": 0.0,
    "w_recency": 0.4,
    "w_regime": 0.0,
    "recency_half_life_days": 21.0
  }
}
```
