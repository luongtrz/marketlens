# Consensus Retriever Decision Head Evaluation

- Data: `data/exports/stockmem_records_eth.ndjson`
- Retriever config: `stockmem/config/majority_consensus_retriever.learned_recency_50_50.json`
- Top-k: `10`
- Label threshold: `±2.00%`

## Selected Head

- Head: `mean_learned_weights_buy0.50_sell0.50`
- Validation score: `0.6550`

## Comparison

| Model | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 | Mean Same@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | val | 174 | 0.5805 | 0.6821 | 0.9943 | 0.5094 | 0.0000 | 0.7957 | 0.5000 | 5.3851 |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | test | 305 | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 | 4.9672 |
| `fixed_knn_rolling_stable` | test_old_strict | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6327 | 0.2759 | 0.1275 | n/a | n/a |
| `fixed_retriever_learned_head` | test_old_strict | 305 | 0.3508 | 0.4500 | 0.8525 | 0.7041 | 0.1552 | 0.1946 | n/a | n/a |
| `learned_finbert_rolling_stable` | test_old_strict | 305 | 0.3410 | 0.4393 | 0.7836 | 0.5408 | 0.2414 | 0.2483 | n/a | n/a |

## Top Validation Heads

| Rank | Head | Score | Val Overall | Val Active | Val Coverage | Val BUY DA | Val SELL DA | Test Overall | Test Active | Test Coverage | Test BUY DA | Test SELL DA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mean_learned_weights_buy0.50_sell0.50` | 0.6550 | 0.5805 | 0.6821 | 0.9943 | 0.5094 | 0.7957 | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.6204 |
| 2 | `mean_learned_weights_buy0.50_sell0.75` | 0.6550 | 0.5805 | 0.6821 | 0.9943 | 0.5094 | 0.7957 | 0.5377 | 0.6034 | 0.9672 | 0.5769 | 0.6204 |
| 3 | `mean_learned_weights_buy0.50_sell1.00` | 0.6550 | 0.5805 | 0.6821 | 0.9943 | 0.5094 | 0.7957 | 0.5344 | 0.6232 | 0.9311 | 0.5769 | 0.6131 |
| 4 | `mean_learned_weights_buy0.50_sell1.25` | 0.6550 | 0.5805 | 0.6821 | 0.9943 | 0.5094 | 0.7957 | 0.5311 | 0.6187 | 0.9115 | 0.5769 | 0.5912 |
| 5 | `mean_learned_weights_buy0.50_sell1.50` | 0.6550 | 0.5805 | 0.6821 | 0.9943 | 0.5094 | 0.7957 | 0.5279 | 0.6168 | 0.8984 | 0.5769 | 0.5766 |
| 6 | `mean_learned_weights_buy0.50_sell2.00` | 0.6527 | 0.5805 | 0.6784 | 0.9828 | 0.5094 | 0.7849 | 0.5246 | 0.6154 | 0.8951 | 0.5769 | 0.5693 |
| 7 | `mean_learned_weights_buy0.50_sell2.25` | 0.6526 | 0.5805 | 0.6824 | 0.9770 | 0.5094 | 0.7849 | 0.5049 | 0.6090 | 0.8721 | 0.5769 | 0.5255 |
| 8 | `mean_learned_weights_buy0.50_sell1.75` | 0.6504 | 0.5747 | 0.6802 | 0.9885 | 0.5094 | 0.7849 | 0.5246 | 0.6154 | 0.8951 | 0.5769 | 0.5693 |
| 9 | `mean_fixed_weights_buy0.50_sell0.50` | 0.6502 | 0.5747 | 0.6763 | 0.9943 | 0.4906 | 0.7957 | 0.5213 | 0.6021 | 0.9475 | 0.5538 | 0.6131 |
| 10 | `mean_fixed_weights_buy0.50_sell0.75` | 0.6502 | 0.5747 | 0.6763 | 0.9943 | 0.4906 | 0.7957 | 0.5213 | 0.6021 | 0.9475 | 0.5538 | 0.6131 |
| 11 | `mean_fixed_weights_buy0.50_sell1.25` | 0.6502 | 0.5805 | 0.6706 | 0.9770 | 0.4906 | 0.7849 | 0.5148 | 0.6029 | 0.9082 | 0.5538 | 0.5766 |
| 12 | `mean_learned_weights_buy0.75_sell0.50` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5344 | 0.5986 | 0.9639 | 0.5692 | 0.6204 |
| 13 | `mean_learned_weights_buy0.75_sell0.75` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5377 | 0.6007 | 0.9607 | 0.5692 | 0.6204 |
| 14 | `mean_learned_weights_buy0.75_sell1.00` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5344 | 0.6206 | 0.9246 | 0.5692 | 0.6131 |
| 15 | `mean_learned_weights_buy0.75_sell1.25` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5311 | 0.6159 | 0.9049 | 0.5692 | 0.5912 |
| 16 | `mean_learned_weights_buy0.75_sell1.50` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5279 | 0.6140 | 0.8918 | 0.5692 | 0.5766 |
| 17 | `mean_learned_weights_buy1.00_sell0.50` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5311 | 0.6098 | 0.9410 | 0.5615 | 0.6204 |
| 18 | `mean_learned_weights_buy1.00_sell0.75` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5344 | 0.6119 | 0.9377 | 0.5615 | 0.6204 |
| 19 | `mean_learned_weights_buy1.00_sell1.00` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5311 | 0.6327 | 0.9016 | 0.5615 | 0.6131 |
| 20 | `mean_learned_weights_buy1.00_sell1.25` | 0.6501 | 0.5747 | 0.6802 | 0.9885 | 0.4906 | 0.7957 | 0.5279 | 0.6283 | 0.8820 | 0.5615 | 0.5912 |
