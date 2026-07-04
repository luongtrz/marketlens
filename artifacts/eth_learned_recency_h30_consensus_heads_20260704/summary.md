# Consensus Retriever Decision Head Evaluation

- Data: `data/exports/stockmem_records_eth.ndjson`
- Retriever config: `stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json`
- Top-k: `10`
- Label threshold: `±2.00%`

## Selected Head

- Head: `mean_learned_weights_buy0.50_sell0.75`
- Validation score: `0.7070`

## Comparison

| Model | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 | Mean Same@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.75` | val | 174 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.0714 | 0.8817 | 0.5575 | 5.1724 |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.75` | test | 305 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.0526 | 0.6496 | 0.5246 | 5.3803 |
| `fixed_knn_rolling_stable` | test_old_strict | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6327 | 0.2759 | 0.1275 | n/a | n/a |
| `fixed_retriever_learned_head` | test_old_strict | 305 | 0.3508 | 0.4500 | 0.8525 | 0.7041 | 0.1552 | 0.1946 | n/a | n/a |
| `learned_finbert_rolling_stable` | test_old_strict | 305 | 0.3410 | 0.4393 | 0.7836 | 0.5408 | 0.2414 | 0.2483 | n/a | n/a |

## Top Validation Heads

| Rank | Head | Score | Val Overall | Val Active | Val Coverage | Val BUY DA | Val SELL DA | Test Overall | Test Active | Test Coverage | Test BUY DA | Test SELL DA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mean_learned_weights_buy0.50_sell0.75` | 0.7070 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.8817 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.6496 |
| 2 | `mean_learned_weights_buy0.75_sell0.75` | 0.7070 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.8817 | 0.5902 | 0.6807 | 0.9344 | 0.6846 | 0.6496 |
| 3 | `mean_learned_weights_buy1.00_sell0.75` | 0.7070 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.8817 | 0.5869 | 0.6820 | 0.9279 | 0.6769 | 0.6496 |
| 4 | `mean_learned_weights_buy1.25_sell0.75` | 0.7070 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.8817 | 0.5902 | 0.6882 | 0.9148 | 0.6769 | 0.6496 |
| 5 | `mean_learned_weights_buy0.50_sell0.50` | 0.7069 | 0.6494 | 0.7396 | 0.9713 | 0.5472 | 0.8817 | 0.5967 | 0.6804 | 0.9541 | 0.7077 | 0.6496 |
| 6 | `mean_learned_weights_buy0.75_sell0.50` | 0.7069 | 0.6494 | 0.7396 | 0.9713 | 0.5472 | 0.8817 | 0.5869 | 0.6818 | 0.9377 | 0.6846 | 0.6496 |
| 7 | `mean_learned_weights_buy1.00_sell0.50` | 0.7069 | 0.6494 | 0.7396 | 0.9713 | 0.5472 | 0.8817 | 0.5836 | 0.6831 | 0.9311 | 0.6769 | 0.6496 |
| 8 | `mean_learned_weights_buy1.25_sell0.50` | 0.7069 | 0.6494 | 0.7396 | 0.9713 | 0.5472 | 0.8817 | 0.5869 | 0.6893 | 0.9180 | 0.6769 | 0.6496 |
| 9 | `mean_learned_weights_buy0.50_sell1.00` | 0.7026 | 0.6437 | 0.7607 | 0.9368 | 0.5472 | 0.8710 | 0.5934 | 0.6807 | 0.9344 | 0.7077 | 0.6277 |
| 10 | `mean_learned_weights_buy0.75_sell1.00` | 0.7026 | 0.6437 | 0.7607 | 0.9368 | 0.5472 | 0.8710 | 0.5836 | 0.6821 | 0.9180 | 0.6846 | 0.6277 |
| 11 | `mean_learned_weights_buy1.00_sell1.00` | 0.7026 | 0.6437 | 0.7607 | 0.9368 | 0.5472 | 0.8710 | 0.5803 | 0.6835 | 0.9115 | 0.6769 | 0.6277 |
| 12 | `mean_learned_weights_buy1.25_sell1.00` | 0.7026 | 0.6437 | 0.7607 | 0.9368 | 0.5472 | 0.8710 | 0.5836 | 0.6898 | 0.8984 | 0.6769 | 0.6277 |
| 13 | `mean_learned_weights_buy1.50_sell0.75` | 0.7022 | 0.6437 | 0.7515 | 0.9483 | 0.5283 | 0.8817 | 0.5836 | 0.6934 | 0.8984 | 0.6615 | 0.6496 |
| 14 | `mean_learned_weights_buy1.50_sell0.50` | 0.7021 | 0.6437 | 0.7381 | 0.9655 | 0.5283 | 0.8817 | 0.5803 | 0.6945 | 0.9016 | 0.6615 | 0.6496 |
| 15 | `mean_learned_weights_buy0.50_sell1.25` | 0.6981 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5869 | 0.6784 | 0.9279 | 0.7077 | 0.6131 |
| 16 | `mean_learned_weights_buy0.50_sell1.50` | 0.6981 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5902 | 0.6797 | 0.9213 | 0.7077 | 0.6131 |
| 17 | `mean_learned_weights_buy0.50_sell1.75` | 0.6981 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5869 | 0.6835 | 0.9115 | 0.7077 | 0.6058 |
| 18 | `mean_learned_weights_buy0.75_sell1.25` | 0.6981 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5770 | 0.6799 | 0.9115 | 0.6846 | 0.6131 |
| 19 | `mean_learned_weights_buy0.75_sell1.50` | 0.6981 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5803 | 0.6812 | 0.9049 | 0.6846 | 0.6131 |
| 20 | `mean_learned_weights_buy0.75_sell1.75` | 0.6981 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5770 | 0.6850 | 0.8951 | 0.6846 | 0.6058 |
