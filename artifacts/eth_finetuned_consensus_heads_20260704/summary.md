# Consensus Retriever Decision Head Evaluation

- Data: `data/exports/stockmem_records_eth.ndjson`
- Retriever config: `stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50.json`
- Top-k: `10`
- Label threshold: `±2.00%`

## Selected Head

- Head: `mean_learned_weights_buy0.50_sell0.75`
- Validation score: `0.7067`

## Comparison

| Model | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 | Mean Same@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.75` | val | 174 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.0714 | 0.8817 | 0.5517 | 5.1609 |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.75` | test | 305 | 0.6033 | 0.6815 | 0.9574 | 0.7308 | 0.0263 | 0.6423 | 0.5279 | 5.3869 |
| `fixed_knn_rolling_stable` | test_old_strict | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6327 | 0.2759 | 0.1275 | n/a | n/a |
| `fixed_retriever_learned_head` | test_old_strict | 305 | 0.3508 | 0.4500 | 0.8525 | 0.7041 | 0.1552 | 0.1946 | n/a | n/a |
| `learned_finbert_rolling_stable` | test_old_strict | 305 | 0.3410 | 0.4393 | 0.7836 | 0.5408 | 0.2414 | 0.2483 | n/a | n/a |

## Top Validation Heads

| Rank | Head | Score | Val Overall | Val Active | Val Coverage | Val BUY DA | Val SELL DA | Test Overall | Test Active | Test Coverage | Test BUY DA | Test SELL DA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mean_learned_weights_buy0.50_sell0.75` | 0.7067 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.8817 | 0.6033 | 0.6815 | 0.9574 | 0.7308 | 0.6423 |
| 2 | `mean_learned_weights_buy0.75_sell0.75` | 0.7067 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.8817 | 0.5869 | 0.6783 | 0.9377 | 0.6923 | 0.6423 |
| 3 | `mean_learned_weights_buy1.00_sell0.75` | 0.7067 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.8817 | 0.5836 | 0.6809 | 0.9246 | 0.6769 | 0.6423 |
| 4 | `mean_learned_weights_buy1.25_sell0.75` | 0.7067 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.8817 | 0.5869 | 0.6846 | 0.9148 | 0.6769 | 0.6423 |
| 5 | `mean_learned_weights_buy0.50_sell0.50` | 0.7066 | 0.6494 | 0.7440 | 0.9655 | 0.5472 | 0.8817 | 0.6000 | 0.6826 | 0.9607 | 0.7308 | 0.6423 |
| 6 | `mean_learned_weights_buy0.75_sell0.50` | 0.7066 | 0.6494 | 0.7440 | 0.9655 | 0.5472 | 0.8817 | 0.5836 | 0.6794 | 0.9410 | 0.6923 | 0.6423 |
| 7 | `mean_learned_weights_buy1.00_sell0.50` | 0.7066 | 0.6494 | 0.7440 | 0.9655 | 0.5472 | 0.8817 | 0.5803 | 0.6820 | 0.9279 | 0.6769 | 0.6423 |
| 8 | `mean_learned_weights_buy1.25_sell0.50` | 0.7066 | 0.6494 | 0.7440 | 0.9655 | 0.5472 | 0.8817 | 0.5836 | 0.6857 | 0.9180 | 0.6769 | 0.6423 |
| 9 | `mean_learned_weights_buy0.50_sell1.00` | 0.7023 | 0.6437 | 0.7607 | 0.9368 | 0.5472 | 0.8710 | 0.5967 | 0.6853 | 0.9377 | 0.7308 | 0.6204 |
| 10 | `mean_learned_weights_buy0.75_sell1.00` | 0.7023 | 0.6437 | 0.7607 | 0.9368 | 0.5472 | 0.8710 | 0.5803 | 0.6821 | 0.9180 | 0.6923 | 0.6204 |
| 11 | `mean_learned_weights_buy1.00_sell1.00` | 0.7023 | 0.6437 | 0.7607 | 0.9368 | 0.5472 | 0.8710 | 0.5770 | 0.6848 | 0.9049 | 0.6769 | 0.6204 |
| 12 | `mean_learned_weights_buy1.25_sell1.00` | 0.7023 | 0.6437 | 0.7607 | 0.9368 | 0.5472 | 0.8710 | 0.5803 | 0.6886 | 0.8951 | 0.6769 | 0.6204 |
| 13 | `mean_learned_weights_buy1.50_sell0.75` | 0.7019 | 0.6437 | 0.7515 | 0.9483 | 0.5283 | 0.8817 | 0.5803 | 0.6873 | 0.9016 | 0.6615 | 0.6423 |
| 14 | `mean_learned_weights_buy1.50_sell0.50` | 0.7018 | 0.6437 | 0.7425 | 0.9598 | 0.5283 | 0.8817 | 0.5770 | 0.6884 | 0.9049 | 0.6615 | 0.6423 |
| 15 | `mean_learned_weights_buy0.50_sell1.25` | 0.6978 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5934 | 0.6890 | 0.9279 | 0.7308 | 0.6131 |
| 16 | `mean_learned_weights_buy0.50_sell1.50` | 0.6978 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5934 | 0.6890 | 0.9279 | 0.7308 | 0.6131 |
| 17 | `mean_learned_weights_buy0.50_sell1.75` | 0.6978 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5902 | 0.6879 | 0.9246 | 0.7308 | 0.6058 |
| 18 | `mean_learned_weights_buy0.75_sell1.25` | 0.6978 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5770 | 0.6859 | 0.9082 | 0.6923 | 0.6131 |
| 19 | `mean_learned_weights_buy0.75_sell1.50` | 0.6978 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5770 | 0.6859 | 0.9082 | 0.6923 | 0.6131 |
| 20 | `mean_learned_weights_buy0.75_sell1.75` | 0.6978 | 0.6379 | 0.7593 | 0.9310 | 0.5472 | 0.8602 | 0.5738 | 0.6848 | 0.9049 | 0.6923 | 0.6058 |
