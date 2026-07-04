# StockMem Experiment Metrics Catalog

This file is the maintained numeric catalog for report writing and audit. It
lists the experiments already run, the source artifact for each table, the
evaluation split, and the main result.

Unless stated otherwise, BTC/ETH held-out tests use:

```text
test window: 2025-07-01 to 2026-05-01
test rows:   305
D7 label:    BUY if future_return_7d > +2%, SELL if < -2%, else HOLD
```

Metric policy:

```text
overall_DA = exact BUY/HOLD/SELL correctness over all evaluated rows
active_DA  = correctness among emitted BUY/SELL predictions
coverage   = share of rows with emitted BUY/SELL prediction
BUY_DA     = class accuracy on true BUY rows
HOLD_DA    = class accuracy on true HOLD rows
SELL_DA    = class accuracy on true SELL rows
```

Audit note: `calibrate_policy` was corrected so HOLD is counted as correct only
when the true D7 class is HOLD. See
`artifacts/audit_runs/stockmem_audit_20260702_232452/metric_audit.md`.

## 1. Maintained Winners

| Asset | Maintained retriever | Decision head | Split | n | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | `learned_recency_50_50` | `count_vote_buy3_sell4` | val | 174 | 0.6379 | 0.7658 | 0.9080 | 0.7614 | 0.0345 | 0.7544 | 0.5920 |
| BTC | `learned_recency_50_50` | `count_vote_buy3_sell4` | test | 305 | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.0000 | 0.7114 | 0.5443 |
| ETH | `eth_learned_recency_50_50_h30` | `mean_learned_weights_buy0.50_sell0.75` | val | 174 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.0714 | 0.8817 | 0.5575 |
| ETH | `eth_learned_recency_50_50_h30` | `mean_learned_weights_buy0.50_sell0.75` | test | 305 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.0526 | 0.6496 | 0.5246 |

Sources:

```text
artifacts/consensus_retriever_heads_20260703/summary.json
artifacts/eth_learned_recency_h30_consensus_heads_20260704/summary.json
stockmem/config/model_profiles.json
```

## 2. BTC Strict Structured Model Comparison

Source: `artifacts/learned_strict_test_v3/summary.json`.

| Model | n | Overall | Active | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 | 0.6295 | 0.2492 | 0.1213 |
| `fixed_retriever_learned_head` | 305 | 0.3508 | 0.4500 | 0.8525 | 0.8361 | 0.6721 | 0.1475 | 0.1803 |
| `learned_retriever_fixed_head` | 305 | 0.3148 | 0.4182 | 0.7213 | 0.8459 | 0.5541 | 0.2787 | 0.1672 |
| `learned_finbert_rolling_stable` | 305 | 0.3410 | 0.4393 | 0.7836 | 0.8459 | 0.5607 | 0.2164 | 0.2230 |

Paired statistics against `fixed_knn_rolling_stable`:

| Challenger | Metric | Delta | 95% bootstrap CI | McNemar p |
| --- | --- | ---: | --- | ---: |
| `fixed_retriever_learned_head` | overall | +0.0330 | `[+0.0000, +0.0689]` | 0.087159 |
| `fixed_retriever_learned_head` | active | +0.0264 | `[+0.0013, +0.0529]` | 0.087159 |
| `fixed_retriever_learned_head` | coverage | +0.1014 | `[+0.0623, +0.1410]` | 0.087159 |
| `learned_finbert_rolling_stable` | overall | +0.0222 | `[-0.0393, +0.0852]` | 0.550709 |

Interpretation: this older strict table showed the learned head helped more
than pure learned retrieval. It is superseded by the consensus-head pipeline for
the maintained endpoint.

## 3. BTC Naive LLM Baseline

Source: `artifacts/current_context_ai_eval/summary.json`.

| Model | n | Overall | Active | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_current_ai` | 305 | 0.2787 | 0.4031 | 0.6426 | 0.8361 | 0.6164 | 0.3574 | 0.0262 |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 | 0.6295 | 0.2492 | 0.1213 |
| `knn_returns` | 305 | 0.2918 | 0.4146 | 0.6721 | 0.8361 | 0.5574 | 0.3279 | 0.1148 |

Interpretation: the current-context-only LLM baseline is weak, especially on
SELL emission. This supports the claim that structured historical memory adds
value over naive prompting.

## 4. BTC Feature-Block Ablation

Source: `artifacts/fixed_knn_component_ablation/summary.json`.

| Variant | Overall | Active | Coverage | Hit@5 same sign |
| --- | ---: | ---: | ---: | ---: |
| `full_fixed_knn` | 0.3180 | 0.4236 | 0.7508 | 0.8361 |
| `no_factor_block` | 0.2918 | 0.4685 | 0.7279 | 0.8525 |
| `no_indicator_block` | 0.3115 | 0.4361 | 0.7443 | 0.8197 |
| `no_price_block` | 0.3443 | 0.4498 | 0.7508 | 0.8492 |
| `factor_only` | 0.3475 | 0.4789 | 0.8557 | 0.8459 |
| `indicator_only` | 0.3475 | 0.5144 | 0.6820 | 0.8557 |
| `price_only` | 0.3344 | 0.4656 | 0.8098 | 0.8754 |

Interpretation: individual blocks contain useful signals, but this ablation
does not prove every block is independently necessary.

## 5. BTC Evidence Retrieval Majority Audit

Source: `artifacts/majority_consensus_retriever_eval_20260703/summary.json`.

The stricter evidence target is:

```text
majority_same@10 = at least 5 of top-10 evidence records share the query D7 class
```

| Model | Val Majority@10 | Test Majority@10 | Full-history Majority@10 |
| --- | ---: | ---: | ---: |
| `fixed_only` | 0.4368 | 0.3639 | 0.3817 |
| `learned_only` | 0.5057 | 0.3541 | 0.3755 |
| `recency_only` | 0.6264 | 0.5180 | 0.5075 |
| `fixed_recency_50_50` | 0.6264 | 0.5180 | 0.5037 |
| `unconstrained` | 0.6379 | 0.5246 | 0.5092 |
| `memory_first_learned030` | 0.5977 | 0.5279 | 0.4967 |
| `memory_first_learned020_fine` | 0.6207 | 0.5180 | 0.4842 |
| `memory_first_min_memory050` | 0.6207 | 0.5115 | 0.4869 |
| `learned_recency_50_50` | 0.5920 | 0.5443 | 0.5106 |

Full-history BTC run:

```text
date range requested: 2018-01-01 to 2026-06-08
eligible rows:        2871
```

Interpretation: pure learned retrieval is not enough. The strongest maintained
evidence retriever is learned similarity plus recency.

## 6. BTC Consensus-Head Search

Source: `artifacts/consensus_retriever_heads_20260703/summary.json`.

| Head | Split | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `count_vote_buy3_sell4` | val | 0.6379 | 0.7658 | 0.9080 | 0.7614 | 0.0345 | 0.7544 | 0.5920 |
| `count_vote_buy3_sell4` | test | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.0000 | 0.7114 | 0.5443 |
| `mean_d7_only_buy0.50_sell0.50` | test | 0.5508 | 0.6840 | 0.9443 | 0.6122 | n/a | 0.7047 | n/a |

Interpretation: the official BTC endpoint keeps `count_vote_buy3_sell4` because
it was selected on validation and gives slightly stronger SELL DA.

## 7. BTC Hybrid Reranking

Source: `artifacts/hybrid_retrieval_frozen_v2/d7_consistency_eval.md`.

Setup:

```text
top_k: 5
candidate_pool_size: 30
hybrid weights: w_knn=0.6, w_learned=0.4, w_prior=0.0, w_regime=0.0
```

| Method | Top5 same D7 sign | nDCG@5 | Downstream DA | Active | Coverage | Evidence coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn` | 0.9312 | 0.3011 | 0.2820 | 0.3320 | 0.8000 | 1.0000 |
| `learned_only` | 0.9109 | 0.2932 | 0.3377 | 0.3782 | 0.7803 | 1.0000 |
| `hybrid_reranker` | 0.9069 | 0.3004 | 0.3377 | 0.3775 | 0.8164 | 1.0000 |
| `fixed_knn_production_head` | 0.9312 | 0.3011 | 0.2852 | 0.3681 | 0.5344 | 1.0000 |

Interpretation: hybrid reranking was useful diagnostically but did not beat the
later learned-recency consensus evidence pipeline.

## 8. BTC Memory-First / Trend-Aware Variants

Sources:

```text
artifacts/memory_first_retriever_20260703/unconstrained/summary.json
artifacts/memory_first_retriever_20260703/min_learned_030/summary.json
artifacts/memory_first_retriever_20260703/min_learned_020_fine/summary.json
artifacts/memory_first_retriever_20260703/historical_exclude_30/summary.json
artifacts/memory_first_retriever_20260703/historical_exclude_90/summary.json
```

Representative BTC test results:

| Variant group | Retriever | Test Hit@10 | Test Majority@10 | Mean Same@10 | BUY majority | HOLD majority | SELL majority |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unconstrained | `learned_recency_50_50` | 0.8361 | 0.5443 | 5.2754 | 0.5816 | 0.1379 | 0.6779 |
| unconstrained | `memory_first_learned_recency_60_30_10` | 0.8590 | 0.5082 | 4.8361 | 0.5510 | 0.0517 | 0.6577 |
| unconstrained | `balanced_fixed_learned_recency_regime` | 0.8951 | 0.4918 | 4.5967 | 0.5408 | 0.0345 | 0.6376 |
| historical exclude 30d | `learned_recency_50_50` | 0.9508 | 0.3410 | 3.6426 | 0.5204 | 0.0000 | 0.3557 |
| historical exclude 90d | `learned_recency_50_50` | 0.9508 | 0.3705 | 3.6918 | 0.6122 | 0.0000 | 0.3557 |

Interpretation: excluding recent records makes evidence quality much worse.
Recent regime context is doing real work, so the final model should be
described as learned memory plus trend awareness, not as pure historical analog
matching.

## 9. ETH Zero-Shot

Source docs/artifacts:

```text
docs/stockmem/eth_zero_shot.md
artifacts/eth_zero_shot_learned_strict_test_20260704/summary.json
artifacts/eth_zero_shot_majority_consensus_20260704/summary.json
artifacts/eth_zero_shot_consensus_heads_20260704/summary.json
```

ETH data summary:

| Field | Value |
| --- | ---: |
| Raw rows | 2908 |
| Rows with matured D7 | 2903 |
| Date range | `2018-01-05` to `2026-07-01` |
| Validation rows | 174 |
| Test rows | 305 |

Strict zero-shot:

| Model | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Hit@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 0.3180 | 0.4498 | 0.7508 | 0.4077 | 0.1842 | 0.2701 | 0.8656 |
| `fixed_retriever_learned_head` | 0.3443 | 0.4661 | 0.8230 | 0.4154 | 0.0789 | 0.3504 | 0.8656 |
| `learned_retriever_fixed_head` | 0.3803 | 0.4792 | 0.7869 | 0.5769 | 0.2895 | 0.2190 | 0.8820 |
| `learned_finbert_rolling_stable` | 0.4098 | 0.5020 | 0.8361 | 0.5308 | 0.2368 | 0.3431 | 0.8820 |

Consensus-head zero-shot with BTC artifact:

| Profile | Split | Overall | Active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | val | 0.5805 | 0.6821 | 0.9943 | 0.5094 | 0.0000 | 0.7957 | 0.5000 |
| `learned_recency_50_50 + mean_learned_weights_buy0.50_sell0.50` | test | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |

## 10. ETH Fine-Tuning

Source docs/artifacts:

```text
docs/stockmem/eth_finetune.md
stockmem/config/learned_retriever_finbert.eth.json
stockmem/config/weights.eth.auto.json
artifacts/eth_finetuned_consensus_heads_20260704/summary.json
artifacts/eth_learned_recency_h30_consensus_heads_20260704/summary.json
artifacts/eth_fusion_tune_eth_weights_20260704/summary.json
```

ETH learned retriever training:

| Metric | Value |
| --- | ---: |
| Validation combined | 0.3045 |
| Validation hit@k | 0.9862 |
| Validation nDCG@k | 0.2909 |
| Seed std | 0.0108 |

ETH fixed-kNN tuned weights:

| Block | Weight |
| --- | ---: |
| `factor_vec` | 0.3488 |
| `indicator_vec` | 0.2885 |
| `price_vec` | 0.3627 |

ETH candidate profiles:

| Candidate profile | Selected head | Val score | Test overall | Test active | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero-shot BTC artifact + h21 | `mean_learned_weights_buy0.50_sell0.50` | 0.6550 | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |
| ETH learned h21 | `mean_learned_weights_buy0.50_sell0.75` | 0.7067 | 0.6033 | 0.6815 | 0.9574 | 0.7308 | 0.0263 | 0.6423 | 0.5279 |
| ETH learned h30 pure | `mean_learned_weights_buy0.50_sell0.75` | 0.7070 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.0526 | 0.6496 | 0.5246 |
| ETH tuned fusion with BTC fixed weights | `mean_learned_weights_buy0.50_sell0.50` | 0.7025 | 0.5902 | 0.6782 | 0.9475 | 0.7000 | 0.0263 | 0.6423 | 0.5279 |
| ETH tuned fusion with ETH fixed weights | `mean_learned_weights_buy0.50_sell0.75` | 0.6984 | 0.5967 | 0.6678 | 0.9377 | 0.6923 | 0.1316 | 0.6350 | 0.5508 |

Maintained ETH endpoint:

```text
retriever: eth_learned_recency_50_50_h30
head:      mean_learned_weights_buy0.50_sell0.75
```

## 11. ETH Majority Retrieval Tuning

Source: `artifacts/eth_fusion_tune_eth_weights_20260704/summary.json`.

Validation-selected evidence config:

```json
{
  "w_fixed": 0.2,
  "w_learned": 0.3,
  "w_recency": 0.4,
  "w_regime": 0.1,
  "recency_half_life_days": 21.0
}
```

ETH test evidence results:

| Retriever | Hit@10 | Majority@10 | Mean Same@10 | BUY majority | HOLD majority | SELL majority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_only` | 0.9607 | 0.3410 | 3.7344 | 0.3615 | 0.0526 | 0.4015 |
| `learned_only` | 0.9738 | 0.2918 | 3.5607 | 0.3077 | 0.0000 | 0.3577 |
| `recency_only` | 0.9016 | 0.5410 | 5.4295 | 0.6308 | 0.1053 | 0.5766 |
| `learned_recency_50_50` | 0.9115 | 0.5279 | 5.3869 | 0.6231 | 0.0526 | 0.5693 |
| `memory_first_learned_recency_60_30_10` | 0.9279 | 0.5410 | 5.0066 | 0.6385 | 0.0000 | 0.5985 |
| `balanced_fixed_learned_recency_regime` | 0.9377 | 0.4754 | 4.6066 | 0.5308 | 0.0000 | 0.5547 |
| `selected_majority_consensus` | 0.9279 | 0.5508 | 5.2000 | 0.6538 | 0.0000 | 0.6058 |

Interpretation: the tuned ETH fusion gives the best evidence-only
`Majority@10`, but the maintained endpoint keeps the cleaner learned+recency
h30 profile because it has the strongest validation-selected decision-head
score while preserving the BTC architecture.

## 12. Corrected Policy Audit

Source: `artifacts/audit_runs/stockmem_audit_20260702_232452/metric_audit.md`.

| Metric | Value |
| --- | ---: |
| DA | 0.354098 |
| BUY DA | 0.427536 |
| SELL DA | 0.619048 |
| Coverage | 0.727869 |
| Sharpe | -0.126956 |
| Sortino | -0.592257 |
| Max drawdown | -0.991154 |
| Brier | 0.840929 |
| ECE | 0.226282 |
| n | 305 |

Interpretation: this was a metric-consistency audit, not the maintained final
model. It is useful for defending HOLD policy and probability calibration.

## 13. Older Cross-Model Trading/ML Reference

Source: `artifacts/metrics/main_table.csv`.

These rows are kept for historical comparison only. They do not define the
maintained StockMem endpoint.

| Model | DA | Balanced acc | Macro F1 | MCC | Coverage | Sharpe | Hit@5 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_fixed_knn` | 0.393443 | 0.430580 | 0.294242 | 0.024068 | 0.806557 | -0.296973 | 0.928826 | 0.320392 |
| `learned_cem_rag` | 0.396721 | 0.404839 | 0.309037 | 0.035539 | 0.783607 | -0.143290 | 0.953737 | 0.307147 |
| `random_forest` | 0.524590 | 0.511623 | 0.327283 | 0.181619 | 0.927869 | 0.704047 | 0.000000 | 0.255848 |
| `patchtst` | 0.576687 | 0.526666 | 0.486866 | 0.076022 | 0.534426 | 0.416554 | 0.000000 | 0.296921 |

Selected paired tests from `artifacts/metrics/stat_tests.json`:

| Comparison | n | DA A | DA B | McNemar p | Bootstrap DA delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn vs random_forest` | 305 | 0.4372 | 0.5548 | 0.000012 | +0.161716 |
| `buy_and_hold vs random_forest` | 305 | 0.4066 | 0.5548 | 0.000002 | +0.108197 |
| `fixed_knn vs patchtst` | 305 | 0.4372 | 0.5767 | 0.294814 | -0.050542 |

Interpretation: Random Forest and PatchTST were useful reference baselines, but
they are not the current StockMem retrieval architecture.

## 14. Artifact Inventory

Primary maintained docs:

```text
docs/stockmem/methodology.md
docs/stockmem/experiments.md
docs/stockmem/trend_aware_retrieval.md
docs/stockmem/multi_asset_stockmem_report.md
docs/stockmem/eth_zero_shot.md
docs/stockmem/eth_finetune.md
docs/stockmem/reproducibility.md
```

Primary result artifacts:

```text
artifacts/current_context_ai_eval/summary.json
artifacts/fixed_knn_component_ablation/summary.json
artifacts/learned_strict_test_v3/summary.json
artifacts/hybrid_retrieval_frozen_v2/d7_consistency_eval.md
artifacts/majority_consensus_retriever_eval_20260703/summary.json
artifacts/consensus_retriever_heads_20260703/summary.json
artifacts/eth_zero_shot_learned_strict_test_20260704/summary.json
artifacts/eth_zero_shot_majority_consensus_20260704/summary.json
artifacts/eth_zero_shot_consensus_heads_20260704/summary.json
artifacts/eth_finetuned_majority_consensus_20260704/summary.json
artifacts/eth_finetuned_consensus_heads_20260704/summary.json
artifacts/eth_learned_recency_h30_consensus_heads_20260704/summary.json
artifacts/eth_fusion_tune_eth_weights_20260704/summary.json
```

Primary model/config artifacts:

```text
stockmem/config/model_profiles.json
stockmem/config/learned_retriever_finbert.json
stockmem/config/learned_retriever_finbert.eth.json
stockmem/config/weights.auto.json
stockmem/config/weights.eth.auto.json
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json
stockmem/config/majority_consensus_retriever.eth.tuned_eth_weights.json
```

Archived/exploratory artifacts remain useful for audit but should not be cited
as the maintained result unless a report explicitly says so.
