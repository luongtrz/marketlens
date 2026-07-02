# Current-Context AI and Fixed-kNN Ablation Plan

**Date:** 2026-06-30  
**Project:** MarketLens  
**Purpose:** Replace the earlier retrieval-to-AI ablation with two experiments that match the actual pipeline.

## 1. Why the previous framing was wrong

The earlier AI ablation assumed the model was making decisions from retrieved historical cases. That is not the current production logic the user wants to defend.

The correct questions are:

1. Can a naive LLM using only current-day market and news context make useful predictions?
2. Inside the current fixed-kNN pipeline, does each major structured block contribute measurable value?

These are different questions and must be evaluated as different experiments.

## 2. Data split

Split logic comes from [stockmem/scripts/cem_dataset.py](/home/nmtc/projects/marketlens/stockmem/scripts/cem_dataset.py).

- `train`: `2018-01-05` to `2024-12-24` (`2363` rows)
- `val`: `2025-01-01` to `2025-06-23` (`174` rows)
- `test`: `2025-07-01` to `2026-05-01` (`305` rows)
- `embargo`: `43` rows

The official evaluation set for both experiments is the `305`-row test split.

## 3. Experiment 1: naive current-context AI baseline

Script:

- [aihub/scripts/evaluate_prediction_ablation.py](/home/nmtc/projects/marketlens/aihub/scripts/evaluate_prediction_ablation.py)

Helper module:

- [aihub/src/predict/ablation.py](/home/nmtc/projects/marketlens/aihub/src/predict/ablation.py)

Input to the LLM:

- latest candle
- recent `1d` and `3d` price change
- current indicators
- 1-day aggregated news sentiment
- compact headline bundle from `summary`

No historical retrieval is passed into the LLM in this experiment.

The script compares:

- `naive_current_ai`
- `fixed_knn_rolling_stable`
- `knn_returns`

Artifacts:

- `artifacts/current_context_ai_eval/naive_current_ai_test.jsonl`
- `artifacts/current_context_ai_eval/fixed_knn_test.jsonl`
- `artifacts/current_context_ai_eval/knn_returns_test.jsonl`
- `artifacts/current_context_ai_eval/summary.json`
- `artifacts/current_context_ai_eval/summary.md`

## 4. Experiment 2: fixed-kNN component ablation

Script:

- [stockmem/scripts/evaluate_knn_component_ablation.py](/home/nmtc/projects/marketlens/stockmem/scripts/evaluate_knn_component_ablation.py)

Shared NDJSON helpers:

- [stockmem/scripts/ndjson_eval_common.py](/home/nmtc/projects/marketlens/stockmem/scripts/ndjson_eval_common.py)

This experiment keeps the prediction head fixed and removes one structured block at a time from the fixed-kNN score.

Variants:

- `full_fixed_knn`
- `no_factor_block`
- `no_indicator_block`
- `no_price_block`
- `factor_only`
- `indicator_only`
- `price_only`

Artifacts:

- `artifacts/fixed_knn_component_ablation/*.jsonl`
- `artifacts/fixed_knn_component_ablation/summary.json`
- `artifacts/fixed_knn_component_ablation/summary.md`

## 5. Metrics

Both experiments report:

- `overall_acc`
- `active_acc`
- `coverage`
- `hit_at_5_same_sign`
- prediction distribution (`BUY`, `HOLD`, `SELL`)

Class labels use `future_return_7d` with a `±2%` threshold:

- `BUY` if `future_return_7d > +2%`
- `SELL` if `future_return_7d < -2%`
- `HOLD` otherwise

## 6. Docker runs

Experiment 1, naive current-context AI:

```bash
docker run -d --name current-context-ai-eval \
  --entrypoint /bin/sh \
  --env-file /home/nmtc/projects/marketlens/.env \
  -v /home/nmtc/projects/marketlens:/app \
  -w /app \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python aihub/scripts/evaluate_prediction_ablation.py --data data/exports/stockmem_records.ndjson --weights stockmem/config/weights.auto.json --out-dir artifacts/current_context_ai_eval --progress-every 10 > artifacts/current_context_ai_eval/run.log 2>&1"
```

Experiment 2, fixed-kNN component ablation:

```bash
docker run -d --name fixed-knn-component-ablation \
  --entrypoint /bin/sh \
  -v /home/nmtc/projects/marketlens:/app \
  -w /app \
  marketlens-aihub:latest \
  -lc "PYTHONPATH=/app python stockmem/scripts/evaluate_knn_component_ablation.py --data data/exports/stockmem_records.ndjson --weights stockmem/config/weights.auto.json --out-dir artifacts/fixed_knn_component_ablation --progress-every 25 > artifacts/fixed_knn_component_ablation/run.log 2>&1"
```

Monitor:

```bash
docker logs -f current-context-ai-eval
docker logs -f fixed-knn-component-ablation
```

## 7. Results

### Primary Strict-Test Table

Source artifacts:

- [artifacts/learned_strict_test_v3/summary.md](/home/nmtc/projects/marketlens/artifacts/learned_strict_test_v3/summary.md)
- [artifacts/learned_strict_test_v3/summary.json](/home/nmtc/projects/marketlens/artifacts/learned_strict_test_v3/summary.json)

This is the primary comparison table for the paper because it places all structured models on the same strict held-out `305`-row test split.

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 |
| `fixed_retriever_learned_head` | 305 | 0.3508 | 0.4500 | 0.8525 | 0.8361 |
| `learned_retriever_fixed_head` | 305 | 0.3148 | 0.4182 | 0.7213 | 0.8459 |
| `learned_finbert_rolling_stable` | 305 | 0.3410 | 0.4393 | 0.7836 | 0.8459 |

Primary takeaways:

- The strongest strict-test model is `fixed_retriever_learned_head`.
- A learned head helps more than a learned retriever in isolation on this protocol.
- The full learned stable pipeline still beats `fixed_knn_rolling_stable`, but it is not the best mechanism variant on this split.

Primary paired statistics against `fixed_knn_rolling_stable`:

- `fixed_retriever_learned_head` delta `overall_acc = +0.0330`, bootstrap 95% CI `[+0.0000, +0.0689]`
- `active_acc = +0.0264`, bootstrap 95% CI `[+0.0013, +0.0529]`
- `coverage = +0.1014`, bootstrap 95% CI `[+0.0623, +0.1410]`
- McNemar exact test: `p = 0.087159`

Secondary paired statistics for `learned_finbert_rolling_stable` vs `fixed_knn_rolling_stable`:

- `overall_acc` delta `+0.0222`, bootstrap 95% CI `[-0.0393, +0.0852]`
- McNemar exact test: `p = 0.550709`

### Experiment 1: naive AI vs tuned fixed-kNN head

Source artifacts:

- [artifacts/current_context_ai_eval/summary.md](/home/nmtc/projects/marketlens/artifacts/current_context_ai_eval/summary.md)
- [artifacts/current_context_ai_eval/summary.json](/home/nmtc/projects/marketlens/artifacts/current_context_ai_eval/summary.json)

Final metrics on the shared `305`-row test split:

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_current_ai` | 305 | 0.2787 | 0.4031 | 0.6426 | 0.8361 | 0.6164 | 0.3574 | 0.0262 |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 | 0.6295 | 0.2492 | 0.1213 |
| `knn_returns` | 305 | 0.2918 | 0.4146 | 0.6721 | 0.8361 | 0.5574 | 0.3279 | 0.1148 |

Observed result:

- The tuned fixed-kNN head is the strongest of the three on both `overall_acc` and `active_acc`.
- The naive current-context LLM underperforms the tuned fixed-kNN head by `3.93` points on `overall_acc` and `2.05` points on `active_acc`.
- The naive current-context LLM is heavily biased toward `BUY` and almost never emits `SELL` (`2.62%` sell rate on a test set whose actual `SELL` count is `149/305`).
- On this evaluation, naive current-context prompting is not a stronger replacement for the structured fixed-kNN pipeline.

Protocol caveat:

- The AI baseline is not yet paper-clean.
- The current artifact has been partially repaired so the first `31` dates are being rerun under the legacy prompt, but the rerun is not complete because Groq introduced a long rate-limit backoff during the repair pass.
- Until that finishes, the saved AI baseline should still be treated as operational evidence rather than a frozen paper-grade number.

### Experiment 2: learned strict-test comparison

Source artifacts:

- [artifacts/learned_strict_test/summary.md](/home/nmtc/projects/marketlens/artifacts/learned_strict_test/summary.md)
- [artifacts/learned_strict_test/summary.json](/home/nmtc/projects/marketlens/artifacts/learned_strict_test/summary.json)

Final metrics on the shared `305`-row test split:

| Model | n | Overall Acc | Active Acc | Coverage | Hit@5 same sign | BUY rate | HOLD rate | SELL rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 | 0.6295 | 0.2492 | 0.1213 |
| `learned_retriever_fixed_head` | 305 | 0.3148 | 0.4182 | 0.7213 | 0.8459 | 0.5541 | 0.2787 | 0.1672 |
| `learned_finbert_rolling_stable` | 305 | 0.3410 | 0.4393 | 0.7836 | 0.8459 | 0.5607 | 0.2164 | 0.2230 |

Observed result:

- Replacing only the retriever while keeping the fixed stable head does not materially improve strict classification accuracy.
- The full learned stable pipeline does improve the strict test result over fixed stable:
  - `+2.30` points on `overall_acc`
  - `+1.57` points on `active_acc`
  - `+3.28` points on `coverage`
  - `+0.98` points on `Hit@5 same sign`
- But the mechanism ablation shows an even stronger variant: `fixed_retriever_learned_head`.

### Experiment 3: fixed-kNN component ablation

Source artifacts:

- [artifacts/fixed_knn_component_ablation/summary.md](/home/nmtc/projects/marketlens/artifacts/fixed_knn_component_ablation/summary.md)
- [artifacts/fixed_knn_component_ablation/summary.json](/home/nmtc/projects/marketlens/artifacts/fixed_knn_component_ablation/summary.json)

Final metrics on the shared `305`-row test split:

| Variant | Overall Acc | Active Acc | Coverage | Hit@5 same sign | Delta overall | Delta active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_fixed_knn` | 0.3180 | 0.4236 | 0.7508 | 0.8361 | +0.0000 | +0.0000 |
| `no_factor_block` | 0.2918 | 0.4685 | 0.7279 | 0.8525 | -0.0262 | +0.0449 |
| `no_indicator_block` | 0.3115 | 0.4361 | 0.7443 | 0.8197 | -0.0066 | +0.0125 |
| `no_price_block` | 0.3443 | 0.4498 | 0.7508 | 0.8492 | +0.0262 | +0.0262 |
| `factor_only` | 0.3475 | 0.4789 | 0.8557 | 0.8459 | +0.0295 | +0.0553 |
| `indicator_only` | 0.3475 | 0.5144 | 0.6820 | 0.8557 | +0.0295 | +0.0908 |
| `price_only` | 0.3344 | 0.4656 | 0.8098 | 0.8754 | +0.0164 | +0.0420 |

Observed result:

- The current tuned full fixed-kNN configuration is not the accuracy-maximizing variant on this strict `BUY/HOLD/SELL` test protocol.
- Several reduced variants outperform `full_fixed_knn` on `overall_acc` and `active_acc`, especially `indicator_only`, `factor_only`, and `price_only`.
- That means the current production weighting is better defended as a tuned operational head from a different training objective, not as the uniquely best classifier under this specific test definition.
- The component-ablation result weakens any claim that all three retrieval blocks are individually necessary for this exact classification target.

## 8. Interpretation

Experiment 1 answers:

> Is a raw current-context LLM good enough without retrieval or structured memory?

Experiment 2 answers:

> Does learned retrieval help on the exact strict held-out test used for the AI comparison?

Experiment 3 answers:

> Does the current fixed-kNN pipeline actually depend on factor, indicator, and price blocks?

Current answer:

- For the first question, no: the naive LLM baseline does not beat the structured pipelines on the shared test split.
- For the second question, yes in the full-pipeline sense, but the main gain comes from the head more than from retriever substitution alone.
- For the third question, not in the simple monotonic sense originally hoped for: disabling or isolating blocks does not uniformly reduce accuracy under this test protocol.

That means the final written report should be careful:

- It can defend that the structured pipelines beat the naive current-context AI baseline on this dataset.
- It can defend that `fixed_retriever_learned_head` is the strongest structured model on the strict held-out test currently available.
- It can defend that the learned head is the main source of the strict-test gain.
- It should not claim that every existing fixed-kNN component is strictly necessary unless a cleaner ablation or a different target metric supports that statement.
