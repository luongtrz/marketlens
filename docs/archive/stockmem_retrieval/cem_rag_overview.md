# CEM-RAG — Tổng quan pipeline và trạng thái implementation

> **CEM-RAG**: Crypto Event Memory Retrieval-Augmented Forecasting  
> Paper target: ESWA (Expert Systems with Applications)

---

## Pipeline đầy đủ

```
Article (title, summary, factors)
        │
        ▼
[aihub] POST /events/extract          ← aihub/src/events/extractor.py
        │  rule-based → keyword → LLM (Flash)
        ▼
EventRecord[]
        │
        ▼
DailyEventState (per day/symbol)      ← stockmem/src/search/event_memory.py
        │  novelty_7d, source_diversity, dominant_groups
        ▼
StockMemRecord                        ← shared/models/memory.py
        │  event_state + event_vec (85d) + factor_vec (75d)
        │  + indicator_vec (5d) + price_vec (60d)
        │  + future_return d1/d3/d7/d15/d30
        ▼
StockMem DB (Supabase)
        │
        ▼
[search] kNN retrieval                ← stockmem/src/search/searcher.py
        │  retriever_type = "fixed_knn" | "learned_linear"
        │
        ├─ fixed_knn: w1·cos(factor) + w2·cos(indicator) + w3·cos(price)
        │             weights từ stockmem/config/weights.auto.json
        │
        └─ learned_linear: LearnedDiagonalMetric               ← stockmem/src/search/learned_metric.py
                           artifact: stockmem/config/learned_retriever.json
                           (InfoNCE + ridge, numpy Adam, 4 blocks = 225d)
        │
        ▼
Top-k similar historical cases
        │
        ▼
[probability] kNN probability         ← stockmem/scripts/calibrate_policy.py
        │  p_up, p_down, p_hold
        │  signal = BUY/SELL/HOLD theo tau từ policy.json
        ▼
CEMRAGPrediction                      ← shared/models/prediction.py
        │  p_up, p_down, p_hold, signal, confidence, explanation
        ▼
Output
```

---

## Trạng thái implementation (2026-06-23)

### ✅ Hoàn thành

| Component | File(s) | Ghi chú |
|---|---|---|
| EventRecord + DailyEventState schema | `shared/models/event.py` | |
| EventExtractor (3-tier) | `aihub/src/events/extractor.py` | Rule-based + keyword + Gemini Flash |
| POST /events/extract | `aihub/src/api.py` | |
| StockMemRecord với event_state + 5 horizons | `shared/models/memory.py` | d1/d3/d7/d15/d30 |
| event_vec embedder (85d) | `stockmem/src/search/event_memory.py` | 62 type + 13 group + 10 scalar |
| LearnedDiagonalMetric + score_batch | `stockmem/src/search/learned_metric.py` | Vectorized 1-vs-N |
| Train pipeline (InfoNCE + ridge) | `stockmem/scripts/train_learned_retriever.py` | Vectorized re-mining, 23× faster eval |
| Trained artifact | `stockmem/config/learned_retriever.json` | val_hit=0.9988, seed_std=0.0024 |
| Evaluation retriever | `stockmem/scripts/evaluate_retriever.py` | McNemar DA + hit@k, acceptance gates |
| Probability model + policy | `stockmem/scripts/calibrate_policy.py` | tau=0.22, test_DA=0.790 |
| CEMRAGPrediction schema | `shared/models/prediction.py` | |
| Eval suite (6 baselines) | `scripts/eval_suite.py` | Macro-F1, MCC, Sharpe, McNemar, bootstrap |
| 2885 records in Supabase | — | BTC 2021–2026, event_vec 99.7% populated |
| Eval artifacts | `artifacts/metrics/` | main_table.csv, stat_tests.json |

### ❌ Chưa có

| Component | Ưu tiên | Ghi chú |
|---|---|---|
| XGBoost/FinBERT/PatchTST baselines | High | Cần để so sánh đầy đủ ESWA |
| DB event extraction đầy đủ | High | `scripts/compute_event_states.py` sẵn sàng, cần local PG |
| Retrain retriever với event_state đầy đủ | High | Sau khi chạy compute_event_states |
| Transaction cost sensitivity | Medium | 5/10/20 bps |
| Parquet dataset artifacts | Medium | `artifacts/datasets/` |
| Predictions JSONL per baseline | Medium | `artifacts/predictions/` |
| Probability calibration v2 | Low | Platt/isotonic, hiện ECE=0.21 |
| Neural fusion model (v2/v3) | Low | Cần nhiều data hơn |
| Architecture figures | Low | `artifacts/figures/` |

---

## Key numbers (test set 2025-07 → 2026-05, 305 queries)

| Metric | Baseline kNN | Learned CEM-RAG | Delta |
|---|---|---|---|
| Directional Accuracy | 0.393 | 0.397 | +0.003 |
| hit@5 | 0.929 | 0.954 | **+0.025** |
| Sharpe (raw, no tau) | −0.297 | −0.143 | **+0.154** |
| DA với tau=0.22 | 0.777 | 0.790 | +0.013 |
| Sharpe với tau=0.22 | −0.140 | **+0.041** | **+0.181** |
| McNemar DA p-value | — | — | 1.00 (n.s.) |
| seed_std hit@5 | — | 0.0024 | ✅ < 0.03 |

**Framing cho paper:** Learned CEM-RAG cải thiện retrieval quality (hit@5 +2.5pp) và risk-adjusted return (Sharpe từ âm lên dương với calibrated policy). Statistical significance chưa đạt (test set nhỏ, 305 days). Contribution chính: framework + leakage correction + marginal improvement.

---

## Acceptance gates (theo CaiTien.md)

| Gate | Kết quả | Pass? |
|---|---|---|
| combined ≥ baseline + 0.01 | 0.2094 vs 0.1767 (+0.033) | ✅ |
| balanced DA +1pp | SELL_DA giảm 0.70→0.62 | ❌ |
| McNemar p < 0.10 | p=0.21 (hit@k) | ❌ |
| seed_std < 0.03 | 0.0024 | ✅ |
| val/test hit delta same sign | val+1.2pp, test+2.5pp | ✅ |

3/5 gates pass. Reportable với framing trung thực.

---

## Cách chạy toàn bộ pipeline research

```bash
cd /home/luong/marketlens
source .venv/bin/activate

# 1. (Cần DB) Populate event_state từ raw DB records
PYTHONPATH=. python scripts/compute_event_states.py

# 2. (Cần DB) Regen optimizer data với event_state đầy đủ
PYTHONPATH=. python stockmem/scripts/regen_optimizer_data.py \
  --output stockmem/data/real_optimizer_v2.json

# 3. Train learned retriever (nặng ~90 phút)
PYTHONPATH=. python stockmem/scripts/train_learned_retriever.py \
  --data stockmem/data/real_optimizer_v2.json \
  --output stockmem/config/learned_retriever.json \
  --trials 10 --epochs 40 --seeds 5

# 4. Calibrate policy (nhẹ ~60s)
PYTHONPATH=. python stockmem/scripts/calibrate_policy.py \
  --data stockmem/data/real_optimizer_v2.json \
  --artifact stockmem/config/learned_retriever.json

# 5. Evaluate so sánh (nhẹ ~30s)
PYTHONPATH=. python scripts/eval_suite.py

# 6. Evaluate retriever chi tiết + acceptance gates
PYTHONPATH=. python stockmem/scripts/evaluate_retriever.py \
  --data stockmem/data/real_optimizer_v2.json
```

---

## Tham khảo

- `docs/cem_rag_learned_retriever_VI.md` — Thiết kế learned retriever chi tiết
- `docs/event_extraction.md` — Event extraction module
- `docs/probability_model.md` — kNN probability + policy calibration
- `docs/eval_suite.md` — Evaluation suite chi tiết
- `docs/upgrade/CaiTien.md` — Roadmap đầy đủ ESWA
- `docs/upgrade/MoTa.md` — Định hướng paper
