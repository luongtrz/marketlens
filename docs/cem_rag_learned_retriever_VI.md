# CEM-RAG — Learned Event-Memory Retriever: Train & Cấu trúc Event

> Tài liệu mô tả chi tiết PR `feat(stockmem): add learned event-memory retriever`,
> tập trung vào **bên cấu trúc event** (event memory) và **bên train** (learned retriever).
> Phần fix kèm trong PR: đóng repo trong `stockmem/tests/test_store.py` để hết treo full test suite.

## 0. PR này làm gì (tóm tắt)

Thay **fixed weighted-cosine kNN** (3 trọng số `w1·factor + w2·indicator + w3·price` do Optuna chỉnh)
bằng một **learned retriever**: học một **metric tương đồng** từ *outcome* (hướng `future_return_7d`),
đồng thời bổ sung một **tầng event-memory point-in-time** (cấu trúc hoá tin tức thành event + đặc trưng
lan toả/novelty) làm **block đặc trưng thứ 4** cho retriever.

Định hướng học thuật (xem `docs/upgrade/CaiTien.md`, `MoTa.md`):
- **FinSeer** → retriever học bằng distillation từ "relevance" (ở đây thay reward LLM bằng outcome thực tế).
- **StockMem** → event memory + **Δinfo** (incremental information / độ lệch kỳ vọng).
- **FinGPT dissemination** → đặc trưng độ lan toả (số nguồn, đa dạng nguồn, novelty).

Tất cả thay đổi **cộng thêm, tương thích ngược**: mặc định `retriever_type="fixed_knn"` ⇒ hành vi cũ giữ nguyên.

---

## 1. Bên cấu trúc event (event memory)

### 1.1. Schema — `shared/models/event.py`

```python
class EventRecord(BaseModel):
    event_group: str          # nhóm sự kiện (13 nhóm trong taxonomy)
    event_type: str           # loại sự kiện (62 loại)
    entities: list[str]
    polarity: float           # [-1, 1]
    confidence: float         # [0, 1]
    observed_at: datetime | None
    description: str | None

class DailyEventState(BaseModel):
    date: date
    symbol: str
    events: list[EventRecord]
    article_count: int
    source_count: int
    source_diversity: float        # entropy chuẩn hoá theo nguồn
    temporal_span_hours: float     # độ trải thời gian của các bài trong ngày
    novelty_7d: float              # 1 - max Jaccard với 7 ngày trước
    novelty_30d: float             # 1 - max Jaccard với 30 ngày trước
    incremental_information: float # ≈ Δinfo của StockMem (novelty_30d × breadth)
    dominant_event_groups: list[str]
```

`StockMemRecord` (`stockmem/src/models.py`) được mở rộng các field:
`event_state: DailyEventState | None`, `event_vector: list[float]`,
`article_sources: list[str]`, `article_published_at: list[datetime]`.
`model_config` vẫn `extra="ignore"` nên các service khác không vỡ.

### 1.2. Xây event state & event vector — `stockmem/src/search/event_memory.py`

`build_daily_event_state(record, history)`:
1. **Trích event** từ `record.factors`: dùng taxonomy (`get_factor_type` / `get_factor_group`),
   dedup theo `(event_group, event_type)`; `polarity`/`confidence`/`observed_at` lấy từ
   `normalized_factors`.
2. **Lan toả (dissemination)**: `source_count` = số nguồn duy nhất (lowercase),
   `source_diversity` = entropy chuẩn hoá của danh sách nguồn,
   `temporal_span_hours` từ `article_published_at`.
3. **Novelty (StockMem-style)**:
   `novelty_w = 1 − max_{record ∈ [T−w, T−1]} Jaccard(current_types, historical_types)`
   cho `w ∈ {7, 30}` ngày — phân biệt "tin shock mới" vs "tin lặp lại".
4. **Δinfo**: `incremental_information = novelty_30d × breadth`, với
   `breadth = log1p(source_count) / log(21)`.

`build_event_vector(state) → 85 chiều` (`EVENT_DIM = 62 type + 13 group + 10 scalar`):
- 62 bit type multi-hot + 13 bit group multi-hot (giống biểu diễn nhị phân của StockMem).
- 10 scalar: `mean_polarity, max_abs_polarity, log(article_count), log(source_count),
  source_diversity, novelty_7d, novelty_30d, mean_confidence, temporal_span/168h,
  incremental_information` (đều được nén về ~[0,1]).

### 1.3. Event vào embedder — `stockmem/src/search/embedder.py`

`embed_split()` giờ trả **4 block** `SplitEmbedding(event_vec(85), factor_vec(75),
indicator_vec(5), price_vec(60))`, mỗi block **L2-normalize riêng**:
- `event_vec` = chuẩn hoá `record.event_vector` nếu đủ 85 chiều, ngược lại dựng từ `record.event_state`.

> **Lưu ý nhất quán train/serve:** `embed()` (joint 140d cho ANN index của fixed kNN) **không**
> chứa event — event chỉ phục vụ learned retriever. Learned path **quét toàn bộ** (không qua ANN
> prefilter) nên không lệch giữa offline và online.

---

## 2. Bên train (learned retriever)

### 2.1. Dataset point-in-time, chống rò rỉ — `stockmem/scripts/cem_dataset.py`

- **Split theo thời gian** (có embargo 7 ngày): Train ≤ `2024-12-24`, Val `2025-01-01..06-23`,
  Test `2025-07-01..2026-05-01`. Không shuffle.
- **Maturity guard (sửa rò rỉ then chốt):** ứng viên `c` chỉ dùng được cho query `q` khi
  `c.date + 7 ngày ≤ q.date` — so theo **ngày lịch** (`is_mature`), áp dụng cả khi tạo cặp huấn
  luyện lẫn khi đánh giá (`matured_pool`). Tránh dùng `future_return_7d` của ngày lịch sử **chưa
  đáo hạn** tại thời điểm dự báo.
- **Nhãn hướng** (`label_rows`): band `0.5·σ` với `σ` = độ lệch chuẩn **nhân quả** (chỉ từ các
  ngày đã đáo hạn trước đó, `_causal_sigma`); UP/DOWN/FLAT.
- **Teacher relevance** (thay reward LLM của FinSeer bằng outcome):
  `teacher_relevance = 0.45·outcome_sim + 0.35·regime_sim + 0.20·surface_sim`,
  chỉ > 0 khi ứng viên **cùng hướng** anchor.
- **mine_candidates**: positives = top theo teacher_relevance (mặc định 3);
  **hard negatives** = ngày **ngược hướng nhưng nhìn rất giống** (top theo cosine cũ rồi theo
  metric đang học); thêm vài **flat negatives**. Đây là điểm mấu chốt: học phân biệt
  *"same news, opposite outcome"*.

### 2.2. Metric & loss — `stockmem/scripts/train_learned_retriever.py` + `learned_metric.py`

- **Mô hình**: `LearnedDiagonalMetric` — diagonal per-feature `d ≥ 0` + `block_scales`,
  tính `score = Σ_block scale · cos(d_block ⊙ q, d_block ⊙ c)` (chuẩn hoá **theo từng block**).
  Khi `d = 1` ⇒ **trùng đúng** weighted-cosine kNN cũ (có test
  `test_identity_diagonal_reproduces_weighted_block_cosine`). Tự nhận **3 hoặc 4 block** theo data.
- **Loss = InfoNCE + distillation mềm + ridge**:
  - target = `softmax(positive_rewards / teacher_temperature)` (nhiều positive), hoặc one-hot nếu 1 positive;
  - `coefficients = probabilities − targets`, gradient closed-form qua Jacobian chuẩn hoá (numpy thuần, không torch);
  - `+ λ·‖d − 1‖²` kéo metric **về phía baseline** (vừa là chống overfit vừa là "lan can an toàn").
- **Huấn luyện**: Adam, clip `d ∈ [0.05, 20]`, `scales` chuẩn hoá tổng = 1, **re-mine mỗi 3 epoch**,
  **early stop theo val hit@5** (patience 10), **trung bình 5 seed** + báo `seed_std`.
- **Optuna** chỉnh `temperature, teacher_temperature, ridge, hard_negs, positive_count,
  learning_rate, k, band` — **chỉ trên validation**.
- **Artifact** `stockmem/config/learned_retriever.json`: `version=learned_cem_v2`, `block_dims`,
  `d`, `block_scales`, `band`, `splits`, `hyperparameters`, `val_hit_at_5`, `seed_std`, `mining_protocol`.

### 2.3. Đánh giá & cổng nghiệm thu — `stockmem/scripts/evaluate_retriever.py`

- So sánh **công bằng trên cùng 962 dòng / cùng split / cùng 7d / cùng maturity guard**:
  `baseline_fixed_guarded`, `baseline_fixed_leaky` (đo mức rò rỉ), `learned_diagonal`,
  cùng các **ablation**: `learned_factor_zeroed`, `learned_event_zeroed` (khi có event), `learned_fixed_band`.
- **Chỉ số retriever**: `hit@5` (cùng hướng, độc lập với `k` downstream), `hard_negative_gap`.
- **Chỉ số giao dịch**: tín hiệu BUY/SELL/HOLD theo ngưỡng (mặc định ±2%), `coverage` thực,
  `buy_da/sell_da/hold_da`, `Sharpe`, `combined = 0.6·DA + 0.4·Sharpe`.
- **Ý nghĩa thống kê**: McNemar (exact) + bootstrap CI (vector hoá).
- **Acceptance gate** (chốt trước khi xem test): combined ≥ baseline + 0.01; (buy_da+sell_da)/2 ≥ +1pp;
  McNemar p < 0.10; `seed_std` < 0.03; dấu delta val & test giống nhau.

### 2.4. Tích hợp serving (cộng thêm, tương thích ngược)

- `SearchRequest.retriever_type: "fixed_knn" | "learned_linear"` (mặc định `fixed_knn`).
- `RecordSearcher`: nhánh `learned_linear` **quét toàn bộ** cache (khớp offline), giữ nguyên
  **regime bonus ±0.15**, gắn `event_match` + `retriever_version` vào kết quả.
- `StockMemService.startup()` nạp artifact (`LEARNED_RETRIEVER_FILE`), `set_weights()` **vẫn truyền lại**
  learned metric (auto-retrain không làm mất). API `/search` truyền `retriever_type`.
- `optimize_weights.py`: thêm `maturity_guard` (mặc định True) xuyên suốt + cờ CLI
  `--maturity-guard/--no-maturity-guard`, ghi `evaluation_protocol.version = "maturity_guard_v1"`
  vào output; `weights_retrainer.py` (auto-retrain) dùng `maturity_guard=True` và ghi rõ.

---

## 3. Train ↔ Event kết nối ra sao

`event_vec` (mục 1) là **block thứ 4** mà learned metric (mục 2) học trọng số `d`/`scale` cho nó,
**cùng cơ chế** với factor/indicator/price. `cem_dataset.LabeledRow.blocks` tự thêm event block khi
`Row.event_vec` khác rỗng; `train_one_seed` tự suy `block_dims` (3 hoặc 4) từ data và khởi tạo
`scales = [0.05, 0.95·weights_cũ]` cho cấu hình 4 block. Nói cách khác: **cùng một trainer/searcher**
chạy được cả khi chưa có và khi đã có event — không phải viết lại.

---

## 4. Hiện trạng dữ liệu & tính trung thực

- Offline `stockmem/data/real_optimizer.json` (962 dòng BTC, 2023→2026): **88.6% `factor_vec` rỗng**
  và **chưa có `event_vec`**. `load_rows` tự **strip** event toàn-0 ⇒ hiện trainer chạy **3 block**.
- Theo quyết định: **đợi data event dày hơn** rồi mới chạy số liệu "headline". Hạ tầng (dataset builder,
  trainer, searcher, evaluator, artifact) đã sẵn để **re-run nguyên trạng** khi event/factor được
  làm dày (qua `regen_optimizer_data.py` / backfill để đổ `event_vec`).
- **Sửa rò rỉ maturity** và **so baseline công bằng** đã có giá trị **ngay bây giờ**, độc lập với việc
  retriever học có thắng fixed kNN hay không. Nếu chỉ ngang bằng, vẫn còn 3 đóng góp trung thực:
  (1) sửa look-ahead, (2) framework retriever học từ outcome (numpy), (3) parity + tính giải thích được.

---

## 5. Lệnh chạy

```bash
# 1) Dựng dataset point-in-time (nhãn + split + band)
python stockmem/scripts/build_cem_dataset.py --data stockmem/data/real_optimizer.json

# 2) Train learned retriever (Optuna + 5 seed) → stockmem/config/learned_retriever.json
python stockmem/scripts/train_learned_retriever.py \
  --data stockmem/data/real_optimizer.json --horizon 7d

# 3) Đánh giá fixed vs learned (guarded/leaky + ablation + McNemar + bootstrap)
python stockmem/scripts/evaluate_retriever.py \
  --data stockmem/data/real_optimizer.json \
  --artifact stockmem/config/learned_retriever.json

# 4) Bật learned ở serving
export LEARNED_RETRIEVER_FILE=stockmem/config/learned_retriever.json
# rồi gọi /search với {"retriever_type": "learned_linear"}
```

---

## 6. Fix kèm trong PR (bug thực tế)

`stockmem/tests/test_store.py` tạo `RecordRepository` (SQLAlchemy + aiosqlite), `init()` nhưng
**không đóng** ⇒ thread nền của aiosqlite còn sống, làm **treo full test suite** lúc interpreter
shutdown. Đã bọc thân test trong `try/finally` và gọi `await repo.close()`. Sau fix: bộ test
stockmem (không cần Postgres) **28 passed**, không treo, ruff sạch.

> Ghi chú minh bạch: các nghi vấn "coverage giả", "ANN prefilter cho learned path", "maturity guard
> đổi hành vi ngầm" nêu trong review ban đầu **không đúng với code đã commit** — đó là do lần đọc file
> bị cắt cụt trước đó. Code thật đã xử lý đúng cả ba (đã xác minh + 27/28 test pass). Bug thật duy
> nhất là rò rỉ kết nối trong `test_store.py` ở trên.
>
> Quan sát nhỏ (không sửa, chờ xác nhận): `cem_dataset.label_rows(fixed_band=0.01)` — với return
> đơn vị **phần trăm**, band 0.01% gần như khiến mọi ngày thành "có hướng" ở nhánh band `fixed`
> (nhánh chính `0.5sigma` không ảnh hưởng). Nếu chủ ý là 1%, nên để `1.0`.
