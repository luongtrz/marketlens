# CEM-RAG — Train & Cấu trúc Event

> Tài liệu mô tả **ý nghĩa và thiết kế** của PR `feat(stockmem): add learned event-memory retriever`.
> Hai trục chính: **(A) Cấu trúc Event** (biến tin tức thành event-memory point-in-time) và
> **(B) Train** (học retriever từ outcome thay cho cosine cố định).
>
> Đây là tài liệu thiết kế/ý nghĩa — không mô tả chi tiết vận hành (deploy) hay các chỉnh sửa lặt vặt.

---

## 0. Ý nghĩa tổng thể của PR

MarketLens cũ truy hồi "ngày lịch sử tương tự" bằng **cosine có trọng số cố định**:
`score = w1·cos(factor) + w2·cos(indicator) + w3·cos(price)` (3 trọng số do Optuna chỉnh).
Hạn chế cốt lõi:

1. **Không học từ kết quả tương lai.** Ba trọng số chỉ phản ánh "độ giống bề mặt", không biết
   ngày nào *thực sự* có giá trị dự báo. Crypto thường có hiện tượng **"cùng tin, ngược kết quả"**
   (vd: "ETF approval" nhưng RSI 85 → sell-the-news) mà cosine bề mặt không phân biệt được.
2. **Tin tức bị bỏ phí.** Factor rời rạc chưa được cấu trúc hoá thành *event* có nhóm/loại, chưa đo
   được **độ lan toả** (bao nhiêu nguồn đưa tin) và **độ mới** (tin shock hay tin lặp lại).

PR này giải quyết cả hai bằng cách, theo đúng tinh thần 3 paper nền tảng:

| Trục | Lấy ý tưởng từ | Hiện thực trong PR |
|---|---|---|
| Event memory + Δinfo | **StockMem** (event-reflection memory) | `DailyEventState`, novelty, `incremental_information` |
| Độ lan toả tin tức | **FinGPT dissemination-aware** | `source_count`, `source_diversity`, `temporal_span` |
| Retriever học từ outcome | **FinSeer** (financial time-series RAG) | `teacher_relevance` + InfoNCE distillation |

**Nguyên tắc xuyên suốt:** mọi thay đổi đều **cộng thêm, tương thích ngược**. Mặc định
`retriever_type="fixed_knn"` ⇒ hệ thống chạy y hệt như trước; learned retriever chỉ bật khi có
artifact và yêu cầu tường minh.

---

# PHẦN A — CẤU TRÚC EVENT

## A.1. Vì sao cần "event memory" thay vì factor rời rạc

Giá crypto vận động theo **sự kiện**. Nhưng một headline đơn lẻ thì yếu; điều có giá trị dự báo là
một **cụm sự kiện** có nhiều nguồn độc lập đưa tin (lan toả cao), có **tính mới** (chưa bị thị trường
hấp thụ), và **giống các cụm sự kiện lịch sử** từng dẫn tới biến động. Vì vậy ta cần biến tin tức
thành một **trạng thái sự kiện theo ngày** (`DailyEventState`) có thể đo lường và truy hồi được,
thay vì danh sách factor thô.

## A.2. Schema — `shared/models/event.py`

```python
class EventRecord(BaseModel):
    event_group: str          # 1 trong 13 nhóm taxonomy
    event_type: str           # 1 trong 62 loại
    entities: list[str]
    polarity: float           # cực tính [-1, 1]
    confidence: float         # độ tin cậy [0, 1]
    observed_at: datetime | None
    description: str | None

class DailyEventState(BaseModel):
    date: date
    symbol: str
    events: list[EventRecord]
    article_count: int             # số bài trong ngày
    source_count: int              # số nguồn DUY NHẤT  → độ lan toả
    source_diversity: float        # entropy chuẩn hoá của nguồn → độ đa dạng
    temporal_span_hours: float     # độ trải thời gian các bài → "dồn dập" hay "rải rác"
    novelty_7d: float              # độ mới so với 7 ngày trước
    novelty_30d: float             # độ mới so với 30 ngày trước
    incremental_information: float # ≈ Δinfo: thông tin mới có trọng số lan toả
    dominant_event_groups: list[str]
```

`StockMemRecord` (`stockmem/src/models.py`) được mở rộng để mang event đi xuyên pipeline:
`event_state`, `event_vector`, `article_sources`, `article_published_at`. `extra="ignore"` giữ cho
các service khác không vỡ khi gặp field mới.

## A.3. Dựng event state — `stockmem/src/search/event_memory.py::build_daily_event_state`

Quy trình và **ý nghĩa từng bước**:

1. **Trích & gộp event từ factor.** Duyệt `record.factors`, ánh xạ qua taxonomy
   (`get_factor_type` / `get_factor_group`), **dedup theo `(event_group, event_type)`** để 20 bài
   cùng nói "ETF inflow" không thành 20 event độc lập. `polarity`/`confidence`/`observed_at` lấy từ
   `normalized_factors`.
2. **Độ lan toả (dissemination — ý tưởng FinGPT).**
   - `source_count` = số nguồn duy nhất (lowercase). Nhiều nguồn độc lập ⇒ event quan trọng hơn.
   - `source_diversity` = **entropy chuẩn hoá** của phân bố nguồn (1 = trải đều nhiều nguồn,
     0 = dồn vào 1 nguồn). Phân biệt "30 bài từ 12 nguồn" vs "30 bài từ 1 nguồn".
   - `temporal_span_hours` = khoảng cách bài sớm nhất → muộn nhất, từ `article_published_at`.
3. **Độ mới (novelty — ý tưởng StockMem).**
   `novelty_w = 1 − max_{ngày ∈ [T−w, T−1]} Jaccard(loại_event_hôm_nay, loại_event_ngày_đó)`,
   với `w ∈ {7, 30}`. Tin lặp lại nhiều ngày ⇒ novelty thấp; tin shock mới ⇒ novelty cao.
4. **Δinfo (incremental information).** `incremental_information = novelty_30d × breadth`, với
   `breadth = log1p(source_count) / log(21)`. Đây là điểm cốt lõi của StockMem: thứ dự báo giá
   không phải cực tính thô của tin, mà là **lượng thông tin MỚI** (lệch khỏi kỳ vọng) **có trọng số
   lan toả**. Tin tốt nhưng đã được "priced-in" sẽ có Δinfo thấp.
5. `dominant_event_groups` = top-3 nhóm theo tần suất — để giải thích/trace.

## A.4. Event vector 85 chiều — `build_event_vector`

`EVENT_DIM = 62 (type) + 13 (group) + 10 (scalar) = 85`:

- **62 bit type multi-hot + 13 bit group multi-hot**: biểu diễn nhị phân loại/nhóm sự kiện
  (giống biểu diễn của StockMem, dùng cho tương đồng Jaccard về bản chất).
- **10 scalar** (đều nén về ~[0,1]): `mean_polarity`, `max_abs_polarity`,
  `log(article_count)`, `log(source_count)`, `source_diversity`, `novelty_7d`, `novelty_30d`,
  `mean_confidence`, `temporal_span/168h`, `incremental_information`.

Vector này là **biểu diễn định lượng, kiểm định được** của tin tức trong ngày — thay vì để LLM
"đọc và quyết định", ta có một feature có thể đưa vào retriever và đo đạc.

## A.5. Event vào embedder — `stockmem/src/search/embedder.py::embed_split`

`embed_split()` nay trả **4 block** L2-normalize riêng từng block:
`SplitEmbedding(event_vec(85), factor_vec(75), indicator_vec(5), price_vec(60))`.
`event_vec` lấy từ `record.event_vector` nếu đủ 85 chiều, ngược lại dựng tại chỗ từ `event_state`.

> **Quan trọng:** vector joint 140d (`embed()`) dùng cho ANN index của fixed kNN **không** chứa
> event — event là đặc trưng **chỉ dành cho learned retriever**. Điều này giữ fixed kNN bất biến và
> tránh lệch khi so sánh.

---

# PHẦN B — TRAIN (LEARNED RETRIEVER)

## B.1. Vì sao "học" retriever, và học từ tín hiệu gì

FinSeer huấn luyện retriever bằng cách **chưng cất (distill)** "độ liên quan" do mô hình dự báo
cung cấp. Ở đây **không có LLM forecaster trong vòng lặp offline**, nên ta thay reward LLM bằng
**kết quả thực tế** (hướng `future_return_7d`): một ngày lịch sử "đáng truy hồi" nếu nó **cùng
hướng tương lai** với query, **cùng regime biến động**, và **giống về bề mặt**.

Mục tiêu là học một **metric tương đồng** sao cho hàng xóm được lấy ra **cùng hướng** với query, và
**đẩy xa** các "hard negative" — ngày nhìn rất giống nhưng kết quả ngược.

## B.2. Dữ liệu point-in-time, chống rò rỉ — `stockmem/scripts/cem_dataset.py`

- **Chia theo thời gian, có embargo 7 ngày** (không shuffle):
  Train ≤ `2024-12-24` · Val `2025-01-01 → 2025-06-23` · Test `2025-07-01 → 2026-05-01`.
  Val chỉ để chỉnh siêu tham số; Test chỉ đánh giá một lần.
- **Point-in-time / maturity (nguyên tắc thiết kế then chốt).** Ứng viên `c` chỉ dùng được cho
  query `q` khi `c.date + 7 ngày ≤ q.date` — so theo **ngày lịch** (`is_mature`). Lý do: tại thời
  điểm dự báo `q`, `future_return_7d` của một ngày lịch sử `c` **chỉ biết được** nếu cửa sổ 7 ngày
  của `c` đã đóng trước `q`. Áp dụng cả khi tạo cặp huấn luyện lẫn khi đánh giá (`matured_pool`).
- **Nhãn hướng (`label_rows`).** UP/DOWN/FLAT theo một **band**:
  - `0.5σ` (mặc định): ngưỡng = `0.5 × σ`, với `σ` là độ lệch chuẩn **nhân quả** của 7d-return
    (chỉ tính từ các ngày đã đáo hạn trước đó — `_causal_sigma`). Band thích nghi theo độ biến động.
  - `fixed`: band cố định ±1% (đơn vị phần trăm, nhất quán với return %). Dùng làm ablation.
- **Teacher relevance** (proxy cho reward không-có-LLM):
  `teacher_relevance = 0.45·outcome_sim + 0.35·regime_sim + 0.20·surface_sim`, **chỉ > 0** khi ứng
  viên cùng hướng anchor. Trong đó `outcome_sim` ~ độ gần về biên độ return, `regime_sim` ~ độ gần
  về biến động (log-ratio), `surface_sim` ~ cosine cũ. Đây là "điểm số giáo viên" để chọn positive.
- **Khai thác cặp (`mine_candidates`):**
  - **Positives**: top theo `teacher_relevance` (mặc định 3).
  - **Hard negatives**: ngày **ngược hướng** nhưng **nhìn rất giống** (lọc theo cosine cũ rồi theo
    chính metric đang học). Đây là linh hồn của thiết kế: bắt model phân biệt "cùng tin, ngược kết quả".
  - **Flat negatives**: vài ngày đi ngang, để model không nhầm "động" với "đứng yên".

## B.3. Metric học được — `stockmem/src/search/learned_metric.py`

`LearnedDiagonalMetric`: mỗi feature một trọng số `d ≥ 0` (diagonal) + trọng số khối `block_scales`:

```
score = Σ_block  scale_block · cos( d_block ⊙ q_block ,  d_block ⊙ c_block )   # chuẩn hoá theo từng block
```

- Khi `d = 1` ⇒ **trùng đúng** weighted-cosine kNN cũ (có unit test bảo chứng). Tức metric học
  **tổng quát hoá baseline**: học `d` chỉ mở rộng năng lực (trọng số theo từng chiều), không phá baseline.
- **Tự nhận 3 hoặc 4 block** theo dữ liệu: chưa có event ⇒ 3 block `(factor, indicator, price)`;
  có event ⇒ 4 block `(event, factor, indicator, price)`.

## B.4. Hàm mất mát & tối ưu — `stockmem/scripts/train_learned_retriever.py`

- **Loss = InfoNCE + distillation mềm + ridge:**
  - Với mỗi anchor, ứng viên gồm `[positives, hard_negs, flat_negs, in-batch negatives]`.
  - **Target mềm**: `softmax(positive_rewards / teacher_temperature)` trên các positive (chưng cất
    "ý kiến giáo viên"), thay vì one-hot cứng.
  - `loss = −Σ targets·log(softmax(score/temperature))`, `coefficients = probs − targets`,
    gradient **dạng đóng** qua Jacobian của phép chuẩn hoá (numpy thuần, **không cần torch**).
  - `+ λ·‖d − 1‖²`: kéo metric **về phía baseline** — vừa chống overfit (chỉ ~600 ngày train,
    1 tài sản), vừa là "lan can an toàn" để không tệ hơn baseline một cách vô lý.
- **Huấn luyện:** Adam; clip `d ∈ [0.05, 20]`; `block_scales` chuẩn hoá tổng = 1; **re-mine mỗi 3
  epoch** (vì metric đổi thì "hard negative" cũng đổi); **early stop theo `val hit@5`** (patience 10);
  **trung bình 5 seed** + báo `seed_std` (đo độ ổn định).
- **Optuna** chỉnh `temperature, teacher_temperature, ridge, hard_negs, positive_count,
  learning_rate, k, band` — **chỉ trên validation**.
- **Artifact** `stockmem/config/learned_retriever.json`: `version`, `block_dims`, `d`,
  `block_scales`, `band`, `splits`, `hyperparameters`, `val_hit_at_5`, `seed_std`, `mining_protocol`.

## B.5. Đánh giá & cổng nghiệm thu — `stockmem/scripts/evaluate_retriever.py`

So sánh **công bằng tuyệt đối** (cùng dữ liệu / cùng split / cùng horizon 7d / cùng maturity guard /
cùng k):

- **Đối tượng**: `baseline_fixed_guarded`, `baseline_fixed_leaky` (đo mức rò rỉ nếu bỏ maturity),
  `learned_diagonal`, cùng các **ablation**: `learned_factor_zeroed`, `learned_event_zeroed`
  (khi có event), `learned_fixed_band`.
- **Chỉ số retriever**: `hit@5` (top-5 cùng hướng query, độc lập với `k` giao dịch),
  `hard_negative_gap` (khoảng cách điểm giữa positive và hard-negative — đo khả năng phân biệt).
- **Chỉ số giao dịch**: tín hiệu BUY/SELL/HOLD theo ngưỡng (±2%), `coverage` thực,
  `buy_da/sell_da/hold_da`, `Sharpe`, `combined = 0.6·DA + 0.4·Sharpe`.
- **Ý nghĩa thống kê**: McNemar (exact) trên đúng/sai theo từng ngày + bootstrap CI cho delta DA.
- **Acceptance gate** (chốt trước khi mở test): `combined ≥ baseline + 0.01`;
  `(buy_da+sell_da)/2 ≥ +1pp`; McNemar `p < 0.10`; `seed_std < 0.03`; delta val & test cùng dấu.

## B.6. Tích hợp serving (cộng thêm, tương thích ngược)

- `SearchRequest.retriever_type ∈ {"fixed_knn", "learned_linear"}` (mặc định `fixed_knn`).
- `RecordSearcher`: nhánh `learned_linear` **quét toàn bộ** (không qua ANN prefilter) để khớp đúng
  đánh giá offline; giữ nguyên **regime bonus ±0.15**; gắn `event_match` (cosine event) +
  `retriever_version` vào mỗi kết quả để trace.
- `StockMemService.startup()` nạp artifact (`LEARNED_RETRIEVER_FILE`); `set_weights()` vẫn truyền
  lại metric đã học (auto-retrain trọng số không làm mất nó); `/search` truyền `retriever_type`.

---

# PHẦN C — TRAIN ↔ EVENT KẾT NỐI RA SAO

`event_vec` (Phần A) chính là **block đặc trưng thứ 4** mà metric học (Phần B) gán trọng số `d`/`scale`,
**cùng cơ chế** với factor/indicator/price. `LabeledRow.blocks` tự thêm event block khi có dữ liệu;
trainer tự suy `block_dims` (3 hoặc 4) và khởi tạo `scales = [0.05, 0.95·trọng_số_cũ]` cho cấu hình
4 block. Hệ quả: **cùng một trainer/searcher** chạy được cả khi chưa có và khi đã có event — không
phải viết lại gì khi event-memory được làm dày.

Nói cách khác: Phần A cung cấp **"tin tức đã cấu trúc hoá + đo lan toả/độ mới"**, Phần B **học cách
cân nhắc** đặc trưng đó (cùng với giá/chỉ báo) để truy hồi đúng các tiền lệ có giá trị dự báo.

---

# PHẦN D — Ý NGHĨA KHOA HỌC & HIỆN TRẠNG DỮ LIỆU

- Dataset offline `stockmem/data/real_optimizer.json` (962 ngày BTC, 2023→2026) hiện **88.6% factor
  rỗng** và **chưa có event_vec** ⇒ trainer đang chạy **3 block**; `event_vec` toàn-0 được tự bỏ
  (`load_rows`). Các con số "headline" sẽ chạy lại **nguyên trạng** khi event/factor được làm dày
  (qua `regen_optimizer_data.py` / backfill để đổ `event_vec`).
- **Ý nghĩa độc lập với việc retriever học có thắng fixed kNN hay không** — PR vẫn có 3 đóng góp:
  1. **Kỷ luật point-in-time**: ràng buộc maturity 7 ngày, đo được mức rò rỉ (`guarded` vs `leaky`).
  2. **Khung retriever học từ outcome** kiểu FinSeer, thay reward LLM bằng kết quả thực tế, hiện
     thực bằng numpy thuần — tái lập dễ.
  3. **Cấu trúc event + đặc trưng lan toả/độ mới** (StockMem + FinGPT) làm tin tức trở thành feature
     kiểm định được, giảm phụ thuộc vào quyết định LLM, và tạo được **trace** để giải thích.

---

## Lệnh chạy

```bash
# 1) Dựng dataset point-in-time (nhãn + split + band)
python stockmem/scripts/build_cem_dataset.py --data stockmem/data/real_optimizer.json

# 2) Train learned retriever (Optuna + 5 seed) → stockmem/config/learned_retriever.json
python stockmem/scripts/train_learned_retriever.py --data stockmem/data/real_optimizer.json --horizon 7d

# 3) Đánh giá fixed vs learned (guarded/leaky + ablation + McNemar + bootstrap)
python stockmem/scripts/evaluate_retriever.py \
  --data stockmem/data/real_optimizer.json --artifact stockmem/config/learned_retriever.json

# 4) Bật learned ở serving
export LEARNED_RETRIEVER_FILE=stockmem/config/learned_retriever.json
# rồi gọi /search với {"retriever_type": "learned_linear"}
```
