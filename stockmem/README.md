# StockMem Module

StockMem là memory/retrieval layer của hệ thống: lưu record thị trường theo ngày và trả về các case lịch sử tương tự để AIHub dùng cho RAG predict/explain.

## 1) Module này làm gì

- Lưu `StockMemRecord` theo khóa duy nhất `(date, symbol)`.
- Chuẩn hóa record thành vector đặc trưng.
- Tìm `k` record gần nhất bằng weighted similarity.
- Quản lý nhãn forward return (`future_return_1d/7d/30d`) để backtest và tối ưu trọng số.
- Tự động re-train trọng số similarity hằng ngày bằng Bayesian optimization (Optuna/TPE) nếu bật cấu hình.

---

## 2) Cấu trúc code và vai trò từng phần

### API layer
- `stockmem/src/api.py`
  - FastAPI app, endpoints:
    - `POST /record`
    - `GET /record/{id}`
    - `POST /search`
    - `GET /records/missing-returns`
    - `PATCH /record/{id}/returns`
    - `POST /weights/retrain` (manual trigger Bayesian retrain)
    - `GET /health`
  - Có background scheduler auto retrain daily khi `AUTO_OPTIMIZE_ENABLED=true`.

### Service layer
- `stockmem/src/service.py`
  - `StockMemService` điều phối toàn bộ module.
  - Chọn repository backend theo `DB_URL`:
    - PostgreSQL: `PGRepository`
    - SQLite: `RecordRepository`
  - Quản lý cache, embedder, searcher, weight reload runtime.
  - `auto_retrain_weights(...)` chạy Bayesian optimize và apply weights mới ngay trong process.

### Models
- `stockmem/src/models.py`
  - Schema chính: `StockMemRecord`, `MarketSnapshot`, `SimilarRecord`.
  - Có bridge để tương thích format market snapshot cũ/mới.

### Config
- `stockmem/src/config.py`
  - Runtime config cho DB/vector backend/weights và learned retriever artifact.
  - `LEARNED_RETRIEVER_FILE`: đường dẫn artifact JSON; file thiếu/rỗng sẽ fallback về fixed kNN.
  - Config auto-optimize daily:
    - `AUTO_OPTIMIZE_ENABLED`
    - `AUTO_OPTIMIZE_HOUR_UTC`
    - `AUTO_OPTIMIZE_MINUTE_UTC`
    - `AUTO_OPTIMIZE_HORIZON`
    - `AUTO_OPTIMIZE_TRIALS`
    - `AUTO_OPTIMIZE_K`
    - `AUTO_OPTIMIZE_WARMUP`
    - `AUTO_OPTIMIZE_MIN_RECORDS`
    - `AUTO_OPTIMIZE_OUTPUT`

### Search/Embedding
- `stockmem/src/search/embedder.py`
  - Split embedding:
    - `event_vec` (85 dims)
    - `factor_vec` (75 dims)
    - `indicator_vec` (5 dims)
    - `price_vec` (60 dims)
- `stockmem/src/search/searcher.py`
  - `fixed_knn` dùng weighted cosine:
    - `score = w1*sim(factor) + w2*sim(indicator) + w3*sim(price)`
  - `learned_linear` dùng learned per-feature diagonal metric và exact scan để đồng nhất với offline evaluation.
  - Nếu artifact không tồn tại, `learned_linear` fallback về `fixed_knn`.
  - Chỉ search trong cùng `symbol` với query.
  - Hỗ trợ `before_date` để tránh look-ahead trong backtest.
- `stockmem/src/search/event_memory.py`
  - Xây `DailyEventState`, novelty point-in-time, source diversity và vector event 85 chiều.
- `stockmem/src/search/index.py`
  - In-memory index (FAISS nếu available, fallback numpy).
- `stockmem/src/search/taxonomy.py`
  - Taxonomy mapping cho factor vector.

### Store/Persistence
- `stockmem/src/store/base.py`: repository protocol.
- `stockmem/src/store/repository.py`: SQLite impl + legacy migration.
- `stockmem/src/store/pg_repository.py`: PostgreSQL impl.
- `stockmem/src/store/writer.py`: upsert + incremental stats/index update (tránh full rebuild mỗi record).
- `stockmem/src/store/reader.py`: helpers đọc record.
- `stockmem/src/store/schema.py`: table constants.

### Bayesian retraining helper
- `stockmem/src/weights_retrainer.py`
  - Build dataset từ records đã có forward returns.
  - Gọi logic Optuna/TPE từ `stockmem/scripts/optimize_weights.py`.
  - Chọn stable weights (median của top trials), apply runtime, và persist snapshot JSON.

---

## 3) Data model chính

`StockMemRecord` gồm:
- Identity: `id`, `date`, `symbol`
- Sentiment/factors: `sentiment_score`, `sentiment_label`, `factors`, `normalized_factors`, `factor_vector`
- Market snapshot: RSI/MACD/MSI/FGI/price-change/candles
- Metadata/event memory: `summary`, `article_ids`, `article_sources`, `article_published_at`, `event_state`, `event_vector`
- Labels: `future_return_1d`, `future_return_7d`, `future_return_30d`

---

## 4) API contract ngắn gọn

### `POST /record`
Upsert record theo `(date, symbol)`.

### `GET /record/{id}`
Lấy record theo id.

### `POST /search`
Truy vấn tương đồng, có thể set `before_date` cho walk-forward và chọn
`retriever_type` là `fixed_knn` (mặc định) hoặc `learned_linear`.

### `GET /records/missing-returns`
Lấy records thiếu nhãn return (nếu thiếu bất kỳ trường nào trong `future_return_1d/7d/30d`).

### `PATCH /record/{id}/returns`
Patch `future_return_1d/7d/30d` cho record.

### `POST /weights/retrain`
Trigger Bayesian retrain ngay lập tức (manual).

### `GET /health`
Health check + backend/db info.

---

## 5) Auto Bayesian dynamic weights

### Cơ chế
- Khi `AUTO_OPTIMIZE_ENABLED=true`, StockMem chạy scheduler UTC mỗi ngày.
- Đến giờ cấu hình, service sẽ:
  1. Lấy records đã có đủ `future_return_1d/7d/30d`
  2. Chạy Bayesian optimize (Optuna TPE)
  3. Chọn stable weights (median top trials)
  4. Apply weights runtime vào searcher
  5. Ghi snapshot ra `AUTO_OPTIMIZE_OUTPUT`
  6. Dùng distributed lock (PostgreSQL advisory lock) để đảm bảo nhiều replica không retrain trùng.

### Thuật toán
- Objective: `0.6*DA + 0.4*Sharpe` (walk-forward).
- Ràng buộc: `w3 = 1 - w1 - w2` + bounds cho từng weight.
- Mặc định horizon: `7d` (configurable).

### Lưu ý vận hành
- Cần đủ dữ liệu đã dán nhãn future returns (`AUTO_OPTIMIZE_MIN_RECORDS`).
- Cần package `optuna` trong runtime (`pyproject` extra stockmem đã thêm).

---

## 6) Future-return backfill

StockMem chỉ cung cấp primitive:
- `GET /records/missing-returns`
- `PATCH /record/{id}/returns`

Việc tính return theo giá tương lai đang được orchestrate bởi Main Controller endpoint:
- `POST /fill-returns` (ở `main_controller/src/api.py`)

---

## 7) Scripts & data

### Scripts
- `stockmem/scripts/backtest_api.py`: walk-forward backtest qua API.
- `stockmem/scripts/optimize_weights.py`: Bayesian/grid optimization offline.
- `stockmem/scripts/benchmark_weights.py`: compare baseline vs candidate.
- `stockmem/scripts/regen_optimizer_data.py`: tái tạo dataset optimizer từ DB.
- `stockmem/scripts/build_cem_dataset.py`: kiểm tra split, label và maturity guard cho CEM dataset.
- `stockmem/scripts/train_learned_retriever.py`: train learned diagonal metric bằng numpy/Adam.
- `stockmem/scripts/evaluate_retriever.py`: so sánh guarded/leaky fixed baseline với learned retriever.
- `stockmem/scripts/generate_mock_data.py`: tạo mock dataset.
- `stockmem/scripts/patch_factors.py`: utility patch factors.

Optimizer và evaluator mặc định yêu cầu outcome của candidate đã mature theo ngày lịch.
Chỉ dùng `--no-maturity-guard` để tái tạo kết quả legacy có look-ahead.

### Data
- `stockmem/data/mock_3y_records.json`
- `stockmem/data/mock_3y_optimizer.json`
- `stockmem/data/real_optimizer.json`

---

## 8) Test files

- `stockmem/tests/test_store.py`
- `stockmem/tests/test_search.py`
- `stockmem/tests/test_event_memory.py`
- `stockmem/tests/test_learned_retriever.py`
- `stockmem/tests/test_vectorize.py`
- `stockmem/tests/test_pg_repository.py`
- `stockmem/tests/test_taxonomy.py`

---

## 9) Bug check (current) và technical debt

### Đã fix trong nhánh hiện tại
- Search có thể trộn record khác `symbol` trong cùng cache.
  - Đã fix: filter cùng `symbol` trong `RecordSearcher.search`.
- `missing-returns` chỉ lọc theo 1d.
  - Đã fix: kiểm tra thiếu bất kỳ `1d/7d/30d`.
- Auto retrain có thể chạy trùng khi đa replica.
  - Đã fix: PostgreSQL advisory lock (`pg_try_advisory_lock`).
- Save record gây rebuild `O(n)` mỗi lần.
  - Đã fix: incremental update cho corpus stats/index.
- Search full-scan tất cả records.
  - Đã fix: ANN prefilter bằng index joint vector + weighted rerank.

### Nợ kỹ thuật còn lại
- Auto retrain mới lock phân tán ở tầng PostgreSQL; môi trường SQLite/local xem như single-node (không có lock liên tiến trình).
- ANN prefilter đang dùng embedding joint rồi weighted rerank; nếu dữ liệu rất lớn vẫn nên tách hẳn ANN index theo chiến lược production (shard/persistent index).

---

## 10) Run local

```bash
source .venv/bin/activate
export PYTHONPATH=/home/luong/marketlens
export VECTOR_BACKEND=memory
export DB_URL=sqlite+aiosqlite:////tmp/stockmem-local.db
uvicorn stockmem.src.api:app --host 127.0.0.1 --port 8003
```

Smoke test:
```bash
curl -sS http://127.0.0.1:8003/health
```
