# MarketLens VPS — API Usage Guide

Base URL: `http://152.42.238.138`

> Tất cả service đều có `/health` → `{"status": "ok"}`. Dùng để kiểm tra service còn sống không.

---

## Tổng quan kiến trúc

```
Frontend :3000  →  MainController :8005
                        ├── Crawler      :8000  (news từ Supabase)
                        ├── AIHub        :8001  (sentiment, factors, predict)
                        ├── MarketData   :8002  (OHLCV, indicators từ Binance)
                        ├── StockMem     :8003  (lưu & tìm similar records)
                        └── FactorLedge  :8004  (normalize & lưu factors)
```

---

## 1. MainController — Port 8005

**Service điều phối toàn bộ pipeline.** Đây là entry point chính.

### `POST /run` — Chạy pipeline phân tích

```bash
curl -X POST "http://152.42.238.138:8005/run?symbol=BTC"
```

| Param | Bắt buộc | Mô tả |
|---|---|---|
| `symbol` | ✅ | Ký hiệu coin: `BTC`, `ETH`, `SOL`, ... |
| `trigger` | ❌ | Nhãn trigger, mặc định `manual` |

Response (trả về ngay, pipeline chạy async):
```json
{
  "run_id": "3d203a58-fb53-4f20-b416-fb816454ef06",
  "status": "pending"
}
```

---

### `GET /status/{run_id}` — Kiểm tra trạng thái

```bash
curl "http://152.42.238.138:8005/status/3d203a58-fb53-4f20-b416-fb816454ef06"
```

Response:
```json
{
  "run_id": "3d203a58-...",
  "symbol": "BTC",
  "status": "done",         // pending | running | done | failed
  "started_at": "2026-05-05T02:17:08Z",
  "finished_at": "2026-05-05T02:17:22Z",
  "has_result": true
}
```

---

### `GET /result/{run_id}` — Lấy kết quả dự báo

```bash
curl "http://152.42.238.138:8005/result/3d203a58-fb53-4f20-b416-fb816454ef06"
```

Response:
```json
{
  "signal": "BUY",            // BUY | SELL | HOLD
  "confidence": 0.8,          // 0.0 – 1.0
  "explanation": "RSI chưa overbought, MACD bullish crossover...",
  "reasoning_steps": [],
  "similar_cases": [
    {
      "record": {
        "date": "2026-04-11",
        "symbol": "BTC",
        "sentiment_score": -0.492,
        "sentiment_label": "bearish",
        "factors": ["btc_price_surge", "us_crypto_regulation"],
        "market_snapshot": { "ohlcv": {...}, "indicators": {...} }
      },
      "similarity": 0.72
    }
  ]
}
```

> **Workflow thông thường:**
> ```bash
> # 1. Trigger pipeline
> run_id=$(curl -s -X POST "http://152.42.238.138:8005/run?symbol=BTC" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
> # 2. Chờ ~10-15 giây rồi lấy kết quả
> sleep 15
> curl "http://152.42.238.138:8005/result/$run_id"
> ```

---

### `POST /backfill` — Đẩy dữ liệu lịch sử vào StockMem

Dùng để populate StockMem với historical records từ Supabase. Gọi nhiều lần với `offset` tăng dần.

```bash
# Batch 1: 30 ngày gần nhất
curl -X POST "http://152.42.238.138:8005/backfill?symbol=BTC&days=30&offset=0"

# Batch 2: 30-60 ngày trước
curl -X POST "http://152.42.238.138:8005/backfill?symbol=BTC&days=30&offset=30"

# Batch 3: 60-90 ngày trước
curl -X POST "http://152.42.238.138:8005/backfill?symbol=BTC&days=30&offset=60"
```

| Param | Mặc định | Mô tả |
|---|---|---|
| `symbol` | — | Coin cần backfill |
| `days` | `30` | Số ngày mỗi batch |
| `offset` | `0` | Bắt đầu từ bao nhiêu ngày trước (so với hôm nay) |

Response:
```json
{
  "symbol": "BTC",
  "days": 30,
  "offset": 0,
  "window": "2026-04-05 → 2026-05-05",
  "articles_fetched": 652,
  "dates_with_articles": 9,
  "saved": 9,
  "skipped_no_ohlcv": 0,
  "errors": []
}
```

> **Backfill toàn bộ DB (script):**
> ```bash
> for offset in $(seq 0 30 1220); do
>   curl -s -X POST "http://152.42.238.138:8005/backfill?symbol=BTC&days=30&offset=$offset"
>   echo ""
> done
> ```

---

### UI Routes (dành cho Frontend)

| Endpoint | Mô tả |
|---|---|
| `GET /api/ai/latest-news?start=&end=&tag=` | Lấy tin tức mới nhất, filter theo ngày / tag |
| `GET /api/ai/historical-news?coinName=BTC&date=2026-05-01` | Tin tức theo ngày cụ thể |
| `POST /api/ai/analyze-article` | Phân tích sentiment 1 bài |
| `POST /api/ai/forecast` | Forecast nhanh (không lưu StockMem) |
| `POST /api/ai/ask-chart` | Hỏi về chart/indicators |
| `POST /api/ai/ask-news` | Hỏi về news context |
| `POST /api/ai/chat` | Chat tổng quát với AI |

---

## 2. Crawler — Port 8000

**Đọc articles từ Supabase** (đã có sentiment_score sẵn).

### `GET /articles/latest` — Lấy tin mới nhất

```bash
curl "http://152.42.238.138:8000/articles/latest?symbol=BTC&limit=10"
```

| Param | Mặc định | Mô tả |
|---|---|---|
| `symbol` | — | Lọc theo coin: `BTC`, `ETH`, ... |
| `limit` | `20` | Số bài tối đa |

Response (mảng article):
```json
[
  {
    "id": "66214",
    "article_name": "Bitcoin climbs past 80K for the first time since January",
    "source": "cointelegraph.com",
    "url": "https://...",
    "date_published": "2026-05-04T06:17:30Z",
    "sentiment_score": 0.85,
    "sentiment_label": "bullish",
    "factors": [],
    "raw_text": "..."
  }
]
```

---

## 3. AIHub — Port 8001

**AI inference:** sentiment, factor extraction, RAG predict.

### `POST /sentiment` — Phân tích sentiment văn bản

```bash
curl -X POST http://152.42.238.138:8001/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin surges to new high driven by institutional buying"}'
```

Response:
```json
{
  "score": 0.82,     // -1.0 (bearish) → +1.0 (bullish)
  "label": "Bullish"
}
```

---

### `POST /factors` — Trích xuất market factors từ text

```bash
curl -X POST http://152.42.238.138:8001/factors \
  -H "Content-Type: application/json" \
  -d '{"ticker": "BTC", "text": "SEC approves Bitcoin ETF, institutional demand surges"}'
```

Response:
```json
{
  "factors": [
    { "name": "bitcoin_etf_approval", "type": "regulatory", "weight": 0.9, "polarity": 0.8 },
    { "name": "institutional_demand",  "type": "macro",      "weight": 0.7, "polarity": 0.7 }
  ]
}
```

---

### `POST /predict` — Dự báo signal dựa trên RAG

Thường được gọi nội bộ bởi MainController, nhưng có thể gọi trực tiếp:

```bash
curl -X POST http://152.42.238.138:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "current": { <StockMemRecord> },
    "similar": [ <SimilarRecord>, ... ]
  }'
```

Response:
```json
{
  "signal": "BUY",
  "confidence": 0.75,
  "explanation": "...",
  "reasoning_steps": []
}
```

---

## 4. MarketData — Port 8002

**Lấy dữ liệu giá từ Binance.**

### `GET /snapshot` — Snapshot giá + indicators hiện tại

```bash
curl "http://152.42.238.138:8002/snapshot?symbol=BTC"
```

Response:
```json
{
  "symbol": "BTC",
  "timestamp": "2026-05-05T02:16:19Z",
  "ohlcv": {
    "open": 79861.01, "high": 80530.71, "low": 79808.72,
    "close": 80464.56, "volume": 1202.96, "interval": "1d"
  },
  "indicators": {
    "rsi": 67.55,
    "macd": { "macd": [...], "signal": [...], "histogram": [...] },
    "bb": { "upper": [...], "middle": [...], "lower": [...] }
  },
  "source": "binance"
}
```

---

### `GET /history` — OHLCV lịch sử

```bash
# 30 nến gần nhất
curl "http://152.42.238.138:8002/history?symbol=BTC&interval=1d&limit=30"

# 30 nến kết thúc tại ngày cụ thể (Unix ms timestamp)
curl "http://152.42.238.138:8002/history?symbol=BTC&interval=1d&limit=30&end_time=1746576000000"
```

| Param | Mặc định | Mô tả |
|---|---|---|
| `symbol` | — | `BTC`, `ETH`, ... |
| `interval` | `1d` | `1m`, `5m`, `1h`, `4h`, `1d` |
| `limit` | — | Số nến (tối đa 1000) |
| `end_time` | now | Unix timestamp milliseconds |

Response: mảng OHLCV objects.

---

### `GET /symbols` — Danh sách coin đang theo dõi

```bash
curl "http://152.42.238.138:8002/symbols"
# → {"symbols": ["BTC", "ETH"]}
```

---

## 5. StockMem — Port 8003

**Lưu trữ và tìm kiếm similar records** (vector similarity).

### `POST /record` — Lưu 1 daily record

```bash
curl -X POST http://152.42.238.138:8003/record \
  -H "Content-Type: application/json" \
  -d '{
    "record": {
      "date": "2026-05-05",
      "symbol": "BTC",
      "sentiment_score": 0.024,
      "sentiment_label": "neutral",
      "factors": ["btc_price_surge", "institutional_demand"],
      "market_snapshot": { ... },
      "summary": "Bitcoin climbs past 80K..."
    }
  }'
# → {"id": "d4d18502-..."}
```

---

### `POST /search` — Tìm k records tương tự nhất

```bash
curl -X POST http://152.42.238.138:8003/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": { <StockMemRecord> },
    "k": 3
  }'
```

Response:
```json
{
  "results": [
    {
      "record": { "date": "2026-04-11", "sentiment_score": -0.49, ... },
      "similarity": 0.72
    }
  ]
}
```

---

### `GET /record/{record_id}` — Lấy 1 record theo ID

```bash
curl "http://152.42.238.138:8003/record/d4d18502-2e59-49ef-b616-87829d8183e5"
```

---

## 6. FactorLedge — Port 8004

**Normalize và lưu market factors.**

### `POST /ingest` — Lưu factors từ 1 bài

```bash
curl -X POST http://152.42.238.138:8004/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "article_id": "article_123",
    "factors": ["btc_price_surge", "institutional_demand"],
    "source": "aihub"
  }'
```

---

### `GET /factors` — Lấy danh sách factors đã normalize

```bash
curl "http://152.42.238.138:8004/factors?symbol=BTC&limit=10"
```

Response: mảng NormalizedFactor với `name`, `type`, `weight`, `polarity`, `sector`.

---

### `GET /summary` — Tóm tắt factors theo symbol

```bash
curl "http://152.42.238.138:8004/summary?symbol=BTC"
```

---

## 7. Frontend — Port 3000

```
http://152.42.238.138:3000
```

UI dashboard — truy cập trực tiếp bằng trình duyệt.

---

## Cron Job

Pipeline tự động chạy mỗi ngày lúc **23:50 UTC** cho symbol `BTC`.  
Config qua env vars trong `.env`:

```env
MAIN_CONTROLLER_CRON_SYMBOLS=BTC,ETH   # thêm coin
MAIN_CONTROLLER_CRON_HOUR=23
MAIN_CONTROLLER_CRON_MINUTE=50
```

---

## Quick Reference

```bash
VPS=http://152.42.238.138

# Chạy pipeline và lấy kết quả
run_id=$(curl -s -X POST "$VPS:8005/run?symbol=BTC" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
sleep 15 && curl "$VPS:8005/result/$run_id"

# Kiểm tra health tất cả service
for port in 8000 8001 8002 8003 8004 8005; do
  echo -n "Port $port: " && curl -s "$VPS:$port/health"
  echo
done

# Backfill 90 ngày gần nhất
for offset in 0 30 60; do
  curl -s -X POST "$VPS:8005/backfill?symbol=BTC&days=30&offset=$offset"
  echo
done
```
