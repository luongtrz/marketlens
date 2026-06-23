# Event Extraction Pipeline — `aihub/src/events/`

> Biến article/factor thô thành `EventRecord` có cấu trúc, phục vụ CEM-RAG event memory.

---

## Tại sao cần module này

`FactorLedge` hiện tại nhận factor string rời rạc (ví dụ `"SEC Regulatory Crackdown"`) và map sang taxonomy 62 type × 13 group.  
Module `events/` mở rộng bước đó: thay vì chỉ map type/group, ta còn gán **polarity** (bearish/bullish), **confidence**, **entities**, và cho phép dùng LLM để xử lý headline không có factor sẵn.

Output `EventRecord` được dùng để:
- Tính `event_vec` trong `StockMemRecord` (85d = 62 type + 13 group + 10 scalar)
- Xây `DailyEventState` (novelty, source diversity, dominant groups)
- Truy hồi evidence khi giải thích signal

---

## Files

```
aihub/src/events/
├── __init__.py
├── schema.py        # EventExtractionRequest / EventExtractionResponse
└── extractor.py     # EventExtractor — 3-tier logic
```

---

## Schema (`schema.py`)

```python
class EventExtractionRequest(BaseModel):
    symbol: str                          # "BTC", "ETH", ...
    title: str                           # Tiêu đề bài báo
    summary: str | None = None           # Tóm tắt (optional, dùng cho LLM tier)
    factors: list[str] = []              # Factor strings từ FactorLedge
    article_id: str | None = None        # ID article gốc để trace
    published_at: datetime | None = None

class EventExtractionResponse(BaseModel):
    events: list[EventRecord]
    method: str   # "rule_based" | "keyword" | "llm"
```

---

## Logic 3 tầng (`extractor.py`)

### Tầng 1 — Rule-based (luôn chạy)

Dùng `get_factor_type()` / `get_factor_group()` / `get_factor_sentiment()` từ `stockmem/src/search/taxonomy.py`.

```
factor string → (event_type, event_group, sentiment)
polarity:  bullish → +0.7 | bearish → -0.7 | neutral → 0.0
confidence: 0.6
```

Dedup theo `(event_group, event_type)`.  
Nếu factors list không rỗng thì tầng 1 thường đủ dùng.

### Tầng 2 — Keyword fallback (chỉ khi tầng 1 ra < 1 event)

Scan `title.lower()` với bảng keyword cứng:

| Keyword | event_type | event_group | polarity |
|---|---|---|---|
| "etf" + "approv" | ETF Approval | Regulation & Legal | +0.8 |
| "etf" + ("reject"\|"delay") | ETF Approval | Regulation & Legal | −0.7 |
| "sec" + ("sue"\|"enforce") | Enforcement Action | Regulation & Legal | −0.8 |
| "halving" | Supply Dynamics | Protocol & Product | +0.5 |
| "hack"\|"exploit"\|"breach" | Security Incident | Risk & Warning | −0.9 |
| "whale"\|"accumul" | Whale Activity | Whale & On-chain | +0.4 |
| "liquidat" | Liquidation Cascade | Market Performance | −0.6 |
| "fed"\|"fomc"\|"interest rate" | Interest Rate Decision | Macroeconomic | 0.0 |

confidence: 0.5

### Tầng 3 — LLM (Gemini Flash, chỉ khi llm được cấp VÀ tầng 1+2 ra < 2 events)

Prompt yêu cầu top-3 events từ title + summary.  
Kết quả được merge và dedup với tầng 1+2.  
Fail gracefully: mọi exception → log warning, giữ kết quả tầng 1+2.

```python
extractor = EventExtractor(llm=None)         # rule-based only, production-safe
extractor = EventExtractor(llm=gemini_client) # enable LLM tier
```

---

## API endpoint

```
POST /events/extract
Content-Type: application/json

{
  "symbol": "BTC",
  "title": "SEC approves spot Bitcoin ETF",
  "factors": ["ETF Approval", "Institutional Adoption"],
  "article_id": "art_12345",
  "published_at": "2026-01-15T10:00:00Z"
}
```

Response:
```json
{
  "events": [
    {
      "event_group": "Regulation & Legal",
      "event_type": "ETF Approval",
      "entities": [],
      "polarity": 0.8,
      "confidence": 0.6,
      "observed_at": "2026-01-15T10:00:00Z"
    }
  ],
  "method": "rule_based"
}
```

---

## Tích hợp vào pipeline

Trong `MainController`, sau bước `step_ai_score`, gọi `/events/extract` với factors đã extract → nhận `EventRecord` list → truyền vào `StockMemRecord.event_state` khi `POST /record`.

```python
event_resp = await aihub_client.extract_events(
    symbol=symbol,
    title=article.title,
    summary=article.summary,
    factors=[f.name for f in factors],
    article_id=article.id,
    published_at=article.published_at,
)
record.event_state = DailyEventState(
    date=today,
    symbol=symbol,
    events=event_resp.events,
    article_count=len(articles),
    ...
)
```

---

## Tests

```bash
PYTHONPATH=/home/luong/marketlens pytest aihub/tests/test_events.py -v
# 10 tests: rule-based lookup, polarity signs, all keyword branches, dedup, empty input
```
