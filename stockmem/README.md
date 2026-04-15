# StockMem

StockMem is the module responsible for record storage and weighted vector similarity search.

## Endpoints

- POST /record
- GET /record/{id}
- POST /search
- GET /health

## Weighted Similarity

Search ranking uses:

score = w1 * sim(factor) + w2 * sim(indicator) + w3 * sim(price)

Default coefficients are loaded from `stockmem/config.yaml`:

- `w1_factor`: 0.35
- `w2_indicator`: 0.20
- `w3_price`: 0.45

Override options at runtime:

- `W1_FACTOR`, `W2_INDICATOR`, `W3_PRICE`
- `WEIGHTS_FILE=/path/to/weights.json`

## Local Setup

Run from repository root:

```bash
cd /home/luong/marketlens
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings httpx sqlalchemy aiosqlite asyncpg numpy pyyaml pytest pytest-asyncio pytest-cov
```

## Run Tests

```bash
cd /home/luong/marketlens
source .venv/bin/activate
PYTHONPATH=/home/luong/marketlens pytest -q stockmem/tests
```

Expected result:

- 2 passed (test_store + test_search)

## Optimize / Benchmark Weights

Optimizer uses Bayesian Optimization (TPE sampler via Optuna), ported from the original logic.

Input data should be vectorized rows with fields:

- `date`
- `factor_vec`
- `indicator_vec`
- `price_vec`
- `future_return_7d`

Install optimizer dependency:

```bash
cd /home/luong/marketlens
source .venv/bin/activate
pip install optuna
```

Optimize:

```bash
cd /home/luong/marketlens
source .venv/bin/activate
python stockmem/scripts/optimize_weights.py \
	--data /path/to/history_real_optimizer.v2.json \
	--trials 120 \
	--horizon 7d \
	--k 5 \
	--warmup 250 \
	--cv-folds 4 \
	--stable-top-k 12 \
	--output stockmem/config/weights.optimized.json
```

Notes:

- Objective: `0.6 * DA + 0.4 * Sharpe`
- Constraint: `w3 = 1 - w1 - w2` and each weight must stay in configured bounds
- Output includes best-trial weights and stable median weights from top trials

Benchmark vs baseline:

```bash
cd /home/luong/marketlens
source .venv/bin/activate
python stockmem/scripts/benchmark_weights.py \
	--data /path/to/history_real_optimizer.v2.json \
	--weights stockmem/config/weights.optimized.json
```

## Run API Locally

```bash
cd /home/luong/marketlens
source .venv/bin/activate
export PYTHONPATH=/home/luong/marketlens
export VECTOR_BACKEND=memory
export DB_URL=sqlite+aiosqlite:////tmp/stockmem-local.db
uvicorn stockmem.src.api:app --host 127.0.0.1 --port 18080
```

## Smoke Test

Open another terminal and run:

```bash
BASE=http://127.0.0.1:18080

curl -sS "$BASE/health"

curl -sS -X POST "$BASE/record" \
	-H 'Content-Type: application/json' \
	-d '{"record":{"date":"2026-04-14","symbol":"BTC","sentiment_score":0.45,"factors":["macro","etf_flow"],"market_snapshot":{"rsi":55.1,"macd_hist":0.012},"summary":"initial summary","article_ids":["a1"]}}'

curl -sS "$BASE/record/<id-from-record-response>"

curl -sS -X POST "$BASE/search" \
	-H 'Content-Type: application/json' \
	-d '{"query":{"date":"2026-04-14","symbol":"BTC","sentiment_score":0.46,"factors":["macro"],"market_snapshot":{"rsi":55.0,"macd_hist":0.010},"summary":"query","article_ids":[]},"k":3}'
```

## Idempotency Check (same date + symbol)

Write the same key twice, then verify IDs are equal and DB has one row for that key.

```bash
BASE=http://127.0.0.1:18080

REC1=$(curl -sS -X POST "$BASE/record" -H 'Content-Type: application/json' -d '{"record":{"date":"2026-04-14","symbol":"BTC","sentiment_score":0.45,"factors":["macro"],"market_snapshot":{"rsi":55.1,"macd_hist":0.012},"summary":"first","article_ids":["a1"]}}')
ID1=$(printf '%s' "$REC1" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')

REC2=$(curl -sS -X POST "$BASE/record" -H 'Content-Type: application/json' -d '{"record":{"date":"2026-04-14","symbol":"BTC","sentiment_score":0.47,"factors":["macro","onchain"],"market_snapshot":{"rsi":56.2,"macd_hist":0.02},"summary":"updated","article_ids":["a2"]}}')
ID2=$(printf '%s' "$REC2" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')

python3 - <<'PY'
import sqlite3
conn = sqlite3.connect('/tmp/stockmem-local.db')
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM stockmem_records WHERE record_date=? AND symbol=?", ('2026-04-14', 'BTC'))
print('row_count_for_key=', cur.fetchone()[0])
conn.close()
PY

echo "ID1=$ID1"
echo "ID2=$ID2"
```

Expected result:

- ID1 equals ID2
- row_count_for_key equals 1

## Notes

- Local sqlite files are ignored by git via *.db rules.
- This README reflects the current migration branch behavior in this repository.
