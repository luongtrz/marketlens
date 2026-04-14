# StockMem Python Demo

Standalone FastAPI demo module for StockMem records and vector similarity search.

## Endpoints

- POST `/record`
- GET `/record/{id}`
- POST `/search`
- GET `/health`

## Standalone Test Mode

The `demo.py` entrypoint defaults to:

- `VECTOR_BACKEND=memory`
- `DB_URL=sqlite+aiosqlite:///test.db`

## Run

```bash
cd /home/luong/bitcoin_stockmem/market-similarity/new
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements-demo.txt
./.venv/bin/python demo.py
```

Then open `http://127.0.0.1:8000/docs`.
