# Crypto News Intelligence Pipeline

A modular, microservice-based system that continuously monitors crypto-related news, scores articles with AI models, correlates with market data, stores and retrieves historical patterns, and uses retrieval-augmented generation (RAG) to produce trading signal explanations and predictions.

## Architecture

The system consists of 6 independent modules, each exposing its own HTTP REST API:

| Module | Port | Responsibility |
|---|---|---|
| **Crawler** | 8000 | RSS feed polling, article enrichment via LLM |
| **AIHub** | 8001 | Sentiment analysis (CryptoBert), factor extraction (SKGP), RAG prediction |
| **MarketData** | 8002 | OHLCV data from Binance/TradingView, technical indicators |
| **StockMem** | 8003 | Daily record storage, vector similarity search |
| **FactorLedge** | 8004 | Factor normalization, cleaning, enrichment |
| **MainController** | 8005 | Pipeline orchestration, scheduling, result assembly |
| **LLMGateway** | 8006 | OpenCode Go gateway for final BUY/HOLD/SELL decisions |

## Quick Start

```bash
# Start all services
docker-compose up -d

# Trigger a pipeline run
curl -X POST http://localhost:8005/run -d '{"symbol": "BTCUSDT", "trigger": "manual"}'

# Check result
curl http://localhost:8005/result/{run_id}
```

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a single module
uvicorn aihub.src.api:app --port 8001 --reload
uvicorn llm_gateway.src.api:app --port 8006 --reload
```

## Configuration

Each module reads from environment variables and optional `config.yaml` files. See `architecture.md` for the full configuration reference.
