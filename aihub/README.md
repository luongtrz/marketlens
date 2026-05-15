# AIHub

AI inference service for the MarketLens crypto pipeline. Exposes three REST endpoints consumed by the main controller.

## Endpoints

| Method | Path         | Description                             |
| ------ | ------------ | --------------------------------------- |
| `GET`  | `/health`    | Liveness check                          |
| `POST` | `/sentiment` | Sentiment score via CryptoBERT          |
| `POST` | `/factors`   | Market factor extraction via SKGP + LLM |
| `POST` | `/predict`   | Trading signal via RAG + LLM            |

## Setup

Secrets live in the repository root `.env`. Config defaults (model names, backends) are in `src/config.py`.

```bash
cp .env.example .env
# then fill in AIHUB_* API key(s)
```

Change `src/config.py` for the LLM and model you use.

## Running

```bash
# Local dev (from repo root)
uvicorn aihub.src.api:app --port 8001 --reload

# Docker (secrets injected at runtime)
docker build -t aihub .
docker run --env-file .env -p 8001:8001 aihub

# All services via compose (repo root)
docker-compose up -d
```

## Module Layout

```text
aihub/
+-- src/
|   +-- api.py          # FastAPI app + lifespan startup
|   +-- config.py       # AIHubConfig (pydantic-settings, AIHUB_* prefix)
|   +-- sentiment/      # CryptoBERT model wrapper
|   +-- factors/        # SKGP factor extractor
|   +-- predict/        # RAG context builder + LLM predict client
|   +-- llm/            # LLM client factory (Gemini / OpenAI / Groq)
+-- Dockerfile
```
