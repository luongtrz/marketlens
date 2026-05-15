FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy lockfile + project manifest first (layer cache)
COPY pyproject.toml uv.lock README.md ./

# Copy source needed at runtime
COPY shared ./shared
COPY stockmem ./stockmem

# Install only the stockmem extra (no dev deps)
RUN uv sync --frozen --extra stockmem --no-dev
# Optuna is required by /weights/retrain (Bayesian optimization)
RUN uv pip install --python /app/.venv/bin/python optuna

ENV PYTHONPATH=/app
ENV DB_URL=postgresql+asyncpg://postgres:pass@localhost:5432/stockmem
ENV VECTOR_BACKEND=memory

CMD ["uv", "run", "uvicorn", "stockmem.src.api:app", "--host", "0.0.0.0", "--port", "8003"]
