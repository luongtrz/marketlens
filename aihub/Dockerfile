FROM python:3.12-slim

WORKDIR /app

COPY ../shared /app/shared
COPY . /app/aihub

RUN pip install --no-cache-dir -r /app/aihub/requirements.txt 2>/dev/null || true

ENV PYTHONPATH=/app

# Secrets are injected at runtime via docker-compose env_file or --env-file flag.
# Config defaults (model, backend, paths) live in aihub/src/config.py — no .env needed inside image.
# Example: docker run --env-file aihub/.env -p 8001:8001 aihub
CMD ["uvicorn", "aihub.src.api:app", "--host", "0.0.0.0", "--port", "8001"]
