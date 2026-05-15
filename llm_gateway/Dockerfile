FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY shared /app/shared
COPY llm_gateway /app/llm_gateway

RUN pip install --no-cache-dir -r /app/llm_gateway/requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "llm_gateway.src.api:app", "--host", "0.0.0.0", "--port", "8006"]
