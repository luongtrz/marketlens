FROM python:3.12-slim

WORKDIR /app

COPY shared /app/shared
COPY market_data /app/market_data

RUN pip install --no-cache-dir -r /app/market_data/requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "market_data.src.api:app", "--host", "0.0.0.0", "--port", "8002"]
