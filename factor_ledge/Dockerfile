FROM python:3.12-slim

WORKDIR /app

COPY ../shared /app/shared
COPY . /app/factor_ledge

RUN pip install --no-cache-dir -r /app/factor_ledge/requirements.txt 2>/dev/null || true

ENV PYTHONPATH=/app

CMD ["uvicorn", "factor_ledge.src.api:app", "--host", "0.0.0.0", "--port", "8004"]
