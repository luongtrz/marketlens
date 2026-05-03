FROM python:3.12-slim

WORKDIR /app

COPY shared /app/shared
COPY factor_ledge /app/factor_ledge

RUN pip install --no-cache-dir -r /app/factor_ledge/requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "factor_ledge.src.api:app", "--host", "0.0.0.0", "--port", "8004"]
