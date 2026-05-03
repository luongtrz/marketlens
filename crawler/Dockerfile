FROM python:3.12-slim

WORKDIR /app

COPY shared /app/shared
COPY crawler /app/crawler

RUN pip install --no-cache-dir -r /app/crawler/requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "crawler.src.api:app", "--host", "0.0.0.0", "--port", "8000"]
