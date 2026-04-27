FROM python:3.12-slim

WORKDIR /app

COPY shared /app/shared
COPY crawler /app/crawler

RUN pip install --no-cache-dir -r /app/crawler/requirements.txt

ENV PYTHONPATH=/app

CMD ["python", "-m", "crawler.src.main"]
