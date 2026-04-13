FROM python:3.12-slim

WORKDIR /app

COPY ../shared /app/shared
COPY . /app/aihub

RUN pip install --no-cache-dir -r /app/aihub/requirements.txt 2>/dev/null || true

ENV PYTHONPATH=/app

CMD ["uvicorn", "aihub.src.api:app", "--host", "0.0.0.0", "--port", "8001"]
