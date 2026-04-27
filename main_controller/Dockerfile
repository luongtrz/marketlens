FROM python:3.12-slim

WORKDIR /app

COPY shared /app/shared
COPY main_controller /app/main_controller

RUN pip install --no-cache-dir -r /app/main_controller/requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "main_controller.src.api:app", "--host", "0.0.0.0", "--port", "8005"]
