FROM python:3.12-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir \
    numpy \
    optuna \
    pydantic

CMD ["python", "stockmem/scripts/retrain_finbert_retriever.py", \
     "--input-ndjson", "data/exports/stockmem_records.ndjson", \
     "--dataset-output", "stockmem/data/real_optimizer_finbert.json", \
     "--artifact-output", "stockmem/config/learned_retriever_finbert.json", \
     "--trials", "10", \
     "--epochs", "40", \
     "--seeds", "5"]
