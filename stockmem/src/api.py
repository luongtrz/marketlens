"""StockMem FastAPI application — /record and /search endpoints."""

from fastapi import FastAPI

from shared.models.memory import StockMemRecord, SimilarRecord

app = FastAPI(title="StockMem", description="Record storage and similarity search service")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/record")
async def save_record(record: StockMemRecord) -> dict:
    """Persist a StockMemRecord.

    Args:
        record: The daily record to store.

    Returns:
        Dict with the assigned record ID.
    """
    raise NotImplementedError


@app.get("/record/{record_id}", response_model=StockMemRecord)
async def get_record(record_id: str) -> StockMemRecord:
    """Retrieve a record by ID.

    Args:
        record_id: UUID of the record.

    Returns:
        The StockMemRecord.
    """
    raise NotImplementedError


@app.post("/search")
async def search(query: StockMemRecord, k: int = 5) -> dict:
    """Search for similar historical records.

    Args:
        query: Current record for similarity comparison.
        k: Number of similar records to return.

    Returns:
        Dict with results list of SimilarRecord objects.
    """
    raise NotImplementedError
