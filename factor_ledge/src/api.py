"""FactorLedge FastAPI application — /ingest, /factors, /summary endpoints."""

from fastapi import FastAPI, Query

from shared.models.factor import NormalizedFactor
from factor_ledge.src.processor.receiver import IngestRequest

app = FastAPI(title="FactorLedge", description="Factor normalization and enrichment service")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(request: IngestRequest) -> dict:
    """Ingest raw factors from the Crawler, process through the pipeline.

    Args:
        request: IngestRequest with article_id, raw factors, and source.

    Returns:
        Dict with processed list of NormalizedFactor objects.
    """
    raise NotImplementedError


@app.get("/factors", response_model=list[NormalizedFactor])
async def get_factors(
    symbol: str = Query(..., description="Symbol to filter by, e.g. BTC"),
    limit: int = Query(50, description="Maximum factors to return"),
) -> list[NormalizedFactor]:
    """Get processed factors for a symbol.

    Args:
        symbol: Symbol to filter by.
        limit: Maximum number of factors.

    Returns:
        List of NormalizedFactor objects.
    """
    raise NotImplementedError


@app.get("/summary")
async def summary(
    symbol: str = Query(..., description="Symbol to summarize, e.g. BTC"),
) -> dict:
    """Get a summary of factors for a symbol.

    Args:
        symbol: Symbol to summarize.

    Returns:
        FactorSummary dict.
    """
    raise NotImplementedError
