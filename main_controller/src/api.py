"""MainController FastAPI application — /run, /status, /result endpoints."""

from fastapi import FastAPI

from shared.models.prediction import PredictionResult

app = FastAPI(title="MainController", description="Pipeline orchestration service")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/run")
async def run(symbol: str, trigger: str = "manual") -> dict:
    """Trigger a pipeline run for the given symbol.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT").
        trigger: "manual" or "scheduled".

    Returns:
        Dict with run_id and status.
    """
    raise NotImplementedError


@app.get("/status/{run_id}")
async def status(run_id: str) -> dict:
    """Get the status of a pipeline run.

    Args:
        run_id: UUID of the run.

    Returns:
        RunStatus dict.
    """
    raise NotImplementedError


@app.get("/result/{run_id}", response_model=PredictionResult)
async def result(run_id: str) -> PredictionResult:
    """Get the result of a completed pipeline run.

    Args:
        run_id: UUID of the run.

    Returns:
        PredictionResult.
    """
    raise NotImplementedError
