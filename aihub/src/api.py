"""AIHub FastAPI application — exposes /sentiment, /factors, /predict endpoints."""

from fastapi import FastAPI

from aihub.src.sentiment.schema import SentimentRequest, SentimentResponse
from aihub.src.factors.schema import FactorRequest, FactorResponse
from aihub.src.predict.schema import PredictRequest, PredictResponse
from aihub.src.factors.skgp import SKGPExtractor
from aihub.src.sentiment.model import CryptoBertModel
from aihub.src.config import AIHubConfig

app = FastAPI(title="AIHub", description="AI inference service for crypto pipeline")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "models_loaded": False}  # TODO: wire up real model status


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(request: SentimentRequest) -> SentimentResponse:
    """Analyze sentiment of the provided text using CryptoBert.

    Args:
        request: SentimentRequest containing the text to analyze.

    Returns:
        SentimentResponse with score and label.
    """
    config = AIHubConfig()
    crypto_bert = CryptoBertModel(model_path=config.model_path, hf_model_path=config.hf_model_path)
    return crypto_bert.predict(request.text)


@app.post("/factors", response_model=FactorResponse)
async def factors(request: FactorRequest) -> FactorResponse:
    """Extract market factors from the provided text using SKGP.

    Args:
        request: FactorRequest containing the text to analyze.

    Returns:
        FactorResponse with list of extracted factors.
    """
    skgp = SKGPExtractor()
    return FactorResponse(factors = skgp.extract(request.ticker, request.text))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate trading signal using RAG with similar historical cases.

    Args:
        request: PredictRequest with current record and similar cases.

    Returns:
        PredictResponse with signal, confidence, and explanation.
    """
    raise NotImplementedError
