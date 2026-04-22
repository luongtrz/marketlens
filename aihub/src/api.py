"""AIHub FastAPI application — exposes /sentiment, /factors, /predict endpoints."""

from fastapi import FastAPI

from aihub.src.sentiment.schema import SentimentRequest, SentimentResponse
from aihub.src.factors.schema import FactorRequest, FactorResponse
from aihub.src.predict.schema import PredictRequest, PredictResponse
from aihub.src.factors.extractor import FactorExtractor
from aihub.src.sentiment.model import CryptoBertModel
from aihub.src.predict.client import PredictClient
from aihub.src.predict.rag_builder import RAGContextBuilder
from aihub.src.config import AIHubConfig
from aihub.src.llm import build_llm_client
from shared.models.prediction import SignalType

app = FastAPI(title="AIHub", description="AI inference service for crypto pipeline")


@app.get("/health")
async def health() -> dict:  # type: ignore[type-arg]
    return {"status": "ok", "models_loaded": False}


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(request: SentimentRequest) -> SentimentResponse:
    """Analyse sentiment of the provided text using CryptoBert."""
    config = AIHubConfig()
    model = CryptoBertModel(model_path=config.model_path, hf_model_path=config.hf_model_path)
    return model.predict(request.text)


@app.post("/factors", response_model=FactorResponse)
async def factors(request: FactorRequest) -> FactorResponse:
    """Extract market factors from the provided text using SKGP + LLM."""
    config = AIHubConfig()
    llm = build_llm_client(config)
    extractor = FactorExtractor(llm)
    return FactorResponse(factors=await extractor.extract(request.ticker, request.text))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate a trading signal using RAG with similar historical cases."""
    config = AIHubConfig()
    llm = build_llm_client(config)
    context = RAGContextBuilder().build(request.current, request.similar)
    result = await PredictClient(llm).generate(context)
    return PredictResponse(
        signal=SignalType(result["signal"]),
        confidence=float(result["confidence"]),
        explanation=result.get("explanation", ""),
        reasoning_steps=result.get("reasoning_steps", []),
    )
