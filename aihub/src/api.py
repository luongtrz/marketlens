"""AIHub FastAPI application — exposes /sentiment, /factors, /predict endpoints."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request

from aihub.src.sentiment.schema import SentimentRequest, SentimentResponse
from aihub.src.factors.schema import FactorRequest, FactorResponse
from aihub.src.predict.schema import PredictRequest, PredictResponse
from aihub.src.factors.extractor import FactorExtractor
from aihub.src.predict.rag_builder import RAGContextBuilder, StockMemClient
from aihub.src.config import AIHubConfig
from aihub.src.llm.models.factory import AIModelFactory
from aihub.src.llm.models.sentiment import SentimentModel
from aihub.src.llm.groq import GroqClient
from shared.models.prediction import SignalType


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load config and heavy models once at startup; clean up on shutdown."""
    config = AIHubConfig()
    factory = AIModelFactory(config)

    app.state.config = config
    app.state.groq_client = factory.get_client("groq")
    stockmem_client = StockMemClient(base_url=config.stockmem_url)
    app.state.rag_builder = RAGContextBuilder(stockmem_client=stockmem_client)
    app.state.sentiment_model = factory.create_sentiment_model()
    app.state.factor_extractor = FactorExtractor(factory.get_default_client())
    yield
    # (teardown if needed)


app = FastAPI(
    title="AIHub",
    description="AI inference service for crypto pipeline",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict: 
    return {"status": "ok", "models_loaded": True}


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(request: SentimentRequest, http: Request) -> SentimentResponse:
    """Analyse sentiment of the provided text using CryptoBert."""
    model: SentimentModel = http.app.state.sentiment_model
    result = await model.run(request.text)
    return SentimentResponse(score=result.score, label=result.label)


@app.post("/factors", response_model=FactorResponse)
async def factors(request: FactorRequest, http: Request) -> FactorResponse:
    """Extract market factors from the provided text using SKGP + LLM."""
    extractor: FactorExtractor = http.app.state.factor_extractor
    return FactorResponse(factors=await extractor.extract(request.ticker, request.text))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, http: Request) -> PredictResponse:
    """Generate a trading signal using RAG with similar historical cases."""
    groq: GroqClient = http.app.state.groq_client
    rag_builder: RAGContextBuilder = http.app.state.rag_builder
    
    from aihub.src.predict.client import PredictClient
    predict_client = PredictClient(llm=groq)

    current_text, similar_text = await rag_builder.build(request.current, request.similar or None)
    
    result = await predict_client.generate(current_text, similar_text)

    return PredictResponse(
        signal=SignalType(result["signal"]),
        confidence=float(result["confidence"]),
        explanation=result.get("explanation", ""),
        reasoning_steps=result.get("reasoning_steps", []),
    )
