"""Request and response models for the LLM gateway."""

from pydantic import BaseModel, ConfigDict, Field

from shared.models.prediction import SignalType


class LLMDecisionRequest(BaseModel):
    """Structured request body accepted by POST /llm."""

    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(..., min_length=1)
    system: str | None = None
    include_reason: bool = False


class LLMDecisionResponse(BaseModel):
    """Normalized trading decision returned by POST /llm."""

    model_config = ConfigDict(extra="ignore")

    signal: SignalType
    reason: str | None = None
