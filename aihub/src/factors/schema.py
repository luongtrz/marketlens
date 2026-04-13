"""Request/Response schemas for the /factors endpoint."""

from pydantic import BaseModel, ConfigDict

from shared.models.factor import Factor


class FactorRequest(BaseModel):
    """Request payload for factor extraction."""

    model_config = ConfigDict(extra="ignore")

    text: str


class FactorResponse(BaseModel):
    """Response from factor extraction."""

    model_config = ConfigDict(extra="ignore")

    factors: list[Factor]
