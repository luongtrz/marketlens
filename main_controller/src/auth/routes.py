"""Auth API routes for MainController."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from main_controller.src.auth.service import AuthService

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupPayload(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginPayload(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class RefreshPayload(BaseModel):
    refresh_token: str = Field(min_length=10)


class UserInfo(BaseModel):
    id: str
    email: EmailStr


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserInfo


def _service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    return service


@router.post("/signup", response_model=TokenResponse)
async def signup(payload: SignupPayload, request: Request) -> TokenResponse:
    service = _service(request)
    try:
        return TokenResponse(**await service.signup(payload.email.lower(), payload.password))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginPayload, request: Request) -> TokenResponse:
    service = _service(request)
    try:
        return TokenResponse(**await service.login(payload.email.lower(), payload.password))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/refresh", response_model=TokenResponse)
async def refresh(payload: RefreshPayload, request: Request) -> TokenResponse:
    service = _service(request)
    try:
        return TokenResponse(**await service.refresh(payload.refresh_token))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
