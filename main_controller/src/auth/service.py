"""Auth service for signup, login, refresh token rotation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Any

import jwt
from passlib.context import CryptContext

from main_controller.src.auth.models import RefreshTokenORM, UserORM
from main_controller.src.auth.repository import AuthRepository


class AuthService:
    def __init__(
        self,
        repository: AuthRepository,
        *,
        jwt_secret: str,
        jwt_algorithm: str,
        access_ttl_minutes: int,
        refresh_ttl_days: int,
        issuer: str,
    ) -> None:
        self._repo = repository
        self._jwt_secret = jwt_secret
        self._jwt_algorithm = jwt_algorithm
        self._access_ttl_minutes = access_ttl_minutes
        self._refresh_ttl_days = refresh_ttl_days
        self._issuer = issuer
        self._pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def _hash_password(self, password: str) -> str:
        if len(password.encode("utf-8")) > 72:
            raise ValueError("Password cannot be longer than 72 bytes")
        return self._pwd_ctx.hash(password)

    def _verify_password(self, password: str, password_hash: str) -> bool:
        if len(password.encode("utf-8")) > 72:
            raise ValueError("Password cannot be longer than 72 bytes")
        return self._pwd_ctx.verify(password, password_hash)

    def _hash_refresh(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _encode_access(self, user: UserORM) -> str:
        now = datetime.now(timezone.utc)
        exp = now + timedelta(minutes=self._access_ttl_minutes)
        payload = {
            "sub": user.id,
            "email": user.email,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "iss": self._issuer,
        }
        return jwt.encode(payload, self._jwt_secret, algorithm=self._jwt_algorithm)

    async def _issue_tokens(self, user: UserORM) -> dict[str, Any]:
        access_token = self._encode_access(user)
        refresh_token = secrets.token_urlsafe(48)
        refresh_hash = self._hash_refresh(refresh_token)
        refresh_exp = datetime.now(timezone.utc) + timedelta(days=self._refresh_ttl_days)

        await self._repo.store_refresh_token(
            RefreshTokenORM(
                token_hash=refresh_hash,
                user_id=user.id,
                expires_at=refresh_exp,
            )
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(self._access_ttl_minutes * 60),
            "user": {"id": user.id, "email": user.email},
        }

    async def signup(self, email: str, password: str) -> dict[str, Any]:
        existing = await self._repo.get_user_by_email(email)
        if existing is not None:
            raise ValueError("Email already registered")

        user = UserORM(
            id=secrets.token_hex(16),
            email=email,
            password_hash=self._hash_password(password),
            is_active=True,
        )
        user = await self._repo.create_user(user)
        return await self._issue_tokens(user)

    async def login(self, email: str, password: str) -> dict[str, Any]:
        user = await self._repo.get_user_by_email(email)
        if user is None or not user.is_active:
            raise ValueError("Invalid credentials")
        if not self._verify_password(password, user.password_hash):
            raise ValueError("Invalid credentials")
        return await self._issue_tokens(user)

    async def refresh(self, refresh_token: str) -> dict[str, Any]:
        token_hash = self._hash_refresh(refresh_token)
        stored = await self._repo.get_refresh_token(token_hash)
        if stored is None:
            raise ValueError("Invalid refresh token")
        now = datetime.now(timezone.utc)
        if stored.expires_at < now:
            await self._repo.delete_refresh_token(token_hash)
            raise ValueError("Refresh token expired")

        user = await self._repo.get_user_by_id(stored.user_id)
        if user is None or not user.is_active:
            await self._repo.delete_refresh_token(token_hash)
            raise ValueError("Invalid refresh token")

        await self._repo.delete_refresh_token(token_hash)
        return await self._issue_tokens(user)
