"""Auth repository backed by SQLAlchemy async engine."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from main_controller.src.auth.models import Base, RefreshTokenORM, UserORM


class AuthRepository:
    def __init__(self, db_url: str) -> None:
        self._engine = create_async_engine(db_url, future=True)
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def get_user_by_email(self, email: str) -> UserORM | None:
        async with self._session_factory() as session:
            return await session.scalar(select(UserORM).where(UserORM.email == email))

    async def get_user_by_id(self, user_id: str) -> UserORM | None:
        async with self._session_factory() as session:
            return await session.scalar(select(UserORM).where(UserORM.id == user_id))

    async def create_user(self, user: UserORM) -> UserORM:
        async with self._session_factory() as session:
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def store_refresh_token(self, token: RefreshTokenORM) -> None:
        async with self._session_factory() as session:
            session.add(token)
            await session.commit()

    async def get_refresh_token(self, token_hash: str) -> RefreshTokenORM | None:
        async with self._session_factory() as session:
            return await session.scalar(
                select(RefreshTokenORM).where(RefreshTokenORM.token_hash == token_hash)
            )

    async def delete_refresh_token(self, token_hash: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                delete(RefreshTokenORM).where(RefreshTokenORM.token_hash == token_hash)
            )
            await session.commit()

    async def delete_expired_tokens(self, now: datetime) -> None:
        async with self._session_factory() as session:
            await session.execute(delete(RefreshTokenORM).where(RefreshTokenORM.expires_at < now))
            await session.commit()
