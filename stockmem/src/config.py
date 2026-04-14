from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vector_backend: str = os.getenv("VECTOR_BACKEND", "memory")
    db_url: str = os.getenv("DB_URL", "sqlite+aiosqlite:///test.db")


settings = Settings()


def sqlite_path_from_url(db_url: str) -> str:
    prefix_async = "sqlite+aiosqlite:///"
    prefix_sync = "sqlite:///"
    if db_url.startswith(prefix_async):
        return db_url[len(prefix_async):]
    if db_url.startswith(prefix_sync):
        return db_url[len(prefix_sync):]
    raise ValueError(
        f"Unsupported DB_URL: {db_url}. Expected sqlite+aiosqlite:///path.db"
    )
