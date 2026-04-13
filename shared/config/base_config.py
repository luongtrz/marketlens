"""Base Pydantic settings shared by all modules."""

from pydantic_settings import BaseSettings


class BaseAppConfig(BaseSettings):
    """Base configuration with fields common to every module."""

    env: str = "development"  # "development" | "production"
    log_level: str = "INFO"  # "DEBUG" | "INFO" | "WARNING"
    db_url: str = "sqlite+aiosqlite:///local.db"
    redis_url: str = "redis://localhost:6379"
    mock: bool = False  # True to disable all external calls

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}
