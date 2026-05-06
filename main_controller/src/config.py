"""MainController service configuration."""

from pydantic_settings import SettingsConfigDict

from shared.config.base_config import BaseAppConfig


class MainControllerConfig(BaseAppConfig):
    """Configuration for the MainController service."""

    crawler_url: str = "http://localhost:8000"
    aihub_url: str = "http://localhost:8001"
    market_data_url: str = "http://localhost:8002"
    stockmem_url: str = "http://localhost:8003"
    factorledge_url: str = "http://localhost:8004"
    k_similar: int = 3
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_access_ttl_minutes: int = 30
    jwt_refresh_ttl_days: int = 14
    jwt_issuer: str = "marketlens"

    model_config = SettingsConfigDict(
        env_prefix="MAIN_CONTROLLER_",
        env_file="main_controller/.env",
        extra="ignore",
    )
