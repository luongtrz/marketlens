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
    k_similar: int = 5
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_access_ttl_minutes: int = 30
    jwt_refresh_ttl_days: int = 14
    jwt_issuer: str = "marketlens"

    # Cron: comma-separated symbols to run daily, e.g. "BTC,ETH"
    cron_symbols: str = "BTC"
    # UTC hour to trigger daily run (0-23)
    cron_hour: int = 23
    cron_minute: int = 50

    cache_enabled: bool = True
    cache_market_snapshot_ttl_seconds: int = 45
    cache_market_history_ttl_seconds: int = 300
    cache_market_historical_history_ttl_seconds: int = 86400
    cache_news_first_page_ttl_seconds: int = 120
    cache_ai_predict_ttl_seconds: int = 90000

    model_config = SettingsConfigDict(
        env_prefix="MAIN_CONTROLLER_",
        env_file="main_controller/.env",
        extra="ignore",
    )
