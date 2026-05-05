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

    # Cron: comma-separated symbols to run daily, e.g. "BTC,ETH"
    cron_symbols: str = "BTC"
    # UTC hour to trigger daily run (0-23)
    cron_hour: int = 23
    cron_minute: int = 50

    model_config = SettingsConfigDict(
        env_prefix="MAIN_CONTROLLER_",
        env_file="main_controller/.env",
        extra="ignore",
    )
