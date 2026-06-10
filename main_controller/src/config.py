"""MainController service configuration."""

from pydantic_settings import SettingsConfigDict

from shared.config.base_config import BaseAppConfig


class MainControllerConfig(BaseAppConfig):
    """Configuration for the MainController service."""

    crawler_url: str = "http://localhost:8000"
    aihub_url: str = "http://localhost:8001"
    llm_gateway_url: str = "http://localhost:8006"
    market_data_url: str = "http://localhost:8002"
    stockmem_url: str = "http://localhost:8003"
    factorledge_url: str = "http://localhost:8004"
    k_similar: int = 5
    predict_provider: str = "knn_returns"  # "knn_returns" | "aihub" | "llm_gateway"

    # kNN-returns signal thresholds (%)
    knn_buy_threshold: float = 2.0
    knn_sell_threshold: float = -2.0

    # kNN-returns horizon weights (must sum <= 1; normalized per-record if horizons missing)
    knn_return_w1d: float = 0.40
    knn_return_w3d: float = 0.30
    knn_return_w7d: float = 0.15
    knn_return_w15d: float = 0.10
    knn_return_w30d: float = 0.05
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

    # LLM signal post-policy tuning
    llm_min_directional_confidence: float = 0.58
    llm_hold_release_bias: float = 2.0

    # kNN confirmation: require similar-case avg 7d return to agree with LLM signal.
    # If LLM says BUY but kNN avg7 < -knn_confirm_threshold -> HOLD (and vice versa).
    # Set to 0.0 to disable.
    llm_knn_confirm_threshold: float = 1.0

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
