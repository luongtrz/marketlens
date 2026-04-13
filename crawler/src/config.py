"""Crawler module configuration."""

from shared.config.base_config import BaseAppConfig


class CrawlerConfig(BaseAppConfig):
    """Configuration specific to the Crawler module."""

    poll_interval_seconds: int = 60
    enable_summary: bool = False
    aihub_url: str = "http://localhost:8001"
    dedup_backend: str = "memory"  # "redis" | "memory"
    factor_publish: bool = True  # Whether to emit factors to MessageBus

    model_config = {"env_prefix": "CRAWLER_", "extra": "ignore"}
