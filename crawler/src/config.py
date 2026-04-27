"""Crawler module configuration."""

from shared.config.base_config import BaseAppConfig


class CrawlerConfig(BaseAppConfig):
    """Configuration specific to the Crawler module."""

    poll_interval_seconds: int = 60
    enable_summary: bool = False
    aihub_url: str = "http://localhost:8001"
    dedup_backend: str = "memory"  # "redis" | "memory"
    factor_publish: bool = True  # Whether to emit factors to MessageBus

    supabase_url: str = "https://esctepjpgpjgrcymnabx.supabase.co"
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    articles_limit: int = 20

    model_config = {"env_prefix": "CRAWLER_", "env_file": "crawler/.env", "extra": "ignore"}
