"""Crawler module configuration."""

from pathlib import Path

from pydantic import BaseModel, Field

from shared.config.base_config import BaseAppConfig


class FeedSourceConfig(BaseModel):
    """Single feed source configuration."""

    name: str
    url: str
    category: str = "crypto_news"


DEFAULT_FEEDS: list[FeedSourceConfig] = [
    FeedSourceConfig(
        name="CoinDesk",
        url="https://www.coindesk.com/arc/outboundfeeds/rss/",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="CoinTelegraph",
        url="https://cointelegraph.com/rss",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="Decrypt",
        url="https://decrypt.co/feed",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="CryptoSlate",
        url="https://cryptoslate.com/feed/",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="The Block",
        url="https://www.theblock.co/rss.xml",
        category="crypto_news",
    ),
]


class CrawlerConfig(BaseAppConfig):
    """Configuration specific to the Crawler module."""

    poll_interval_seconds: int = 60
    enable_summary: bool = False
    aihub_url: str = "http://localhost:8001"
    dedup_backend: str = "memory"  # "redis" | "memory"
    factor_publish: bool = True  # Whether to emit factors to MessageBus
    fetch_content: bool = True
    enable_sitemap_backfill: bool = True
    sitemap_max_urls_per_source: int = 50000
    sitemap_max_sitemap_files: int = 0
    min_publish_year: int = 2023
    only_btc_eth: bool = True
    data_dir: str = str(Path(__file__).resolve().parents[1] / "data")
    seen_file: str = str(Path(__file__).resolve().parents[1] / "data" / "seen.json")
    articles_file: str = str(Path(__file__).resolve().parents[1] / "data" / "articles.jsonl")
    feeds: list[FeedSourceConfig] = Field(default_factory=lambda: list(DEFAULT_FEEDS))

    model_config = {"env_prefix": "CRAWLER_", "extra": "ignore"}
