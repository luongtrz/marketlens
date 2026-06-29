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
        name="CoinDesk Ethereum",
        url="https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tag=ethereum",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="CoinTelegraph",
        url="https://cointelegraph.com/rss",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="CoinTelegraph Ethereum",
        url="https://cointelegraph.com/rss/tag/ethereum",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="Decrypt",
        url="https://decrypt.co/feed",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="Decrypt Ethereum",
        url="https://decrypt.co/feed/ethereum",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="CryptoSlate",
        url="https://cryptoslate.com/feed/",
        category="crypto_news",
    ),
    FeedSourceConfig(
        name="CryptoSlate Ethereum",
        url="https://cryptoslate.com/category/ethereum/feed/",
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

    supabase_url: str = "https://esctepjpgpjgrcymnabx.supabase.co"
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    articles_limit: int = 20

    model_config = {"env_prefix": "CRAWLER_", "env_file": "crawler/.env", "extra": "ignore"}
