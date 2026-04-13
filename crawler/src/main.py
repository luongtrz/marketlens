"""Crawler module entry point — starts the RSS polling loop."""

import asyncio
import logging

from crawler.src.config import CrawlerConfig
from shared.config.loader import load_config

logger = logging.getLogger(__name__)


async def start() -> None:
    """Initialize and start the Crawler polling loop."""
    config = load_config(CrawlerConfig, yaml_path="crawler/config.yaml")
    logger.info("Starting Crawler with poll_interval=%ds", config.poll_interval_seconds)
    # TODO: Initialize RSSFetcher, Deduplicator, LLMClient, DBWriter
    # TODO: Start the polling loop
    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(start())
