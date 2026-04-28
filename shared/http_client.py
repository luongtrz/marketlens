"""Shared async HTTP client with retry and timeout configuration."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx


DEFAULT_CONNECT_TIMEOUT = 10.0  # seconds
DEFAULT_READ_TIMEOUT = 120.0  # seconds
DEFAULT_MAX_RETRIES = 3


@asynccontextmanager
async def get_client(
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client with retry and timeout configuration.

    Usage::

        async with get_client() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    Args:
        connect_timeout: Connection timeout in seconds.
        read_timeout: Read timeout in seconds.
        max_retries: Maximum number of retry attempts with exponential backoff.

    Yields:
        Configured httpx.AsyncClient instance.
    """
    timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
    transport = httpx.AsyncHTTPTransport(retries=max_retries)

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        yield client
