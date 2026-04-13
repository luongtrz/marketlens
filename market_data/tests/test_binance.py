"""Tests for Binance market data adapter."""

import pytest


class TestBinanceSource:
    """Test suite for BinanceSource."""

    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self) -> None:
        """Test OHLCV candle fetching."""
        # TODO: Mock Binance API
        pass

    @pytest.mark.asyncio
    async def test_fetch_ticker(self) -> None:
        """Test ticker fetching."""
        # TODO: Mock Binance API
        pass
