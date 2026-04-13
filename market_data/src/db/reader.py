"""Query historical candles from the market database."""

from shared.models.market import OHLCV


class MarketDBReader:
    """Reads historical OHLCV data from the database.

    Args:
        db_url: SQLAlchemy-compatible async database URL.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    async def get_candles(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[OHLCV]:
        """Query historical candles from the database.

        Args:
            symbol: Trading pair.
            interval: Candle interval.
            limit: Maximum number of candles to return.

        Returns:
            List of OHLCV candles ordered by timestamp ascending.
        """
        raise NotImplementedError
