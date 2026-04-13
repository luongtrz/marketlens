"""Write OHLCV + indicators to the market database."""

from shared.models.market import OHLCV, MarketSnapshot


class MarketDBWriter:
    """Persists OHLCV candles and indicator data to the database.

    Args:
        db_url: SQLAlchemy-compatible async database URL.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    async def write_ohlcv(self, candles: list[OHLCV], symbol: str) -> None:
        """Write a batch of OHLCV candles to the database.

        Args:
            candles: List of candles to persist.
            symbol: Trading pair symbol.
        """
        raise NotImplementedError

    async def write_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Write a complete market snapshot to the database.

        Args:
            snapshot: Snapshot including OHLCV and indicators.
        """
        raise NotImplementedError
