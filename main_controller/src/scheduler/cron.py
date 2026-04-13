"""APScheduler-based cron jobs for scheduled pipeline runs."""


class PipelineCronScheduler:
    """Schedules pipeline runs using APScheduler.

    Args:
        pipeline: Pipeline instance to execute.
        default_symbol: Default trading pair for scheduled runs.
    """

    def __init__(self, pipeline=None, default_symbol: str = "BTCUSDT") -> None:
        self._pipeline = pipeline
        self._default_symbol = default_symbol

    def start(self) -> None:
        """Start the cron scheduler."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the cron scheduler."""
        raise NotImplementedError

    def add_job(self, symbol: str, cron_expression: str) -> str:
        """Add a scheduled pipeline run.

        Args:
            symbol: Trading pair.
            cron_expression: Cron schedule expression.

        Returns:
            Job ID.
        """
        raise NotImplementedError
