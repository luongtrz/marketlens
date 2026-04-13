"""Manual and event-driven pipeline triggers."""


class PipelineTrigger:
    """Handles manual and event-driven pipeline run triggers.

    Args:
        pipeline: Pipeline instance to execute.
    """

    def __init__(self, pipeline=None) -> None:
        self._pipeline = pipeline

    async def trigger_manual(self, symbol: str) -> str:
        """Manually trigger a pipeline run.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Run ID (UUID string).
        """
        raise NotImplementedError

    async def trigger_event(self, event: dict) -> str:
        """Trigger a pipeline run based on an external event.

        Args:
            event: Event payload.

        Returns:
            Run ID.
        """
        raise NotImplementedError
