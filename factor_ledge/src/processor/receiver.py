"""Accept raw factor lists from Crawler (via HTTP or message bus)."""

from pydantic import BaseModel, ConfigDict


class IngestRequest(BaseModel):
    """Request payload for factor ingestion."""

    model_config = ConfigDict(extra="ignore")

    article_id: str
    factors: list[str]
    source: str


class FactorReceiver:
    """Receives raw factor lists from the Crawler module.

    Can receive via HTTP POST or via MessageBus subscription.
    """

    async def receive(self, request: IngestRequest) -> list[str]:
        """Accept and validate raw factors from a crawler.

        Args:
            request: Ingest request with article_id, raw factors, and source.

        Returns:
            Validated raw factor list.
        """
        raise NotImplementedError
