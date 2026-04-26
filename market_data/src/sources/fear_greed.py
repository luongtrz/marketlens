"""Alternative.me Fear & Greed Index adapter."""

from shared.http_client import get_client


class FearGreedSourceError(Exception):
    """Raised when the Fear & Greed API returns an error or unusable response."""


class FearGreedSource:
    """Fetches the Crypto Fear & Greed Index from Alternative.me (free, no auth).

    Args:
        base_url: Alternative.me API base URL.
    """

    def __init__(self, base_url: str = "https://api.alternative.me") -> None:
        self._base_url = base_url.rstrip("/")

    async def fetch(self) -> int:
        """Fetch the current Fear & Greed index value.

        Returns:
            Integer in [0, 100] (0=Extreme Fear, 100=Extreme Greed).

        Raises:
            FearGreedSourceError: On HTTP error or malformed response.
        """
        url = f"{self._base_url}/fng/?limit=1"

        async with get_client() as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as exc:
                raise FearGreedSourceError(f"Fear & Greed HTTP error: {exc}") from exc

            try:
                body = resp.json()
                raw_value = int(body["data"][0]["value"])
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise FearGreedSourceError(f"Fear & Greed response malformed: {exc}") from exc

        return max(0, min(100, raw_value))
