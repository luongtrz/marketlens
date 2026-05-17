"""Client exception hierarchy for main_controller HTTP clients."""


class ClientError(Exception):
    """Base class for all HTTP client errors."""


class CrawlerClientError(ClientError):
    """Raised when Crawler service call fails."""


class MarketClientError(ClientError):
    """Raised when MarketData service call fails."""


class AIHubClientError(ClientError):
    """Raised when AIHub service call fails."""


class StockMemClientError(ClientError):
    """Raised when StockMem service call fails."""


class FactorLedgeClientError(ClientError):
    """Raised when FactorLedge service call fails."""


class LLMGatewayClientError(ClientError):
    """Raised when LLM Gateway service call fails."""
