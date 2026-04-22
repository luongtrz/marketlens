"""Text → factor list extractor using SKGP and supplementary heuristics."""

from shared.models.factor import Factor
from aihub.src.llm.base import LLMClient
from aihub.src.factors.skgp import SKGPExtractor


class FactorExtractor:
    """Orchestrates factor extraction from article text.

    Delegates to SKGPExtractor (LLM-based). Extend here to add
    rule-based heuristics or post-processing deduplication.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._skgp = SKGPExtractor(llm)

    async def extract(self, ticker: str, text: str) -> list[Factor]:
        """Extract factors from the given text.

        Args:
            ticker: Crypto ticker symbol (e.g. "BTC").
            text: Article text.

        Returns:
            Deduplicated list of Factor objects.
        """
        return await self._skgp.extract(ticker, text)
