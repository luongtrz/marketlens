"""SKGP (Structured Knowledge-Guided Parsing) — LLM-based factor extraction."""

import logging

from shared.models.factor import Factor, FactorType
from aihub.src.llm.base import LLMClient

logger = logging.getLogger(__name__)

K_FACTORS = 5

COIN_TABLE = {
    "BTC":   "Bitcoin",
    "ETH":   "Ethereum",
    "SOL":   "Solana",
    "BNB":   "BNB",
    "XRP":   "Ripple",
    "ADA":   "Cardano",
    "AVAX":  "Avalanche",
    "LINK":  "Chainlink",
    "DOGE":  "Dogecoin",
    "DOT":   "Polkadot",
    "UNI":   "Uniswap",
    "MATIC": "Polygon",
}

_SYSTEM = (
    "You are a quantitative crypto market analyst. "
    "Respond only with valid JSON — no markdown, no extra text."
)


class SKGPExtractor:
    """Structured Knowledge-Guided Parser for factor extraction.

    Calls the LLM to identify the top K market factors from an article,
    classifying each by type, polarity, and confidence.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract(self, ticker: str, text: str) -> list[Factor]:
        """Extract market factors from article text via LLM.

        Args:
            ticker: Crypto ticker symbol (e.g. "BTC").
            text: Article text to analyse.

        Returns:
            List of Factor objects; empty list on parse failure.
        """
        company = COIN_TABLE.get(ticker, ticker)
        prompt = (
            f"Analyse the news below and identify the top {K_FACTORS} factors "
            f"impacting the crypto price of {company} ({ticker}).\n\n"
            f"For each factor provide:\n"
            f"- factor: short label (e.g. \"SEC Regulatory Crackdown\")\n"
            f"- type: one of [macro, regulatory, technical, sentiment, on_chain, exchange]\n"
            f"- action: one sentence describing the specific event or action\n"
            f"- polarity: float from -1.0 (very bearish) to 1.0 (very bullish)\n"
            f"- effect: one sentence describing the expected effect on {company}'s price\n"
            f"- impact: \"Positive\", \"Negative\", or \"Neutral\"\n"
            f"- confidence: integer 0-100\n\n"
            f'Respond with JSON: {{"factors": [...]}}\n\n'
            f"News:\n{text}"
        )
        try:
            data = await self._llm.generate_json(prompt, system=_SYSTEM)
        except Exception as exc:
            logger.warning("SKGP LLM call failed: %s", exc)
            return []

        factors_raw = data.get("factors", [])
        results: list[Factor] = []
        for f in factors_raw:
            try:
                # Normalise type: "on-chain" → "on_chain", uppercase → lower
                raw_type = str(f.get("type", "macro")).lower().replace("-", "_").replace(" ", "_")
                results.append(Factor(
                    name=f["factor"],
                    type=FactorType(raw_type),
                    polarity=float(f["polarity"]),
                    confidence=float(f["confidence"]) / 100.0,
                ))
            except Exception as exc:
                logger.debug("Skipping malformed factor %s: %s", f, exc)
        return results
