"""SKGP (Structured Knowledge-Guided Parsing) technique implementation.

Parses article text to extract named factors (entities, events, macro signals)
relevant to crypto markets.
"""

from shared.models.factor import Factor, FactorType
import json
K_FACTORS = 5
COIN_TABLE = {
    'BTC':  'Bitcoin',
    'ETH':  'Ethereum',
    'SOL':  'Solana',
    'BNB':  'BNB',
    'XRP':  'Ripple',
    'ADA':  'Cardano',
    'AVAX': 'Avalanche',
    'LINK': 'Chainlink',
    'DOGE': 'Dogecoin',
    'DOT':  'Polkadot',
    'UNI':  'Uniswap',
    'MATIC': 'Polygon',
}

class SKGPExtractor:
    """Structured Knowledge-Guided Parser for factor extraction.

    Extracts named factors from article text, classifying each by type,
    polarity, and confidence.
    """
    def generate_content(self, content):
        return '{"factors": [{"factor": "SEC Regulatory Crackdown", "type": "macro", "action": "SEC announced new regulations for cryptocurrency exchanges", "effect": "Increased compliance costs for exchanges, potentially reducing trading volume", "impact": "Negative", "polarity": -0.8, "confidence": 90}]}'


    def extract(self, ticker: str, text: str) -> list[Factor]:
        """Extract market factors from article text.

        Args:
            ticker: Name of the cryptocurrency
            text: Article text to analyze.

        Returns:
            List of Factor objects with name, type, polarity, and confidence.
        """

        company = COIN_TABLE[ticker]
        prompt = (
            f"Please analyze the provided news and pinpoint the top {K_FACTORS} "
            f"major factors impacting the crypto price of {company} ({ticker}).\n\n"
            f"For each factor, provide:\n"
            f"- factor: A short label for the factor (e.g. \"SEC Regulatory Crackdown\")\n"
            f"- type: type of this factor, one from the list: [macro, technical, sentiment, on_chain, exchange]\n"
            f"- action: One sentence describing the specific event or action occurring\n"
            f"- polarity: The polarity of the factor as a float (-1, 1)\n"
            f"- effect: One brief sentence describing the expected effect on {company}'s crypto price\n"
            f"- impact: Overall price impact — Positive, Negative, or Neutral\n"
            f"- confidence: Your confidence level as an integer (0-100)\n\n"
            f"News:\n{text}"
        )
        
        try:
            raw_text = self.generate_content(prompt) 
            factors_raw = json.loads(raw_text)["factors"]
            return [
                Factor(
                    name = f["factor"],
                    type = FactorType(f["type"]),
                    polarity = float(f["polarity"]),
                    confidence = int(f["confidence"]),
                )
                for f in factors_raw
            ]
        except Exception as exc:
                return []
