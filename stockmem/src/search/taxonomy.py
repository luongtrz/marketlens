"""
StockMem event taxonomy (Appendix A of the StockMem paper, adapted for crypto).

13 groups, 62 event types. Each factor string maps to exactly one event type;
multiple factors may consolidate to the same type.

Public API:
  - build_type_vector(factors)   -> list[int]  # 62d binary
  - build_group_vector(factors)  -> list[int]  # 13d binary
  - get_factor_type(factor)      -> str | None
  - get_factor_group(factor)     -> str | None
  - get_factor_sentiment(factor) -> "bullish" | "bearish" | "neutral" | None
  - BULLISH_FACTORS / BEARISH_FACTORS / NEUTRAL_FACTORS: lists for mock sampling
  - NUM_TYPES = 62, NUM_GROUPS = 13
"""
from __future__ import annotations

from typing import Literal


EVENT_TAXONOMY: dict[str, list[str]] = {
    "Regulation & Legal": [
        "Regulatory Announcement",
        "Enforcement Action",
        "Legislation Progress",
        "Government Stance",
        "International Sanctions or Bans",
    ],
    "Macroeconomic": [
        "Interest Rate Decision",
        "Inflation Data",
        "Dollar Index Movement",
        "Quantitative Easing or Tightening",
    ],
    "Industry Standards & Opinions": [
        "Protocol Proposal",
        "Industry Report",
        "Analyst or Influencer Opinion",
    ],
    "Protocol & Product": [
        "Protocol Upgrade",
        "New Feature Launch",
        "Testnet or Mainnet Launch",
        "Adoption Metric Change",
        "Fee or Gas Change",
        "Hash Rate Change",
        "Supply Dynamics",
    ],
    "Technology & Development": [
        "Technical Breakthrough",
        "Development Milestone",
        "Audit or Certification",
        "Node or Validator Update",
        "Ecosystem Integration",
        "Developer Tooling",
    ],
    "Exchange & Trading": [
        "Listing or Delisting",
        "Funding Round",
        "Revenue Report",
        "Acquisition",
        "Partnership Deal",
        "Custody Agreement",
        "Liquidation Event",
        "Reserve Proof",
    ],
    "DeFi & Ecosystem": [
        "Protocol Launch",
        "Protocol Migration",
        "Cross-chain Expansion",
    ],
    "Whale & On-chain": [
        "Whale Accumulation",
        "Whale Distribution",
        "On-chain Flow Anomaly",
        "Miner Selling",
    ],
    "Key Figures": [
        "Executive Appointment",
        "Founder Statement",
        "Legal Action Against Individual",
    ],
    "Market Performance": [
        "Market Cap Milestone",
        "Sector Rotation",
        "BTC Dominance Shift",
        "Volume Surge",
        "ETF Flow",
        "Institutional View",
    ],
    "TradFi Crossover": [
        "Stock Correlation",
        "Bond Signal",
        "Commodity Correlation",
        "Stablecoin Flow",
    ],
    "Partnership & Adoption": [
        "Strategic Partnership",
        "Payment Integration",
        "Institutional Adoption",
        "Alliance Formation",
    ],
    "Risk & Warning": [
        "Security Breach or Hack",
        "Rug Pull or Scam",
        "Regulatory Risk",
        "Systemic Risk",
        "Exchange Insolvency",
    ],
}


GROUPS: list[str] = list(EVENT_TAXONOMY.keys())
NUM_GROUPS: int = len(GROUPS)  # 13

ALL_TYPES: list[str] = [t for types in EVENT_TAXONOMY.values() for t in types]
NUM_TYPES: int = len(ALL_TYPES)  # 62

GROUP_INDEX: dict[str, int] = {g: i for i, g in enumerate(GROUPS)}
TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(ALL_TYPES)}
TYPE_TO_GROUP: dict[str, str] = {
    t: group for group, types in EVENT_TAXONOMY.items() for t in types
}


Sentiment = Literal["bullish", "bearish", "neutral"]


BULLISH_FACTORS: dict[str, str] = {
    "SEC reviewing new ETF approval": "Regulatory Announcement",
    "Strong whale accumulation": "Whale Accumulation",
    "CPI lower than expected": "Inflation Data",
    "Fed holds interest rate steady": "Interest Rate Decision",
    "BlackRock increases BTC holdings": "Institutional Adoption",
    "Major corporation accepts BTC payments": "Payment Integration",
    "Hash rate hits new all-time high": "Hash Rate Change",
    "Institutional adoption increasing": "Institutional Adoption",
    "Record ETF inflows": "ETF Flow",
    "Gold positively correlated with BTC": "Commodity Correlation",
    "Stablecoin inflows to exchanges rising": "Stablecoin Flow",
    "Partnership with major bank": "Strategic Partnership",
    "Successful protocol upgrade": "Protocol Upgrade",
    "Significant volume surge": "Volume Surge",
    "Developer activity surging": "Development Milestone",
    "Positive on-chain metrics": "On-chain Flow Anomaly",
    "Supply decreasing due to halving effect": "Supply Dynamics",
    "DXY dollar index declining": "Dollar Index Movement",
    "BTC dominance rising": "BTC Dominance Shift",
    "New payment integration": "Payment Integration",
    "Grayscale GBTC premium rising": "ETF Flow",
    "MicroStrategy buys more BTC": "Institutional Adoption",
    "El Salvador increases BTC reserves": "Government Stance",
    "Lightning Network adoption growing": "Adoption Metric Change",
    "DeFi TVL on Bitcoin rising": "Protocol Launch",
    "Fidelity opens BTC custody service": "Custody Agreement",
    "JP Morgan positive outlook on BTC": "Analyst or Influencer Opinion",
    "Hash rate recovering after sell-off": "Hash Rate Change",
    "Binance Proof of Reserve stable": "Reserve Proof",
    "Bitcoin spot volume hits record": "Volume Surge",
    "Mining difficulty adjustment decreasing": "Hash Rate Change",
    "Central bank record gold buying - bullish for BTC": "Commodity Correlation",
    "Nasdaq positively correlated with crypto": "Stock Correlation",
    "New Layer 2 scaling solution": "Technical Breakthrough",
    "Fed pivot signal - market expects rate cut": "Interest Rate Decision",
    "US Treasury yield falling - capital flows to risk assets": "Bond Signal",
    "Ordinals and BRC-20 adoption growing": "Adoption Metric Change",
    "Bitcoin ETF options approved": "Regulatory Announcement",
    "Tether minting USDT - liquidity flowing in": "Stablecoin Flow",
    "Coinbase revenue beats expectations": "Revenue Report",
}


BEARISH_FACTORS: dict[str, str] = {
    "SEC rejects new ETF": "Enforcement Action",
    "Strong whale selling": "Whale Distribution",
    "CPI higher than expected": "Inflation Data",
    "Fed raises interest rate": "Interest Rate Decision",
    "Regulatory concerns from China": "Government Stance",
    "Major exchange hack": "Security Breach or Hack",
    "Miner selling pressure increasing": "Miner Selling",
    "Large market liquidations": "Liquidation Event",
    "Significant ETF outflows": "ETF Flow",
    "Stablecoin outflows from exchanges": "Stablecoin Flow",
    "Regulatory risk from EU": "Regulatory Risk",
    "Exchange insolvency concerns": "Exchange Insolvency",
    "Extreme greed index - correction risk": "Institutional View",
    "Dollar index surging": "Dollar Index Movement",
    "Bond yield rising - risk for crypto": "Bond Signal",
    "Volume declining - liquidity drying up": "Volume Surge",
    "BTC dominance declining": "BTC Dominance Shift",
    "Systemic risk concerns": "Systemic Risk",
    "Major project rug pull": "Rug Pull or Scam",
    "Legal action against founder": "Legal Action Against Individual",
    "FTX exchange collapse event": "Exchange Insolvency",
    "Terra Luna collapse impact": "Systemic Risk",
    "Celsius Network freezes withdrawals": "Exchange Insolvency",
    "Genesis Trading halts operations": "Exchange Insolvency",
    "Three Arrows Capital bankruptcy": "Systemic Risk",
    "USDC temporary depeg": "Systemic Risk",
    "Mt. Gox distributing BTC to creditors": "Whale Distribution",
    "SEC sues Binance and Coinbase": "Enforcement Action",
    "Silvergate Bank closes": "Exchange Insolvency",
    "Tether FUD - reserve concerns": "Regulatory Risk",
    "Mining ban in Kazakhstan": "International Sanctions or Bans",
    "China tightens crypto regulations again": "International Sanctions or Bans",
    "Iran temporary mining ban": "International Sanctions or Bans",
    "US debt ceiling concerns": "Quantitative Easing or Tightening",
    "Grayscale GBTC discount widening": "ETF Flow",
    "Leverage ratio too high - cascade liquidation risk": "Liquidation Event",
    "Dormant BTC wallet suddenly moves funds": "On-chain Flow Anomaly",
    "SEC investigates staking services": "Enforcement Action",
    "CBDC competing with crypto": "Government Stance",
    "Whale sends large BTC to exchange": "Whale Distribution",
}


NEUTRAL_FACTORS: dict[str, str] = {
    "Market sideways waiting for signal": "Market Cap Milestone",
    "Analyst opinions divided": "Analyst or Influencer Opinion",
    "Neutral industry report": "Industry Report",
    "Protocol proposal under review": "Protocol Proposal",
    "Routine developer milestone": "Development Milestone",
    "Minor sector rotation": "Sector Rotation",
    "Market cap stable": "Market Cap Milestone",
    "Normal on-chain flow": "On-chain Flow Anomaly",
    "New testnet under testing": "Testnet or Mainnet Launch",
    "Industry report compilation": "Industry Report",
    "Consolidation phase - accumulation": "Market Cap Milestone",
    "Neutral funding rate": "Liquidation Event",
    "Open interest stable": "Liquidation Event",
    "Hashrate unchanged": "Hash Rate Change",
    "DXY sideways": "Dollar Index Movement",
    "No notable macro data": "Inflation Data",
    "Weekend options expiry": "Listing or Delisting",
    "Bitcoin Pizza Day - no price impact": "Market Cap Milestone",
    "Crypto conference in Europe": "Alliance Formation",
    "US Congress crypto hearing": "Legislation Progress",
}


FACTOR_TYPE_MAP: dict[str, str] = {
    **BULLISH_FACTORS,
    **BEARISH_FACTORS,
    **NEUTRAL_FACTORS,
}


_FACTOR_SENTIMENT: dict[str, Sentiment] = {}
for _f in BULLISH_FACTORS:
    _FACTOR_SENTIMENT[_f] = "bullish"
for _f in BEARISH_FACTORS:
    _FACTOR_SENTIMENT[_f] = "bearish"
for _f in NEUTRAL_FACTORS:
    _FACTOR_SENTIMENT[_f] = "neutral"

# Case-insensitive lookup (built once at import time for O(1) access)
_FACTOR_TYPE_MAP_LOWER: dict[str, str] = {k.lower(): v for k, v in FACTOR_TYPE_MAP.items()}

# Maps FactorType enum values (from AIHub) to taxonomy group names.
# Used when a factor name doesn't match the taxonomy but its type is known.
FACTOR_TYPE_TO_GROUPS: dict[str, list[str]] = {
    "macro":      ["Macroeconomic", "TradFi Crossover"],
    "regulatory": ["Regulation & Legal"],
    "technical":  ["Technology & Development", "Protocol & Product"],
    "sentiment":  ["Industry Standards & Opinions", "Market Performance"],
    "on_chain":   ["Whale & On-chain", "DeFi & Ecosystem"],
    "exchange":   ["Exchange & Trading"],
}


def _resolve_event_type(factor: str) -> str | None:
    """Try exact match then case-insensitive match against FACTOR_TYPE_MAP."""
    result = FACTOR_TYPE_MAP.get(factor)
    if result is not None:
        return result
    return _FACTOR_TYPE_MAP_LOWER.get(factor.lower())


def get_factor_type(factor: str) -> str | None:
    return _resolve_event_type(factor)


def get_factor_group(factor: str) -> str | None:
    event_type = _resolve_event_type(factor)
    if event_type is None:
        return None
    return TYPE_TO_GROUP.get(event_type)


def get_factor_sentiment(factor: str) -> Sentiment | None:
    return _FACTOR_SENTIMENT.get(factor)


def build_type_vector(factors: list[str]) -> list[int]:
    """StockMem formula (3): V_t[m] = 1 if event type m occurs."""
    vec = [0] * NUM_TYPES
    for f in factors:
        event_type = _resolve_event_type(f)
        if event_type is None:
            continue
        idx = TYPE_INDEX.get(event_type)
        if idx is not None:
            vec[idx] = 1
    return vec


def build_group_vector(factors: list[str]) -> list[int]:
    """StockMem formula (4): G_t[g] = 1 if any event in group g occurs."""
    vec = [0] * NUM_GROUPS
    for f in factors:
        group = get_factor_group(f)
        if group is None:
            continue
        idx = GROUP_INDEX.get(group)
        if idx is not None:
            vec[idx] = 1
    return vec


def build_group_vector_from_types(factor_types: list[str | None]) -> list[int]:
    """Build group vector from FactorType enum values (e.g. 'macro', 'regulatory').

    Used as fallback when factor names don't match the taxonomy — AIHub always
    provides a FactorType even for free-form factor names.
    """
    vec = [0] * NUM_GROUPS
    for ft in factor_types:
        if ft is None:
            continue
        groups = FACTOR_TYPE_TO_GROUPS.get(str(ft).lower(), [])
        for g in groups:
            idx = GROUP_INDEX.get(g)
            if idx is not None:
                vec[idx] = 1
    return vec
