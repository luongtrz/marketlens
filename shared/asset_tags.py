"""Detect BTC / ETH relevance in news text and URLs (shared by crawler + API filters)."""

from __future__ import annotations

import re

# Whole-token asset names (avoids ``eth`` inside ``tether``, ``method``, …).
_BTC_RE = re.compile(r"\b(btc|bitcoin)\b", re.IGNORECASE)
_ETH_RE = re.compile(r"\b(eth|ethereum|ether)\b", re.IGNORECASE)

# Ecosystem terms that strongly imply Ethereum when whole-word / token appears in slug.
_ETH_ECOSYSTEM_RE = re.compile(
    r"\b("
    r"vitalik|buterin|erc-?20|erc-?721|beacon\s+chain|"
    r"mega\s*eth|megaeth|restaking|liquid\s+staking|"
    r"arbitrum|optimism|base\s+chain|linea|zksync|starknet|"
    r"uniswap|aave|lido|eigenlayer|"
    r"eth\s*2\.?0|eth2|ethereum\s+foundation|ethereum\s+upgrade|"
    r"proof[\s-]of[\s-]stake|pos\s+chain|smart\s+contract\s+platform"
    r")\b",
    re.IGNORECASE,
)

# URL path / slug hints (sitemap entries often lack body text until fetch).
_ETH_URL_RE = re.compile(
    r"(?:^|[/_-])(?:ethereum|eth(?:-|$)|tag/ethereum|tags/ethereum)(?:[/_-]|$)",
    re.IGNORECASE,
)
_BTC_URL_RE = re.compile(
    r"(?:^|[/_-])(?:bitcoin|btc(?:-|$)|tag/bitcoin|tags/bitcoin)(?:[/_-]|$)",
    re.IGNORECASE,
)


def detect_asset_tags(title: str, content: str = "", url: str = "") -> frozenset[str]:
    """Return ``{"BTC", "ETH"}`` subsets; empty when no asset signal."""

    text = f"{title or ''} {content or ''}".strip()
    tags: set[str] = set()

    if text:
        if _BTC_RE.search(text):
            tags.add("BTC")
        if _ETH_RE.search(text) or _ETH_ECOSYSTEM_RE.search(text):
            tags.add("ETH")

    if url:
        path = url.split("?", 1)[0].lower()
        if _BTC_URL_RE.search(path):
            tags.add("BTC")
        if _ETH_URL_RE.search(path):
            tags.add("ETH")

    return frozenset(tags)


def primary_asset_tag(title: str, content: str = "", url: str = "") -> str:
    """Legacy single tag for UI: BTC if both match (BTC checked first historically)."""

    tags = detect_asset_tags(title, content, url)
    if "BTC" in tags:
        return "BTC"
    if "ETH" in tags:
        return "ETH"
    return "General"


def text_matches_symbol(header: str, content: str, symbol: str) -> bool:
    """Trading-pair filter (``BTCUSDT``, ``ETHUSDT``, …) aligned with :func:`detect_asset_tags`."""

    sym = (symbol or "").upper().strip()
    if not sym:
        return True
    tags = detect_asset_tags(header, content, "")
    if "BTC" in sym:
        return "BTC" in tags
    if "ETH" in sym:
        return "ETH" in tags
    base = sym.replace("USDT", "").replace("USD", "").replace("BUSD", "")
    if base.isalpha() and 2 <= len(base) <= 8:
        return base in tags
    return False
