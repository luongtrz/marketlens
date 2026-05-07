"""Read articles from Supabase news_articles table via REST API."""

from __future__ import annotations

from shared.http_client import get_client
from shared.supabase_news import supabase_row_to_ingestion


_TABLE = "news_articles"

# Map ticker/symbol → keywords to search in article header
_KEYWORDS: dict[str, list[str]] = {
    "BTC":     ["bitcoin", "btc"],
    "BTCUSDT": ["bitcoin", "btc"],
    "ETH":     ["ethereum", "eth"],
    "ETHUSDT": ["ethereum", "eth"],
    "SOL":     ["solana", "sol"],
    "SOLUSDT": ["solana", "sol"],
    "BNB":     ["bnb", "binance coin"],
    "BNBUSDT": ["bnb", "binance coin"],
    "XRP":     ["xrp", "ripple"],
    "XRPUSDT": ["xrp", "ripple"],
    "ADA":     ["cardano", "ada"],
    "ADAUSDT": ["cardano", "ada"],
    "DOGE":    ["dogecoin", "doge"],
    "DOGEUSDT":["dogecoin", "doge"],
}

_CRYPTO_GENERIC = ["crypto", "bitcoin", "blockchain", "defi", "web3"]


class SupabaseReader:
    def __init__(self, supabase_url: str, service_key: str, anon_key: str) -> None:
        self._base = supabase_url.rstrip("/")
        self._headers = {
            "apikey": anon_key,
            "Authorization": f"Bearer {service_key or anon_key}",
        }

    async def get_latest(self, symbol: str, limit: int = 20) -> list[IngestionRecord]:
        keywords = _KEYWORDS.get(symbol.upper(), _CRYPTO_GENERIC)
        or_filter = ",".join(f"header.ilike.*{kw}*" for kw in keywords)
        params = {
            "or": f"({or_filter})",
            "order": "publish_at.desc",
            "limit": str(limit),
        }
        url = f"{self._base}/rest/v1/{_TABLE}"
        async with get_client() as client:
            resp = await client.get(url, headers=self._headers, params=params)
            resp.raise_for_status()
            rows = resp.json()
        return [supabase_row_to_ingestion(r) for r in rows if isinstance(r, dict)]
