"""LLM API wrapper — calls AIHub or direct LLM for article enrichment."""

import re

import httpx

from shared.models.article import RawArticle, EnrichedFields


class LLMClient:
    """Enriches articles with sentiment, summary, and factors via LLM.

    Calls AIHub /sentiment and /factors endpoints, or falls back to direct
    LLM API if AIHub is unavailable.

    Args:
        aihub_url: Base URL for the AIHub service.
        enable_summary: Whether to generate article summaries.
    """

    def __init__(self, aihub_url: str, enable_summary: bool = False) -> None:
        self._aihub_url = aihub_url
        self._enable_summary = enable_summary
        self._client = httpx.AsyncClient(timeout=20)

    async def enrich(self, article: RawArticle) -> EnrichedFields:
        """Enrich a raw article with LLM-derived fields.

        Args:
            article: The raw article to enrich.

        Returns:
            EnrichedFields containing sentiment_score, summary, and factors.
        """
        text = (article.text or article.title or "").strip()
        if not text:
            return EnrichedFields(sentiment_score=0.0, summary=None, factors=[])

        # Best-effort AIHub call; fallback to deterministic heuristic.
        try:
            resp = await self._client.post(
                f"{self._aihub_url.rstrip('/')}/analysis",
                json={"data": text[:4000]},
            )
            if 200 <= resp.status_code < 300:
                payload = resp.json()
                score = float(payload.get("sentiment_score", payload.get("score", 0.0)))
                factors = payload.get("factors") or []
                summary = payload.get("summary") if self._enable_summary else None
                return EnrichedFields(
                    sentiment_score=max(-1.0, min(1.0, score)),
                    summary=summary,
                    factors=[str(f).strip() for f in factors if str(f).strip()],
                )
        except Exception:
            pass

        score = self._heuristic_sentiment_score(text)
        factors = self._heuristic_factors(text)
        summary = await self.summarize(text) if self._enable_summary else None
        return EnrichedFields(sentiment_score=score, summary=summary, factors=factors)

    async def summarize(self, text: str) -> str | None:
        """Generate a summary of the given text via LLM.

        Args:
            text: Article body text.

        Returns:
            Summary string or None if summarization is disabled.
        """
        if not self._enable_summary:
            return None
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return None
        return cleaned[:320]

    def _heuristic_sentiment_score(self, text: str) -> float:
        lower = text.lower()
        bullish_hits = sum(1 for k in ("surge", "rally", "gain", "bullish", "soar") if k in lower)
        bearish_hits = sum(1 for k in ("drop", "fall", "bearish", "slump", "crash") if k in lower)
        raw = (bullish_hits - bearish_hits) / 5.0
        return max(-1.0, min(1.0, raw))

    def _heuristic_factors(self, text: str) -> list[str]:
        lower = text.lower()
        factors = []
        if "etf" in lower:
            factors.append("ETF-related development")
        if "regulat" in lower or "sec" in lower:
            factors.append("Regulatory update")
        if "hack" in lower or "exploit" in lower:
            factors.append("Security incident risk")
        if "fed" in lower or "interest rate" in lower:
            factors.append("Macro monetary policy")
        return factors
