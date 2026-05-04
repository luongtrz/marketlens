"""RAG context builder — assembles prompt context from current + similar records.

Includes:
  - StockMemClient: async HTTP client to retrieve similar records from StockMem.
  - record_to_text(): translates a StockMemRecord (including raw vectors)
    into human-readable text for LLM consumption.
  - RAGContextBuilder: orchestrates search + text assembly.
"""

from __future__ import annotations

from typing import Any

import httpx

from shared.models.memory import StockMemRecord, SimilarRecord


# ---------------------------------------------------------------------------
# Indicator-vec index labels (must stay in sync with
# stockmem/src/search/embedder.py  _extract_raw_numerical)
# ---------------------------------------------------------------------------
INDICATOR_LABELS: list[str] = [
    "MSI (Market Sentiment Index, 0-100)",
    "RSI (Relative Strength Index, 0-100)",
    "Sentiment Score (-1 to +1)",
    "Fear & Greed Index (0-100)",
    "Price Change %",
]


# ---------------------------------------------------------------------------
# StockMem HTTP client
# ---------------------------------------------------------------------------
class StockMemClient:
    """Async client for the StockMem similarity-search API."""

    def __init__(self, base_url: str = "http://localhost:8003") -> None:
        self._base_url = base_url.rstrip("/")

    async def search(
        self,
        query: StockMemRecord,
        k: int = 5,
        *,
        timeout: float = 10.0,
    ) -> list[SimilarRecord]:
        """Call POST /search on StockMem and return parsed SimilarRecord list.

        Args:
            query: The current-day record to use as query.
            k: Number of nearest neighbors to retrieve.
            timeout: HTTP request timeout in seconds.

        Returns:
            List of SimilarRecord ordered by descending similarity.
        """
        payload = {
            "query": query.model_dump(mode="json"),
            "k": k,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{self._base_url}/search",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[SimilarRecord] = []
        for item in data.get("results", []):
            results.append(SimilarRecord.model_validate(item))
        return results


# ---------------------------------------------------------------------------
# Record → human-readable text
# ---------------------------------------------------------------------------
def _format_indicator_vec(vec: list[float]) -> str:
    """Translate an indicator_vec into labelled lines.

    The vector produced by the StockMem embedder has 5 dimensions that are
    z-scored and L2-normalized.  Even in normalized form the sign and relative
    magnitude carry useful meaning for the LLM.
    """
    if not vec:
        return "  (no indicator vector available)"
    lines: list[str] = []
    for i, val in enumerate(vec):
        label = INDICATOR_LABELS[i] if i < len(INDICATOR_LABELS) else f"indicator[{i}]"
        lines.append(f"  {label}: {val:+.4f}")
    return "\n".join(lines)


def _format_future_returns(record: StockMemRecord) -> str | None:
    """Format known future-return labels (ground-truth outcomes)."""
    parts: list[str] = []
    if record.future_return_1d is not None:
        parts.append(f"1d={record.future_return_1d:+.2f}%")
    if record.future_return_7d is not None:
        parts.append(f"7d={record.future_return_7d:+.2f}%")
    if record.future_return_30d is not None:
        parts.append(f"30d={record.future_return_30d:+.2f}%")
    return "  ".join(parts) if parts else None


def _format_indicators(indicators: dict[str, Any]) -> str:
    """Format the indicators dict from MarketSnapshot into a readable string."""
    if not indicators:
        return ""
    parts: list[str] = []
    for key, value in indicators.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return "  ".join(parts)


def record_to_text(record: StockMemRecord, *, include_outcome: bool = False) -> str:
    """Convert a StockMemRecord into a human-readable text block.

    This is the canonical translator from structured record data to natural
    language that an LLM can understand.  It covers:

      - Identity (symbol, date)
      - Sentiment (label + score)
      - Factors (human-readable event descriptions)
      - Market snapshot (latest candle + indicators dict)
      - Indicator vector (z-scored market metrics with labels)
      - Summary
      - Optionally: known future returns (for historical cases)

    Args:
        record: The StockMemRecord to translate.
        include_outcome: If True, append future-return ground-truth labels.

    Returns:
        Multi-line human-readable string.
    """
    lines: list[str] = []

    # --- Identity ---
    lines.append(f"Symbol: {record.symbol}  |  Date: {record.date}")

    # --- Sentiment ---
    lines.append(
        f"Sentiment: {record.sentiment_label} (score={record.sentiment_score:+.2f})"
    )

    # --- Factors ---
    if record.factors:
        lines.append("Factors: " + "; ".join(record.factors))

    # --- Normalized factors (with type / weight) ---
    if record.normalized_factors:
        nf_parts = [
            f"{nf.name} [{nf.type.value}, w={nf.weight:.2f}, pol={nf.polarity:+.2f}]"
            for nf in record.normalized_factors
        ]
        lines.append("Normalized Factors: " + "; ".join(nf_parts))

    # --- Market Snapshot ---
    snap = record.market_snapshot
    ohlcv = snap.ohlcv
    lines.append(
        f"Price: close={ohlcv.close:.4f}  open={ohlcv.open:.4f}  "
        f"high={ohlcv.high:.4f}  low={ohlcv.low:.4f}  volume={ohlcv.volume:.2f}"
    )
    if snap.indicators:
        lines.append("Indicators: " + _format_indicators(snap.indicators))

    # --- Indicator vector (z-scored market metrics) ---
    if record.indicator_vec:
        lines.append("Market Index (z-scored, L2-normalized):")
        lines.append(_format_indicator_vec(record.indicator_vec))

    # --- Summary ---
    if record.summary:
        lines.append(f"Summary: {record.summary}")

    # --- Future returns (ground truth for historical records) ---
    if include_outcome:
        returns_str = _format_future_returns(record)
        if returns_str:
            lines.append(f"Actual Returns: {returns_str}")

    return "\n".join(lines)


# Groq free / on_demand tier enforces tight per-request input limits (~8k tokens for
# some models). Oversized prompts raise 413 / rate_limit_exceeded — clamp RAG aggressively.
_MAX_PREDICT_SIMILAR_COUNT = 3
_MAX_PREDICT_CHARS_CURRENT = 3600
_MAX_PREDICT_CHARS_PER_CASE = 1200
_MAX_PREDICT_CHARS_SIMILAR_TOTAL = 6000


def _clip_predict_context(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# RAG context builder
# ---------------------------------------------------------------------------
class RAGContextBuilder:
    """Builds RAG prompt context from current record and similar historical cases.

    Can optionally fetch similar records from StockMem if none are provided.
    """

    def __init__(
        self,
        stockmem_client: StockMemClient | None = None,
        default_k: int = 3,
    ) -> None:
        self._stockmem = stockmem_client
        self._default_k = default_k

    async def fetch_similar(
        self,
        current: StockMemRecord,
        k: int | None = None,
    ) -> list[SimilarRecord]:
        """Retrieve similar records from StockMem.

        Args:
            current: Query record for similarity search.
            k: Number of neighbors (defaults to ``self._default_k``).

        Returns:
            List of SimilarRecord. Empty list if no client configured or
            if StockMem retrieval fails.
        """
        if self._stockmem is None:
            return []
        try:
            eff = min(k or self._default_k, _MAX_PREDICT_SIMILAR_COUNT)
            return await self._stockmem.search(current, k=eff)
        except (httpx.RequestError, httpx.HTTPStatusError):
            return []

    async def build(
        self,
        current: StockMemRecord,
        similar: list[SimilarRecord] | None = None,
        *,
        k: int | None = None,
    ) -> tuple[str, str]:
        """Assemble prompt context strings.

        If ``similar`` is not provided (or empty), the builder will call
        StockMem to retrieve similar records automatically.

        Returns:
            Tuple of (current_text, similar_text).
        """
        eff_k = min(k or self._default_k, _MAX_PREDICT_SIMILAR_COUNT)
        similar_list = similar

        # Auto-fetch from StockMem when no similar records supplied
        if not similar_list:
            similar_list = await self.fetch_similar(current, k=eff_k)

        capped = list(similar_list or [])[:_MAX_PREDICT_SIMILAR_COUNT]

        current_txt = _clip_predict_context(
            record_to_text(current, include_outcome=False),
            _MAX_PREDICT_CHARS_CURRENT,
        )

        similar_lines: list[str] = []
        if capped:
            for i, case in enumerate(capped, 1):
                rec = case.record
                similar_lines.append(
                    f"Case {i}  similarity={case.similarity:.3f}  date={rec.date}"
                )
                body = _indent(record_to_text(rec, include_outcome=True), prefix="  ")
                similar_lines.append(_clip_predict_context(body, _MAX_PREDICT_CHARS_PER_CASE))
                if case.outcome:
                    similar_lines.append(f"  Outcome: {case.outcome}")
                similar_lines.append("")
        else:
            similar_lines.append("No similar historical cases found.")

        similar_txt = _clip_predict_context(
            "\n".join(similar_lines).strip(),
            _MAX_PREDICT_CHARS_SIMILAR_TOTAL,
        )
        return current_txt, similar_txt


def _indent(text: str, prefix: str = "  ") -> str:
    """Indent every line of *text* with *prefix*."""
    return "\n".join(prefix + line for line in text.splitlines())
