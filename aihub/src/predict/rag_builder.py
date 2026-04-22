"""RAG context builder — assembles prompt context from current + similar records."""

from shared.models.memory import StockMemRecord, SimilarRecord


class RAGContextBuilder:
    """Builds RAG prompt context from current record and similar historical cases."""

    def build(self, current: StockMemRecord, similar: list[SimilarRecord]) -> str:
        """Assemble a prompt context string.

        Format::

            === Current Situation ===
            Symbol: BTC  |  Date: 2024-01-15
            Sentiment: bullish (score=+0.72)
            Factors: SEC Approval; ETF Inflows
            Price: close=42000.00  open=41500.00  high=42800.00  low=41200.00  volume=...
            Indicators: rsi=62.3  macd=...

            === Similar Historical Cases ===
            Case 1  similarity=0.923  date=2023-10-26
              Sentiment: bullish (score=+0.65)
              Factors: ...
              Price: close=34500.00  volume=...
              Outcome: +8.4% over next 5 days

        Args:
            current: The current pipeline run's StockMemRecord.
            similar: List of similar historical records with similarity scores.

        Returns:
            Formatted context string for the prediction prompt.
        """
        lines: list[str] = ["=== Current Situation ==="]
        lines.append(f"Symbol: {current.symbol}  |  Date: {current.date}")
        lines.append(
            f"Sentiment: {current.sentiment_label} (score={current.sentiment_score:+.2f})"
        )
        if current.factors:
            lines.append("Factors: " + "; ".join(current.factors))
        if current.summary:
            lines.append(f"Summary: {current.summary}")

        snap = current.market_snapshot
        c = snap.ohlcv
        lines.append(
            f"Price: close={c.close:.4f}  open={c.open:.4f}  "
            f"high={c.high:.4f}  low={c.low:.4f}  volume={c.volume:.2f}"
        )
        if snap.indicators:
            lines.append(
                "Indicators: " + "  ".join(f"{k}={v}" for k, v in snap.indicators.items())
            )

        if similar:
            lines.append("\n=== Similar Historical Cases ===")
            for i, case in enumerate(similar, 1):
                rec = case.record
                lines.append(
                    f"\nCase {i}  similarity={case.similarity:.3f}  date={rec.date}"
                )
                lines.append(
                    f"  Sentiment: {rec.sentiment_label} (score={rec.sentiment_score:+.2f})"
                )
                if rec.factors:
                    lines.append("  Factors: " + "; ".join(rec.factors))
                sc = rec.market_snapshot.ohlcv
                lines.append(f"  Price: close={sc.close:.4f}  volume={sc.volume:.2f}")
                if case.outcome:
                    lines.append(f"  Outcome: {case.outcome}")

        return "\n".join(lines)
