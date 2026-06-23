"""Tests for the event extraction module (aihub/src/events/)."""

from __future__ import annotations

import pytest

from aihub.src.events.extractor import EventExtractor
from aihub.src.events.schema import EventExtractionRequest


def _make_extractor() -> EventExtractor:
    return EventExtractor(llm=None)


# ---------------------------------------------------------------------------
# Rule-based extraction via taxonomy lookup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rule_based_extraction() -> None:
    """Known factor 'SEC Regulatory Crackdown' should produce a Regulation & Legal event."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="SEC takes action",
        # Use a factor that matches BEARISH_FACTORS in taxonomy
        factors=["SEC sues Binance and Coinbase"],
    )
    response = await extractor.extract(request)

    assert response.method == "rule_based"
    assert len(response.events) >= 1

    ev = response.events[0]
    assert ev.event_group == "Regulation & Legal"
    assert ev.event_type == "Enforcement Action"
    assert ev.polarity < 0  # bearish factor
    assert ev.confidence == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_rule_based_bullish_factor() -> None:
    """Bullish factor should produce positive polarity."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="Whale accumulation detected",
        factors=["Strong whale accumulation"],
    )
    response = await extractor.extract(request)

    assert len(response.events) >= 1
    ev = response.events[0]
    assert ev.event_group == "Whale & On-chain"
    assert ev.polarity > 0


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_keyword_etf_approval() -> None:
    """'SEC approves Bitcoin ETF' should trigger ETF Approval with positive polarity."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="SEC approves Bitcoin ETF",
        factors=[],
    )
    response = await extractor.extract(request)

    assert len(response.events) >= 1
    ev = response.events[0]
    assert ev.event_type == "ETF Approval"
    assert ev.event_group == "Regulation & Legal"
    assert ev.polarity > 0
    assert ev.confidence == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_keyword_etf_rejection() -> None:
    """'ETF rejected' should produce negative polarity."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="Bitcoin ETF rejected by SEC",
        factors=[],
    )
    response = await extractor.extract(request)

    assert len(response.events) >= 1
    etf_events = [ev for ev in response.events if ev.event_type == "ETF Approval"]
    assert etf_events, "Expected ETF Approval event"
    assert etf_events[0].polarity < 0


@pytest.mark.asyncio
async def test_keyword_hack() -> None:
    """'Major DeFi protocol hacked' should produce Security Incident with negative polarity."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="ETH",
        title="Major DeFi protocol hacked for $50M",
        factors=[],
    )
    response = await extractor.extract(request)

    assert len(response.events) >= 1
    ev = response.events[0]
    assert ev.event_type == "Security Incident"
    assert ev.event_group == "Risk & Warning"
    assert ev.polarity < 0


@pytest.mark.asyncio
async def test_keyword_halving() -> None:
    """'halving' keyword should produce Supply Dynamics event."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="Bitcoin halving expected next month",
        factors=[],
    )
    response = await extractor.extract(request)

    assert any(ev.event_type == "Supply Dynamics" for ev in response.events)


@pytest.mark.asyncio
async def test_keyword_fed() -> None:
    """'Fed' keyword should produce Interest Rate Decision event."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="Fed signals rate cut at next FOMC meeting",
        factors=[],
    )
    response = await extractor.extract(request)

    assert any(ev.event_type == "Interest Rate Decision" for ev in response.events)


# ---------------------------------------------------------------------------
# Empty / no-match input
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_input() -> None:
    """Unrelated title with no factors should return empty events without crashing."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="BTC price update",
        factors=[],
    )
    response = await extractor.extract(request)

    # Should not crash; may return 0 or more events
    assert response is not None
    assert isinstance(response.events, list)
    assert response.method in ("rule_based", "llm")


@pytest.mark.asyncio
async def test_deduplication() -> None:
    """Duplicate factors mapping to the same (group, type) should be deduplicated."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="Multiple whale signals",
        factors=[
            "Strong whale accumulation",
            "Strong whale accumulation",  # exact duplicate
        ],
    )
    response = await extractor.extract(request)

    # Should not have duplicate (group, type) pairs
    keys = [(ev.event_group, ev.event_type) for ev in response.events]
    assert len(keys) == len(set(keys)), "Duplicate events should be deduplicated"


@pytest.mark.asyncio
async def test_multiple_factors() -> None:
    """Multiple different factors should each produce their own events."""
    extractor = _make_extractor()
    request = EventExtractionRequest(
        symbol="BTC",
        title="Market update",
        factors=[
            "Strong whale accumulation",
            "CPI lower than expected",
        ],
    )
    response = await extractor.extract(request)

    assert len(response.events) >= 2
    groups = {ev.event_group for ev in response.events}
    assert "Whale & On-chain" in groups
    assert "Macroeconomic" in groups
