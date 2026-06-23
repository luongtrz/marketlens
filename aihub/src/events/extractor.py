"""Event extraction: rule-based taxonomy lookup + optional LLM enhancement."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from shared.models.event import EventRecord
from stockmem.src.search.taxonomy import (
    get_factor_group,
    get_factor_sentiment,
    get_factor_type,
)

from aihub.src.events.schema import EventExtractionRequest, EventExtractionResponse

if TYPE_CHECKING:
    from aihub.src.llm.base import LLMClient

logger = logging.getLogger(__name__)

_POLARITY_MAP = {
    "bullish": 0.7,
    "bearish": -0.7,
    "neutral": 0.0,
}

_SYSTEM_PROMPT = "You are a crypto market analyst. Respond only with valid JSON."

_VALID_GROUPS = {
    "Regulation & Legal",
    "Macroeconomic",
    "Protocol & Product",
    "Market Performance",
    "Whale & On-chain",
    "Risk & Warning",
    "Industry Standards & Opinions",
    "Technology & Development",
    "Sentiment & Narrative",
    # Include all taxonomy groups so LLM output can be accepted
    "Exchange & Trading",
    "DeFi & Ecosystem",
    "Key Figures",
    "TradFi Crossover",
    "Partnership & Adoption",
}


def _dedup(events: list[EventRecord]) -> list[EventRecord]:
    """Deduplicate by (event_group, event_type), keeping first occurrence."""
    seen: set[tuple[str, str]] = set()
    result: list[EventRecord] = []
    for ev in events:
        key = (ev.event_group, ev.event_type)
        if key not in seen:
            seen.add(key)
            result.append(ev)
    return result


class EventExtractor:
    def __init__(self, llm: "LLMClient | None" = None) -> None:
        self._llm = llm

    async def extract(self, request: EventExtractionRequest) -> EventExtractionResponse:
        events = self._rule_based(request)

        if not events:
            events = self._keyword_fallback(request)

        method = "rule_based"
        if self._llm is not None and len(events) < 2:
            try:
                llm_events = await self._llm_extract(request)
                if llm_events:
                    merged = _dedup(events + llm_events)
                    events = merged
                    method = "llm"
            except Exception as exc:
                logger.warning("LLM event extraction failed, using rule-based only: %s", exc)

        return EventExtractionResponse(events=events, method=method)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Rule-based: taxonomy lookup via factor names
    # ------------------------------------------------------------------

    def _rule_based(self, request: EventExtractionRequest) -> list[EventRecord]:
        events: list[EventRecord] = []
        for factor in request.factors:
            event_type = get_factor_type(factor)
            event_group = get_factor_group(factor)
            if not event_group:
                # Skip factors that can't be mapped to any group
                continue
            if not event_type:
                # Use group as a fallback type label
                event_type = event_group

            sentiment = get_factor_sentiment(factor)
            polarity = _POLARITY_MAP.get(sentiment or "neutral", 0.0)

            events.append(EventRecord(
                event_group=event_group,
                event_type=event_type,
                polarity=polarity,
                confidence=0.6,
                observed_at=request.published_at,
            ))

        return _dedup(events)

    # ------------------------------------------------------------------
    # Keyword fallback: scan title when factors produce nothing
    # ------------------------------------------------------------------

    def _keyword_fallback(self, request: EventExtractionRequest) -> list[EventRecord]:
        title = request.title.lower()
        events: list[EventRecord] = []

        def _add(group: str, etype: str, polarity: float) -> None:
            events.append(EventRecord(
                event_group=group,
                event_type=etype,
                polarity=polarity,
                confidence=0.5,
                observed_at=request.published_at,
            ))

        # ETF
        if "etf" in title:
            if any(kw in title for kw in ("approval", "approved", "approves")):
                _add("Regulation & Legal", "ETF Approval", +0.8)
            elif any(kw in title for kw in ("reject", "rejected", "denied", "delay", "delayed")):
                _add("Regulation & Legal", "ETF Approval", -0.7)

        # SEC enforcement
        if "sec" in title and any(kw in title for kw in ("sues", "charges", "enforcement")):
            _add("Regulation & Legal", "Enforcement Action", -0.8)

        # Halving / supply
        if "halving" in title or "halved" in title:
            _add("Protocol & Product", "Supply Dynamics", +0.5)

        # Security breach
        if any(kw in title for kw in ("hack", "exploit", "breach")):
            _add("Risk & Warning", "Security Incident", -0.9)

        # Whale / accumulation
        if "whale" in title or "accumul" in title:
            _add("Whale & On-chain", "Whale Activity", +0.4)

        # Liquidations
        if "liquidat" in title:
            _add("Market Performance", "Liquidation Cascade", -0.6)

        # Macro / Fed
        if "interest rate" in title or "fed" in title or "fomc" in title:
            _add("Macroeconomic", "Interest Rate Decision", 0.0)

        return _dedup(events)

    # ------------------------------------------------------------------
    # LLM enhancement
    # ------------------------------------------------------------------

    async def _llm_extract(self, request: EventExtractionRequest) -> list[EventRecord]:
        summary_part = f"\nSummary: {request.summary}" if request.summary else ""
        prompt = (
            "Extract up to 3 structured events from this crypto news.\n\n"
            f"Title: {request.title}"
            f"{summary_part}\n\n"
            "For each event provide:\n"
            "- event_group: one of [Regulation & Legal, Macroeconomic, Protocol & Product, "
            "Market Performance, Whale & On-chain, Risk & Warning, Industry Standards & Opinions, "
            "Technology & Development, Exchange & Trading, DeFi & Ecosystem, Key Figures, "
            "TradFi Crossover, Partnership & Adoption]\n"
            "- event_type: one of [Acquisition, Adoption Metric Change, Alliance Formation, "
            "Analyst or Influencer Opinion, Audit or Certification, BTC Dominance Shift, "
            "Bond Signal, Commodity Correlation, Cross-chain Expansion, Custody Agreement, "
            "Developer Tooling, Development Milestone, Dollar Index Movement, ETF Flow, "
            "Ecosystem Integration, Enforcement Action, Exchange Insolvency, Executive Appointment, "
            "Fee or Gas Change, Founder Statement, Funding Round, Government Stance, "
            "Hash Rate Change, Industry Report, Inflation Data, Institutional Adoption, "
            "Institutional View, Interest Rate Decision, International Sanctions or Bans, "
            "Legal Action Against Individual, Legislation Progress, Liquidation Event, "
            "Listing or Delisting, Market Cap Milestone, Miner Selling, New Feature Launch, "
            "Node or Validator Update, On-chain Flow Anomaly, Partnership Deal, "
            "Payment Integration, Protocol Launch, Protocol Migration, Protocol Proposal, "
            "Protocol Upgrade, Quantitative Easing or Tightening, Regulatory Announcement, "
            "Regulatory Risk, Reserve Proof, Revenue Report, Rug Pull or Scam, Sector Rotation, "
            "Security Breach or Hack, Stablecoin Flow, Stock Correlation, Strategic Partnership, "
            "Supply Dynamics, Systemic Risk, Technical Breakthrough, Testnet or Mainnet Launch, "
            "Volume Surge, Whale Accumulation, Whale Distribution]\n"
            "- polarity: float -1 to 1\n"
            "- confidence: float 0 to 1\n"
            "- entities: list of strings\n\n"
            'Respond with JSON: {"events": [...]}'
        )
        data = await self._llm.generate_json(prompt, system=_SYSTEM_PROMPT)  # type: ignore[union-attr]
        raw_events = data.get("events", [])

        events: list[EventRecord] = []
        for ev in raw_events:
            try:
                group = str(ev.get("event_group", ""))
                etype = str(ev.get("event_type", ""))
                if not group or not etype:
                    continue
                polarity = float(ev.get("polarity", 0.0))
                confidence = float(ev.get("confidence", 0.5))
                entities = [str(e) for e in ev.get("entities", [])]
                events.append(EventRecord(
                    event_group=group,
                    event_type=etype,
                    polarity=max(-1.0, min(1.0, polarity)),
                    confidence=max(0.0, min(1.0, confidence)),
                    entities=entities,
                    observed_at=request.published_at,
                ))
            except Exception as exc:
                logger.debug("Skipping malformed LLM event %s: %s", ev, exc)

        return events
