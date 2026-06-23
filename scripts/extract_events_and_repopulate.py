"""Extract event states and repopulate optimizer dataset (offline, no DB/API).

WHAT THIS SCRIPT DOES
---------------------
Loads all records from stockmem/data/real_optimizer_v2.json and, for each
record, populates an `event_state` field derived from the pre-computed
`event_vec` already stored in that file.

WHY NOT LOAD FROM SUPABASE
---------------------------
`real_optimizer_v2.json` is the *flattened* optimizer dataset — it contains
pre-computed vector embeddings (event_vec, factor_vec, indicator_vec,
price_vec) but NOT the raw `StockMemRecord` payload (factors, normalized_factors,
article_sources, etc.).

If you need to re-derive event_state from raw factor strings, use:
    scripts/compute_event_states.py
which connects to local PostgreSQL and reads the full `stockmem_records` table
(requires: postgres running locally with the dataset loaded).

WHAT `compute_event_states.py` DOES (summary)
----------------------------------------------
1. Connects to local PG via asyncpg.
2. Loads all BTC records sorted by date.
3. For each record i, calls build_daily_event_state(record, history[:i])
   which derives EventRecord objects from record.factors via taxonomy lookup.
4. Writes the computed event_state back into the payload column.
5. Can be re-run after stockmem DB is populated via push_stockmem_to_supabase.py.

REGEN OPTIMIZER COMMAND (after compute_event_states.py has run)
----------------------------------------------------------------
To rebuild real_optimizer_v2.json from scratch (requires DB access):

    PYTHONPATH=/home/luong/marketlens \\
        python stockmem/scripts/regen_optimizer_data.py \\
        --output stockmem/data/real_optimizer_v2.json

This re-reads all stockmem_records rows, embeds each with RecordEmbedder
(which calls build_event_vector(record.event_state) to produce event_vec),
and writes the new JSON. Running compute_event_states.py first ensures
event_state is populated in the DB before regen runs.

OFFLINE PROCESSING (this script)
----------------------------------
Since the optimizer JSON already contains event_vec (85-dim), this script:
1. Reads each row's event_vec.
2. Reconstructs a minimal DailyEventState from the scalar slots in event_vec
   (indices NUM_TYPES+NUM_GROUPS .. end) — see build_event_vector() for layout.
3. Writes real_optimizer_v2_events.json with an added `event_state` field.

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/extract_events_and_repopulate.py
    PYTHONPATH=/home/luong/marketlens python scripts/extract_events_and_repopulate.py \\
        --input  stockmem/data/real_optimizer_v2.json \\
        --output stockmem/data/real_optimizer_v2_events.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "stockmem" / "data" / "real_optimizer_v2.json"
DEFAULT_OUTPUT = ROOT / "stockmem" / "data" / "real_optimizer_v2_events.json"


# ---------------------------------------------------------------------------
# Event-vec layout constants (must match event_memory.build_event_vector)
# ---------------------------------------------------------------------------

def _load_taxonomy_dims() -> tuple[int, int]:
    """Return (NUM_TYPES, NUM_GROUPS) from taxonomy module."""
    from stockmem.src.search.taxonomy import NUM_TYPES, NUM_GROUPS  # type: ignore[import]
    return NUM_TYPES, NUM_GROUPS


# ---------------------------------------------------------------------------
# Reconstruct minimal DailyEventState from pre-computed event_vec
# ---------------------------------------------------------------------------

def _event_state_from_vec(
    event_vec: list[float],
    record_date: str,
    num_types: int,
    num_groups: int,
    type_names: list[str],
    group_names: list[str],
) -> dict:
    """
    Reconstruct a partial DailyEventState from a pre-computed 85-dim event_vec.

    Layout (from build_event_vector):
        [0 .. num_types)              → one-hot event types
        [num_types .. num_types+num_groups) → one-hot event groups
        scalar[0]  = mean_polarity
        scalar[1]  = max_abs_polarity
        scalar[2]  = log-scaled article_count proxy
        scalar[3]  = log-scaled source_count proxy
        scalar[4]  = source_diversity
        scalar[5]  = novelty_7d
        scalar[6]  = novelty_30d
        scalar[7]  = mean_confidence
        scalar[8]  = temporal_span_hours proxy
        scalar[9]  = incremental_information

    Returns a dict compatible with DailyEventState.model_dump().
    """
    import math

    scalar_offset = num_types + num_groups
    scalars = event_vec[scalar_offset : scalar_offset + 10]

    # Recover active event types / groups from one-hot bits.
    # The bits may be L2-normalized (e.g. ~0.27 for 5 active types),
    # so accept any value strictly above a small epsilon.
    _EPS = 1e-4
    active_types = [
        type_names[i] for i in range(num_types) if event_vec[i] > _EPS
    ]
    active_groups: list[str] = [
        group_names[i] for i in range(num_groups)
        if event_vec[num_types + i] > _EPS
    ]

    # Build EventRecord stubs — one per active type, matched to first active group
    events: list[dict] = []
    mean_polarity = scalars[0] if len(scalars) > 0 else 0.0
    mean_confidence = scalars[7] if len(scalars) > 7 else 0.6

    for i, et in enumerate(active_types):
        eg = active_groups[i] if i < len(active_groups) else (active_groups[0] if active_groups else "Unknown")
        events.append({
            "event_group": eg,
            "event_type": et,
            "polarity": float(mean_polarity),
            "confidence": float(mean_confidence),
            "observed_at": None,
            "description": None,
            "entities": [],
        })

    # Recover scalar fields (inverse of the log-scale compression where possible)
    article_count_proxy = scalars[2] if len(scalars) > 2 else 0.0
    source_count_proxy = scalars[3] if len(scalars) > 3 else 0.0
    # log1p(x)/log(51) = v  =>  x = exp(v*log(51)) - 1
    article_count = max(0, int(round(math.exp(article_count_proxy * math.log(51)) - 1)))
    source_count  = max(0, int(round(math.exp(source_count_proxy  * math.log(21)) - 1)))

    temporal_hours_proxy = scalars[8] if len(scalars) > 8 else 0.0
    temporal_span_hours = max(0.0, float(temporal_hours_proxy) * 168.0)

    try:
        parsed_date = date.fromisoformat(record_date)
    except Exception:
        parsed_date = date.today()

    return {
        "date": record_date,
        "symbol": "BTC",
        "events": events,
        "article_count": article_count,
        "source_count": source_count,
        "source_diversity": float(scalars[4]) if len(scalars) > 4 else 0.0,
        "temporal_span_hours": temporal_span_hours,
        "novelty_7d": float(scalars[5]) if len(scalars) > 5 else 0.0,
        "novelty_30d": float(scalars[6]) if len(scalars) > 6 else 0.0,
        "incremental_information": float(scalars[9]) if len(scalars) > 9 else 0.0,
        "dominant_event_groups": active_groups[:3],
    }


def main(input_path: Path, output_path: Path) -> None:
    # ------------------------------------------------------------------
    # Load taxonomy metadata
    # ------------------------------------------------------------------
    try:
        from stockmem.src.search.taxonomy import (  # type: ignore[import]
            NUM_TYPES,
            NUM_GROUPS,
            TYPE_INDEX,
            GROUP_INDEX,
        )
        # Invert index → name list
        type_names: list[str] = [""] * NUM_TYPES
        for name, idx in TYPE_INDEX.items():
            type_names[idx] = name
        group_names: list[str] = [""] * NUM_GROUPS
        for name, idx in GROUP_INDEX.items():
            group_names[idx] = name
        log.info("Taxonomy: %d types, %d groups", NUM_TYPES, NUM_GROUPS)
    except Exception as exc:
        log.error("Cannot import taxonomy: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load input JSON
    # ------------------------------------------------------------------
    log.info("Loading %s ...", input_path)
    try:
        rows: list[dict] = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Cannot load input: %s", exc)
        sys.exit(1)

    total = len(rows)
    log.info("Loaded %d records", total)

    # ------------------------------------------------------------------
    # Process each record
    # ------------------------------------------------------------------
    with_event_state = 0
    zero_event_vec   = 0
    errors           = 0

    out_rows: list[dict] = []

    for i, row in enumerate(rows):
        if i % 100 == 0 and i > 0:
            log.info("  [%d/%d] with_event_state=%d zero_vec=%d errors=%d",
                     i, total, with_event_state, zero_event_vec, errors)

        event_vec: list[float] = row.get("event_vec", [])
        record_date: str = row.get("date", "")

        try:
            has_signal = any(v != 0.0 for v in event_vec)

            if not has_signal:
                zero_event_vec += 1
                event_state = None
            else:
                event_state = _event_state_from_vec(
                    event_vec,
                    record_date,
                    NUM_TYPES,
                    NUM_GROUPS,
                    type_names,
                    group_names,
                )
                with_event_state += 1

            out_row = dict(row)
            out_row["event_state"] = event_state
            out_rows.append(out_row)

        except Exception as exc:
            log.warning("  [%s] error: %s", record_date, exc)
            errors += 1
            out_rows.append(dict(row))  # pass through unchanged

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(out_rows, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("DONE")
    log.info("  Total records           : %d", total)
    log.info("  Records with event_state: %d  (%.1f%%)",
             with_event_state, 100.0 * with_event_state / max(total, 1))
    log.info("  Records with zero vec   : %d  (event_state=null)", zero_event_vec)
    log.info("  Errors (skipped)        : %d", errors)
    log.info("  Output                  : %s", output_path)
    log.info("=" * 60)
    log.info("")
    log.info("NOTE: event_state fields are reconstructed from pre-computed event_vec.")
    log.info("For full event_state with article metadata (sources, temporal_span, etc.),")
    log.info("run:  python scripts/compute_event_states.py  (requires local PostgreSQL).")
    log.info("Then: python stockmem/scripts/regen_optimizer_data.py \\")
    log.info("          --output stockmem/data/real_optimizer_v2.json")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Populate event_state in optimizer JSON from pre-computed event_vec (offline).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input optimizer JSON (default: stockmem/data/real_optimizer_v2.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON with event_state added (default: stockmem/data/real_optimizer_v2_events.json)",
    )
    args = parser.parse_args()
    main(args.input, args.output)


if __name__ == "__main__":
    cli()
