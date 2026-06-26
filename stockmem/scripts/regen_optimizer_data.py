"""Regenerate optimizer/training data from StockMem records.

Supports either PostgreSQL or a local NDJSON export of stockmem_records.
Can rebuild indicator vectors using raw sentiment_score, finbert sentiment,
or finbert with fallback to sentiment_score.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "stockmem" / "data" / "real_optimizer.json"
DEFAULT_NDJSON = ROOT / "data" / "exports" / "stockmem_records.ndjson"

DB_URL = os.getenv(
    "STOCKMEM_DB_URL",
    "postgresql+asyncpg://postgres:pass@localhost:5432/postgres",
)

SentimentSource = Literal["sentiment_score", "finbert", "auto"]


def _output_row(rec, split) -> dict[str, object]:
    return {
        "date": str(rec.date),
        "event_vec": split.event_vec.tolist(),
        "factor_vec": split.factor_vec.tolist(),
        "indicator_vec": split.indicator_vec.tolist(),
        "price_vec": split.price_vec.tolist(),
        "future_return_1d": rec.future_return_1d or 0.0,
        "future_return_3d": rec.future_return_3d or 0.0,
        "future_return_7d": rec.future_return_7d or 0.0,
        "future_return_15d": rec.future_return_15d or 0.0,
        "future_return_30d": rec.future_return_30d or 0.0,
    }


def _warn_zero_vectors(rec, split) -> None:
    if split.factor_vec.sum() == 0:
        print(f"  warn: zero factor_vec for {rec.date} (factors={rec.factors[:3]})")
    if split.indicator_vec.sum() == 0:
        print(f"  warn: zero indicator_vec for {rec.date}")
    if split.price_vec.sum() == 0:
        print(f"  warn: zero price_vec for {rec.date} (candles={len(rec.market_snapshot.candles)})")


def _load_records_from_ndjson(ndjson_path: Path):
    from stockmem.src.models import StockMemRecord

    records: list[StockMemRecord] = []
    for line in ndjson_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        payload = row.get("payload", {})
        try:
            records.append(StockMemRecord.model_validate(payload))
        except Exception as exc:
            print(f"  skip malformed row from NDJSON: {exc}")
    return records


async def _load_records_from_db():
    import sys
    sys.path.insert(0, str(ROOT))

    import asyncpg
    from stockmem.src.models import StockMemRecord

    dsn = DB_URL.replace("postgresql+asyncpg://", "postgresql://")
    print(f"Connecting to {dsn.split('@')[-1]} ...")
    pool = await asyncpg.create_pool(dsn)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT payload FROM stockmem_records"
            " WHERE (payload::json->>'future_return_7d') IS NOT NULL"
            " ORDER BY record_date"
        )

    records: list[StockMemRecord] = []
    for r in rows:
        try:
            records.append(StockMemRecord.model_validate(json.loads(r["payload"])))
        except Exception as e:
            print(f"  skip malformed row: {e}")
    await pool.close()
    return records


async def main(
    output: Path,
    *,
    input_ndjson: Path | None,
    sentiment_source: SentimentSource,
) -> None:
    import sys
    sys.path.insert(0, str(ROOT))

    from stockmem.src.search.embedder import RecordEmbedder

    if input_ndjson is not None:
        print(f"Loading records from {input_ndjson} ...")
        records = _load_records_from_ndjson(input_ndjson)
    else:
        records = await _load_records_from_db()

    print(f"Parsed {len(records)} valid records")
    finbert_available = sum(
        1 for rec in records if rec.finbert_sentiment_score is not None
    )
    print(
        f"FinBERT sentiment available for {finbert_available}/{len(records)} records "
        f"(source={sentiment_source})"
    )

    embedder = RecordEmbedder(sentiment_source=sentiment_source)
    embedder.rebuild_corpus(records)

    out_rows = []
    skipped = 0
    for rec in records:
        try:
            split = embedder.embed_split(rec)
        except Exception as e:
            print(f"  embed failed for {rec.date}: {e}")
            skipped += 1
            continue

        _warn_zero_vectors(rec, split)
        out_rows.append(_output_row(rec, split))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_rows, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(out_rows)} rows to {output}  (skipped={skipped})")

    # Quick stats
    zero_factor = sum(1 for r in out_rows if all(v == 0 for v in r["factor_vec"]))
    zero_event = sum(1 for r in out_rows if all(v == 0 for v in r["event_vec"]))
    zero_ind = sum(1 for r in out_rows if all(v == 0 for v in r["indicator_vec"]))
    zero_price = sum(1 for r in out_rows if all(v == 0 for v in r["price_vec"]))
    print(f"  zero event_vec     : {zero_event}/{len(out_rows)}")
    print(f"  zero factor_vec    : {zero_factor}/{len(out_rows)}")
    print(f"  zero indicator_vec : {zero_ind}/{len(out_rows)}")
    print(f"  zero price_vec     : {zero_price}/{len(out_rows)}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--input-ndjson",
        default=None,
        help="Optional stockmem_records NDJSON export. If omitted, read from PostgreSQL.",
    )
    parser.add_argument(
        "--sentiment-source",
        choices=["sentiment_score", "finbert", "auto"],
        default="sentiment_score",
        help="Indicator sentiment source. 'finbert' falls back to sentiment_score when missing.",
    )
    args = parser.parse_args()
    ndjson_path = Path(args.input_ndjson) if args.input_ndjson else None
    asyncio.run(
        main(
            Path(args.output),
            input_ndjson=ndjson_path,
            sentiment_source=args.sentiment_source,
        )
    )


if __name__ == "__main__":
    cli()
