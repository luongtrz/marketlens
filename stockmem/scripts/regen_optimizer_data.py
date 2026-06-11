"""Regenerate stockmem/data/real_optimizer.json from live PostgreSQL records.

Reads all records with non-null future_return_7d, embeds them with the current
RecordEmbedder, and writes the JSON expected by optimize_weights.py.

Usage:
    PYTHONPATH=/home/luong/marketlens python stockmem/scripts/regen_optimizer_data.py
    PYTHONPATH=/home/luong/marketlens python stockmem/scripts/regen_optimizer_data.py --output stockmem/data/real_optimizer_v2.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "stockmem" / "data" / "real_optimizer.json"

DB_URL = os.getenv(
    "STOCKMEM_DB_URL",
    "postgresql+asyncpg://postgres:pass@localhost:5432/postgres",
)


async def main(output: Path) -> None:
    import sys
    sys.path.insert(0, str(ROOT))

    import asyncpg
    from stockmem.src.models import StockMemRecord
    from stockmem.src.search.embedder import RecordEmbedder

    dsn = DB_URL.replace("postgresql+asyncpg://", "postgresql://")
    print(f"Connecting to {dsn.split('@')[-1]} ...")
    pool = await asyncpg.create_pool(dsn)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT payload FROM stockmem_records"
            " WHERE (payload::json->>'future_return_7d') IS NOT NULL"
            " ORDER BY record_date"
        )

    print(f"Loaded {len(rows)} records with future_return_7d")

    records: list[StockMemRecord] = []
    for r in rows:
        try:
            records.append(StockMemRecord.model_validate(json.loads(r["payload"])))
        except Exception as e:
            print(f"  skip malformed row: {e}")

    print(f"Parsed {len(records)} valid records")

    embedder = RecordEmbedder()
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

        # Warn about zero vectors
        if split.factor_vec.sum() == 0:
            print(f"  warn: zero factor_vec for {rec.date} (factors={rec.factors[:3]})")
        if split.indicator_vec.sum() == 0:
            print(f"  warn: zero indicator_vec for {rec.date}")
        if split.price_vec.sum() == 0:
            print(f"  warn: zero price_vec for {rec.date} (candles={len(rec.market_snapshot.candles)})")

        out_rows.append({
            "date": str(rec.date),
            "event_vec": split.event_vec.tolist(),
            "factor_vec": split.factor_vec.tolist(),
            "indicator_vec": split.indicator_vec.tolist(),
            "price_vec": split.price_vec.tolist(),
            "future_return_1d": rec.future_return_1d or 0.0,
            "future_return_7d": rec.future_return_7d or 0.0,
            "future_return_30d": rec.future_return_30d or 0.0,
        })

    await pool.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_rows, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(out_rows)} rows to {output}  (skipped={skipped})")

    # Quick stats
    zero_factor = sum(1 for r in out_rows if all(v == 0 for v in r["factor_vec"]))
    zero_event = sum(1 for r in out_rows if all(v == 0 for v in r["event_vec"]))
    zero_ind = sum(1 for r in out_rows if all(v == 0 for v in r["indicator_vec"]))
    zero_price = sum(1 for r in out_rows if all(v == 0 for v in r["price_vec"]))
    print(f"  zero event_vec  : {zero_event}/{len(out_rows)}")
    print(f"  zero factor_vec : {zero_factor}/{len(out_rows)}")
    print(f"  zero indicator_vec: {zero_ind}/{len(out_rows)}")
    print(f"  zero price_vec    : {zero_price}/{len(out_rows)}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    asyncio.run(main(Path(args.output)))


if __name__ == "__main__":
    cli()
