#!/usr/bin/env python3
"""Import local stockmem_records NDJSON into a PostgreSQL Docker container.

Expected input format:
  one JSON object per line with keys: id, record_date, symbol, payload

Example:
  python3 scripts/import_stockmem_records_to_postgres.py
  python3 scripts/import_stockmem_records_to_postgres.py \
    --input data/exports/stockmem_records.ndjson \
    --container marketlens-stockmem-db
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "exports" / "stockmem_records.ndjson"
DEFAULT_CONTAINER = "marketlens-stockmem-db"
DEFAULT_DB = "stockmem"
DEFAULT_USER = "postgres"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stockmem_records (
    id TEXT PRIMARY KEY,
    record_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE (record_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_stockmem_records_symbol_date
ON stockmem_records (symbol, record_date);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to NDJSON export.")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help="Docker container name.")
    parser.add_argument("--db", default=DEFAULT_DB, help="PostgreSQL database name.")
    parser.add_argument("--user", default=DEFAULT_USER, help="PostgreSQL user.")
    return parser.parse_args()


def run_psql(container: str, user: str, db: str, sql: str) -> None:
    cmd = [
        "docker", "exec", "-i", container,
        "psql", "-v", "ON_ERROR_STOP=1", "-U", user, "-d", db, "-c", sql,
    ]
    subprocess.run(cmd, check=True)


def copy_file_to_container(container: str, source: Path, target: str) -> None:
    subprocess.run(["docker", "cp", str(source), f"{container}:{target}"], check=True)


def run_psql_file(container: str, user: str, db: str, source: Path, target: str) -> None:
    copy_file_to_container(container, source, target)
    cmd = [
        "docker", "exec", "-i", container,
        "psql", "-v", "ON_ERROR_STOP=1", "-U", user, "-d", db, "-f", target,
    ]
    subprocess.run(cmd, check=True)


def load_rows_to_tsv(input_path: Path, tsv_path: Path) -> int:
    total = 0
    with input_path.open("r", encoding="utf-8") as src, tsv_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            payload = json.dumps(row["payload"], ensure_ascii=True)
            fields = [
                str(row["id"]),
                str(row["record_date"]),
                str(row["symbol"]).upper(),
                payload,
            ]
            escaped = [
                field.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
                for field in fields
            ]
            dst.write("\t".join(escaped))
            dst.write("\n")
            total += 1
    return total


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    run_psql(args.container, args.user, args.db, CREATE_TABLE_SQL)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tsv", delete=False) as tmp:
        tsv_path = Path(tmp.name)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sql", delete=False) as tmp_sql:
        sql_path = Path(tmp_sql.name)

    try:
        total = load_rows_to_tsv(input_path, tsv_path)
        container_tsv = "/tmp/stockmem_records.tsv"
        copy_file_to_container(args.container, tsv_path, container_tsv)

        copy_sql = f"""
CREATE TEMP TABLE stockmem_records_stage (
    id TEXT,
    record_date TEXT,
    symbol TEXT,
    payload TEXT
);
\\copy stockmem_records_stage (id, record_date, symbol, payload) FROM '{container_tsv}' WITH (FORMAT text);
INSERT INTO stockmem_records (id, record_date, symbol, payload)
SELECT id, record_date, symbol, payload
FROM stockmem_records_stage
ON CONFLICT (record_date, symbol) DO UPDATE
SET id = EXCLUDED.id,
    payload = EXCLUDED.payload;
DROP TABLE stockmem_records_stage;
"""
        sql_path.write_text(copy_sql, encoding="utf-8")
        run_psql_file(args.container, args.user, args.db, sql_path, "/tmp/stockmem_import.sql")
        run_psql(args.container, args.user, args.db, "SELECT COUNT(*) FROM stockmem_records;")
        print(f"done: {total} rows imported into PostgreSQL container {args.container}")
        return 0
    finally:
        try:
            tsv_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            sql_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
