#!/usr/bin/env python3
"""Test a PostgreSQL connection string.

Examples:
    SUPABASE_DB_PASSWORD='...' python scripts/test_postgres_connection.py
    python scripts/test_postgres_connection.py --dsn 'postgresql://user:pass@host:6543/postgres?sslmode=require'
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from urllib.parse import quote


DEFAULT_DSN = (
    "postgresql://postgres.esctepjpgpjgrcymnabx:YOUR_PASSWORD"
    "@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require"
)


def build_dsn(dsn: str, prompt_password: bool) -> str:
    if "YOUR_PASSWORD" not in dsn:
        return dsn

    password = os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("PGPASSWORD")
    if not password and prompt_password:
        password = getpass.getpass("Database password: ")

    if not password:
        raise ValueError(
            "Connection string still contains YOUR_PASSWORD. Set SUPABASE_DB_PASSWORD "
            "or pass a complete --dsn."
        )

    return dsn.replace("YOUR_PASSWORD", quote(password, safe=""))


async def test_connection(dsn: str) -> None:
    try:
        import asyncpg
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: asyncpg. Install project dependencies or run "
            "`python -m pip install asyncpg`."
        ) from exc

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            "select current_database() as database, current_user as user, "
            "inet_server_addr()::text as server_addr, version() as version"
        )
    finally:
        await conn.close()

    if row is None:
        raise RuntimeError("Connection succeeded, but the test query returned no rows.")

    print("OK: connected to PostgreSQL")
    print(f"database: {row['database']}")
    print(f"user: {row['user']}")
    print(f"server_addr: {row['server_addr']}")
    print(f"version: {row['version'].splitlines()[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL") or DEFAULT_DSN,
        help="PostgreSQL connection string. Defaults to DATABASE_URL or the Supabase DSN.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt for a password if the DSN contains YOUR_PASSWORD.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        dsn = build_dsn(args.dsn, prompt_password=not args.no_prompt)
        asyncio.run(test_connection(dsn))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
