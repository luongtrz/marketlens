"""Validate data readiness for StockMem backtest.

Usage:
    cd F:/DATN/marketlens_backtest/marketlens
    python ..myspace/scripts/check_backtest_data.py --symbol BTC --start 2026-01-01 --end 2026-05-10
    python ..myspace/scripts/check_backtest_data.py --symbol BTC --start 2025-01-01 --end 2025-12-31 --verbose

Prerequisites:
    Set environment variables or create a .env:
      SUPABASE_URL=https://xxxx.supabase.co
      SUPABASE_SERVICE_ROLE_KEY=xxx
      STOCKMEM_URL=http://localhost:8003  (default)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# -- Path Setup ----------------------------------------------------------------─
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Hardcode load .env from root (..myspace excluded from git)
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ[key] = val  # always set, override any existing
    print(f"[env] Loaded .env from {env_file}")


# -- Helpers --------------------------------------------------------------------
def _fmt_date(d: date) -> str:
    return d.isoformat()


def _fmt_pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


# -- Supabase Check ------------------------------------------------------------─
class SupabaseChecker:
    def __init__(self) -> None:
        import httpx
        self._http = httpx.AsyncClient(timeout=30.0)
        self._base = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        self._key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""

    @property
    def available(self) -> bool:
        return bool(self._base and self._key)

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
        }

    async def ping(self) -> bool:
        try:
            resp = await self._http.get(
                f"{self._base}/rest/v1/news_articles?select=id&limit=1",
                headers=self._headers(),
            )
            return 200 <= resp.status_code < 300
        except Exception:
            return False

    async def count_rows(self, table: str, filters: list[tuple[str, str]] | None = None) -> int | None:
        params: list[tuple[str, str]] = [("select", "id"), ("limit", "1")]
        if filters:
            params.extend(filters)
        headers = {**self._headers(), "Prefer": "count=exact"}
        try:
            resp = await self._http.get(
                f"{self._base}/rest/v1/{table}",
                headers=headers,
                params=params,
            )
            if 200 <= resp.status_code < 300:
                cr = resp.headers.get("content-range", "")
                if "/" in cr:
                    total_str = cr.rsplit("/", 1)[-1].strip()
                    if total_str.isdigit():
                        return int(total_str)
            return None
        except Exception:
            return None

    async def select(self, table: str, filters: list[tuple[str, str]] | None = None,
                     columns: str = "*", limit: int = 100, offset: int = 0,
                     order: str | None = None) -> list[dict]:
        params: list[tuple[str, str]] = [
            ("select", columns),
            ("limit", str(limit)),
        ]
        if offset > 0:
            params.append(("offset", str(offset)))
        if order:
            params.append(("order", order))
        if filters:
            params.extend(filters)
        try:
            resp = await self._http.get(
                f"{self._base}/rest/v1/{table}",
                headers=self._headers(),
                params=params,
            )
            if 200 <= resp.status_code < 300:
                data = resp.json()
                return data if isinstance(data, list) else []
            return []
        except Exception:
            return []

    async def close(self) -> None:
        await self._http.aclose()

    async def check_news_articles(self, symbol: str, start: date, end: date) -> dict:
        result: dict = {"table": "news_articles", "error": None, "gaps": [], "stats": {}}
        total = await self.count_rows("news_articles")
        result["total_rows"] = total

        # Count in date range
        in_range = await self.count_rows("news_articles", [
            ("publish_at", f"gte.{start.isoformat()}"),
            ("publish_at", f"lte.{end.isoformat()}"),
        ])
        result["in_range"] = in_range

        if in_range is None or in_range == 0:
            result["error"] = f"No articles found in range {start} -&gt; {end}"
            return result

        days_total = (end - start).days + 1

        # ── Scan ALL dates via pagination (each page = 1 day's worth, desc order) ──
        # We paginate through dates in reverse-chronological order to get full coverage
        seen_dates: set[str] = set()
        total_sentiment_articles = 0
        total_zero_sentiment = 0
        offset = 0
        page_size = 5000

        while True:
            page = await self.select(
                "news_articles",
                filters=[
                    ("publish_at", f"gte.{start.isoformat()}"),
                    ("publish_at", f"lte.{end.isoformat()}"),
                ],
                columns="publish_at,sentiment_score",
                limit=page_size,
                offset=offset,
                order="publish_at.desc",
            )
            if not page:
                break

            for r in page:
                pub = r.get("publish_at", "")
                if pub:
                    d = pub[:10]
                    seen_dates.add(d)
                sc = float(r.get("sentiment_score") or 0)
                if sc == 0.0:
                    total_zero_sentiment += 1
                else:
                    total_sentiment_articles += 1

            if len(page) < page_size:
                break
            offset += page_size

        days_with_data = len(seen_dates)
        result["stats"]["days_covered"] = days_with_data
        result["stats"]["days_total"] = days_total
        result["stats"]["coverage_pct"] = _fmt_pct(days_with_data, days_total)

        total_scored = total_sentiment_articles + total_zero_sentiment
        result["stats"]["articles_with_sentiment"] = total_sentiment_articles
        result["stats"]["articles_missing_sentiment"] = total_zero_sentiment
        result["stats"]["sentiment_quality"] = _fmt_pct(total_sentiment_articles, total_scored) if total_scored else "N/A"

        # Earliest & latest date seen
        if seen_dates:
            result["stats"]["first_date"] = min(seen_dates)
            result["stats"]["last_date"] = max(seen_dates)

        # List missing dates
        all_dates = {_fmt_date(start + timedelta(days=i)) for i in range(days_total)}
        missing_dates = sorted(all_dates - seen_dates)
        if missing_dates:
            result["gaps"] = missing_dates[:30]
            result["gap_count"] = len(missing_dates)

        return result

    async def check_daily_factor_snapshots(self, symbol: str, start: date, end: date) -> dict:
        result: dict = {"table": "daily_factor_snapshots", "error": None, "gaps": [], "stats": {}}
        total = await self.count_rows("daily_factor_snapshots")
        result["total_rows"] = total

        in_range = await self.count_rows("daily_factor_snapshots", [
            ("symbol", f"eq.{symbol.upper()}"),
            ("snapshot_date", f"gte.{start.isoformat()}"),
            ("snapshot_date", f"lte.{end.isoformat()}"),
        ])
        result["in_range"] = in_range

        if in_range is None or in_range == 0:
            result["error"] = f"No factor snapshots found for {symbol} in {start} -&gt; {end}"
            return result

        # Check factors_json validity
        rows = await self.select(
            "daily_factor_snapshots",
            filters=[
                ("symbol", f"eq.{symbol.upper()}"),
                ("snapshot_date", f"gte.{start.isoformat()}"),
                ("snapshot_date", f"lte.{end.isoformat()}"),
            ],
            columns="snapshot_date,factors_json,factor_vector",
            limit=2000,
            order="snapshot_date.asc",
        )

        valid = 0
        invalid = 0
        has_vector = 0
        dates_seen: set[str] = set()
        for r in rows:
            snap_date = str(r.get("snapshot_date", ""))[:10]
            dates_seen.add(snap_date)
            fj = r.get("factors_json")
            if isinstance(fj, list) and len(fj) > 0:
                valid += 1
            elif isinstance(fj, str):
                try:
                    parsed = json.loads(fj)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        valid += 1
                    else:
                        invalid += 1
                except json.JSONDecodeError:
                    invalid += 1
            else:
                invalid += 1
            fv = r.get("factor_vector")
            if fv and isinstance(fv, list) and len(fv) > 0:
                has_vector += 1

        days_total = (end - start).days + 1
        result["stats"]["days_covered"] = len(dates_seen)
        result["stats"]["days_total"] = days_total
        result["stats"]["coverage_pct"] = _fmt_pct(len(dates_seen), days_total)
        result["stats"]["valid_factors"] = valid
        result["stats"]["invalid_factors"] = invalid
        result["stats"]["with_factor_vector"] = has_vector

        all_dates = {_fmt_date(start + timedelta(days=i)) for i in range(days_total)}
        missing_dates = sorted(all_dates - dates_seen)
        if missing_dates:
            result["gaps"] = missing_dates[:30]
            result["gap_count"] = len(missing_dates)

        return result


# -- Service Health Checks ------------------------------------------------------
async def check_service(name: str, url: str, endpoint: str = "/health") -> dict:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            resp = await c.get(f"{url}{endpoint}")
            ok = 200 <= resp.status_code < 300
            body = resp.json() if ok and resp.headers.get("content-type", "").startswith("application/json") else {}
            return {"name": name, "url": url, "reachable": ok, "status": resp.status_code, "body": body}
    except Exception as exc:
        return {"name": name, "url": url, "reachable": False, "error": str(exc)[:120]}


async def check_stockmem(url: str, symbol: str | None = None) -> dict:
    import httpx
    result: dict = {"name": "stockmem", "url": url, "reachable": False, "records": None, "missing_returns": None}
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            # Health check
            resp = await c.get(f"{url}/health")
            if resp.status_code != 200:
                result["error"] = f"health returned {resp.status_code}"
                return result
            result["reachable"] = True

            # Missing returns (checks data quality)
            params = {"symbol": symbol} if symbol else {}
            resp = await c.get(f"{url}/records/missing-returns", params=params)
            if resp.status_code == 200:
                records = resp.json()
                result["missing_returns"] = len(records)
            else:
                result["error"] = f"missing-returns returned {resp.status_code}"
    except Exception as exc:
        result["error"] = str(exc)[:120]
    return result


# -- Main ----------------------------------------------------------------------─
async def main() -> int:
    parser = argparse.ArgumentParser(description="Check backtest data readiness")
    parser.add_argument("--symbol", default="BTC", help="Symbol to check (default: BTC)")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-05-10", help="End date YYYY-MM-DD")
    parser.add_argument("--stockmem-url", default="http://localhost:8003")
    parser.add_argument("--market-data-url", default="http://localhost:8002")
    parser.add_argument("--factorledge-url", default="http://localhost:8004")
    parser.add_argument("--aihub-url", default="http://localhost:8001")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        print(f"Error: end date {end} is before start {start}")
        return 1

    print(f"{_bold('=== Backtest Data Readiness Check ===')}\n")
    print(f"Symbol      : {symbol}")
    print(f"Date range  : {start} -&gt; {end} ({(end - start).days} days)")
    print(f"StockMem    : {args.stockmem_url}")
    print()

    errors = 0

    # -- 1. Supabase ------------------------------------------------------─
    print(_bold("-- 1. Supabase Connectivity --"))
    sup = SupabaseChecker()
    if not sup.available:
        print(f"  {_red('FAIL')} — SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY not set")
        print("  Skipping Supabase checks.\n")
    else:
        ok = await sup.ping()
        if not ok:
            print(f"  {_red('FAIL')} — Cannot reach Supabase PostgREST")
            errors += 1
        else:
            print(f"  {_green('OK')}   — PostgREST reachable ({sup._base})")

            # news_articles
            print(f"\n{_bold('-- 2. Supabase: news_articles --')}")
            na = await sup.check_news_articles(symbol, start, end)
            print(f"  Total rows in table       : {na.get('total_rows', '?')}")
            print(f"  Rows in date range        : {na.get('in_range', '?')}")
            if na["error"]:
                print(f"  {_red('FAIL')} — {na['error']}")
                errors += 1
            else:
                stats = na.get("stats", {})
                print(f"  Days covered              : {stats.get('days_covered', '?')}/{stats.get('days_total', '?')} ({stats.get('coverage_pct', '?')})")
                print(f"  Articles with sentiment   : {stats.get('articles_with_sentiment', '?')}")
                print(f"  Articles missing sentiment: {stats.get('articles_missing_sentiment', '?')} (quality: {stats.get('sentiment_quality', '?')})")
                gaps = na.get("gaps", [])
                if gaps:
                    print(f"  {_yellow('WARN')} — {na.get('gap_count', '?')} dates with no articles (first 5): {gaps[:5]}")
                else:
                    print(f"  {_green('OK')}   — No gaps in daily article coverage")

            # daily_factor_snapshots
            print(f"\n{_bold('-- 3. Supabase: daily_factor_snapshots --')}")
            fs = await sup.check_daily_factor_snapshots(symbol, start, end)
            print(f"  Total rows in table       : {fs.get('total_rows', '?')}")
            print(f"  Rows for {symbol} in range : {fs.get('in_range', '?')}")
            if fs["error"]:
                print(f"  {_yellow('WARN')} — {fs['error']} (will use AIHub fallback)")
            else:
                stats = fs.get("stats", {})
                print(f"  Days covered              : {stats.get('days_covered', '?')}/{stats.get('days_total', '?')} ({stats.get('coverage_pct', '?')})")
                print(f"  Valid factor JSON         : {stats.get('valid_factors', '?')}")
                print(f"  Invalid/missing factors   : {stats.get('invalid_factors', '?')}")
                print(f"  With factor_vector (75d)  : {stats.get('with_factor_vector', '?')}")
                gaps = fs.get("gaps", [])
                if gaps:
                    print(f"  {_yellow('WARN')} — {fs.get('gap_count', '?')} dates missing factor snapshots (will fall back to AIHub)")
                else:
                    print(f"  {_green('OK')}   — Factor snapshots cover all days")

        await sup.close()

    print()

    # -- 4. Services ------------------------------------------------------─
    print(_bold("-- 4. Service Health --"))
    services = await asyncio.gather(
        check_stockmem(args.stockmem_url, symbol),
        check_service("market_data", args.market_data_url),
        check_service("factor_ledge", args.factorledge_url),
        check_service("aihub", args.aihub_url),
    )

    for svc in services:
        if svc["reachable"]:
            extra = ""
            if svc["name"] == "stockmem":
                missing = svc.get("missing_returns")
                if missing is not None:
                    extra = f" | records missing returns: {missing}"
            print(f"  {_green('OK')}   — {svc['name']} ({svc['url']}){extra}")
        else:
            err = svc.get("error", "unreachable")
            required = svc["name"] in ("stockmem", "market_data")
            tag = _red("FAIL (required)") if required else _yellow("WARN (optional)")
            print(f"  {tag} — {svc['name']} ({svc['url']}): {err}")
            if required:
                errors += 1

    print()

    # -- 5. Summary --------------------------------------------------------
    print(_bold("-- 5. Summary --"))
    if errors == 0:
        print(f"  {_green('All critical checks passed.')} Ready to backfill and run backtest.")
    else:
        print(f"  {_red(f'{errors} critical issue(s) found.')} Fix before proceeding.")

    print()
    print(_bold("Next steps (in order):"))
    print(f"  1. POST /backfill?symbol={symbol}&days=90&offset=0  — populate StockMem")
    print(f"  2. Repeat backfill with offset+=90 until reaching {start}")
    print(f"  3. POST /fill-returns?symbol={symbol}              — compute forward returns")
    print(f"  4. POST /run?symbol={symbol}&date=YYYY-MM-DD        — run single backtest prediction")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
