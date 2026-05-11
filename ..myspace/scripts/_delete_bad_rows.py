"""Delete bad backtest rows (Groq failures: confidence=0)."""
import asyncio, os, httpx
from pathlib import Path

env_file = Path(r"F:\DATN\marketlens_backtest\marketlens\.env")
with open(env_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "=" not in line: continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip("\"'")

BASE = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

async def main():
    async with httpx.AsyncClient(timeout=30) as c:
        headers = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
        accept = {**headers, "Accept": "application/json"}

        # Find ALL rows with confidence=0 (Groq failures)
        r = await c.get(f"{BASE}/rest/v1/backtest_results",
            headers=accept,
            params=[("select", "id,backtest_date,signal,confidence"),
                    ("symbol", "eq.BTC"), ("confidence", "eq.0"),
                    ("order", "backtest_date.asc"), ("limit", "500")])

        if r.status_code != 200:
            print(f"Error: {r.status_code} {r.text[:200]}")
            return

        rows = r.json()
        if not rows:
            print("No bad rows found.")
            return

        print(f"Found {len(rows)} bad rows (conf=0 = Groq rate limit):")
        for row in rows:
            print(f"  {row['backtest_date']} signal={row['signal']} id={row['id']}")

        # Delete all bad rows (confidence=0)
        id_list = ",".join(str(row["id"]) for row in rows)
        d = await c.delete(f"{BASE}/rest/v1/backtest_results",
            headers={**headers, "Prefer": "return=representation"},
            params=[("id", f"in.({id_list})")])

        if 200 <= d.status_code < 300:
            print(f"\nDeleted {len(rows)} rows.")
        else:
            print(f"\nDelete failed ({d.status_code}): {d.text[:300]}")

asyncio.run(main())
