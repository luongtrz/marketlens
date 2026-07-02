"""Push learned_retriever_v4 + policy_v4 artifacts to Supabase.

Run AFTER creating tables via scripts/migrations/002_cem_rag_models.sql.

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/push_cem_models.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for ef in [ROOT / ".env"]:
        if ef.exists():
            for line in ef.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env.setdefault(k.strip(), v.strip())
    env.update(os.environ)
    return env


ENV = _load_env()
SUPABASE_URL = ENV.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = ENV.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }


def upsert(table: str, payload: dict, client: httpx.Client) -> None:
    r = client.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_headers(),
        content=json.dumps(payload),
        timeout=20,
    )
    if r.status_code in (200, 201):
        print(f"  ✓ upserted {table}/{payload['id']}")
    else:
        print(f"  ✗ {table}/{payload['id']}: {r.status_code} {r.text[:300]}")


def main() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")
        sys.exit(1)

    # Load artifacts
    retriever = json.loads((ROOT / "stockmem/config/learned_retriever_v4.json").read_text())
    policy = json.loads((ROOT / "stockmem/config/policy_v4.json").read_text())

    retriever_row = {
        "id": "learned-diagonal-v4",
        "version": retriever["version"],
        "type": retriever["type"],
        "dim": retriever["dim"],
        "block_dims": retriever["block_dims"],
        "block_scales": retriever["block_scales"],
        "d": retriever["d"],
        "band": retriever["band"],
        "splits": retriever["splits"],
        "hyperparameters": retriever["hyperparameters"],
        "mining_protocol": retriever.get("mining_protocol"),
        "val_hit_at_k": retriever.get("val_hit_at_k"),
        "val_combined": retriever.get("val_combined"),
        "seed_std": retriever.get("seed_std"),
        "seeds": retriever.get("seeds"),
        "source_data": retriever.get("source"),
        "trained_at": retriever.get("trained_at"),
        "notes": "4-block: event85+factor75+indicator5+price60=225d; best variant on v4 data (93% event_vec)",
    }

    test_m = policy["test_metrics"]
    policy_row = {
        "id": "policy-v4",
        "retriever_id": "learned-diagonal-v4",
        "method": policy["method"],
        "tau": policy["tau"],
        "k": policy["k"],
        "val_sharpe": policy["val_sharpe"],
        "test_n": test_m["n"],
        "test_n_buy": test_m["n_buy"],
        "test_n_sell": test_m["n_sell"],
        "test_n_hold": test_m["n_hold"],
        "test_da": test_m["da"],
        "test_buy_da": test_m["buy_da"],
        "test_sell_da": test_m["sell_da"],
        "test_coverage": test_m["coverage"],
        "test_sharpe": test_m["sharpe"],
        "test_sortino": test_m["sortino"],
        "test_max_drawdown": test_m["max_drawdown"],
        "test_brier": test_m["brier"],
        "test_ece": test_m["ece"],
        "test_metrics": test_m,
        "val_best_metrics": policy["val_best_metrics"],
        "notes": "tau=0.22; SELL-DA 72.5% (+15pp vs v3); coverage 44.6%; Groq llama-3.1-8b event extraction",
    }

    with httpx.Client() as client:
        upsert("cem_rag_retrievers", retriever_row, client)
        upsert("cem_rag_policies", policy_row, client)


if __name__ == "__main__":
    main()
