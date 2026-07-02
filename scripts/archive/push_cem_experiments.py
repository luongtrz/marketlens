"""Push CEM-RAG experiment results to Supabase cem_rag_experiments table.

Run AFTER creating the table via scripts/migrations/001_cem_rag_experiments.sql.

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/push_cem_experiments.py
"""
from __future__ import annotations

import json
import sys
from datetime import timezone
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
    import os; env.update(os.environ)
    return env


ENV = _load_env()
SUPABASE_URL = ENV.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = ENV.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",  # upsert
    }


EXPERIMENTS = [
    # ── v3: 75% event_vec, best variant was event_zeroed ──────────────────────
    {
        "id": "cem-rag-v3",
        "run_name": "v3 — dense optimizer, 75% event_vec",
        "data_version": "real_optimizer_v3_initial",
        "retriever_version": "learned_retriever_v3.json",
        "policy_version": "policy_v3.json",
        "total_rows": 2886,
        "event_vec_coverage": 0.754,
        "val_hit_at_k": 1.0000,
        "seed_std": 0.0000,
        "trials": 10, "epochs": 40, "seeds": 5,
        "test_n": 305, "test_n_buy": 83, "test_n_sell": 40, "test_n_hold": 182,
        "test_da": 0.783607,
        "test_buy_da": 0.409639,
        "test_sell_da": 0.575,
        "test_coverage": 0.403279,
        "test_sharpe": -0.009055,
        "test_sortino": -0.792559,
        "test_max_drawdown": -0.96925,
        "test_brier": 1.151808,
        "test_ece": 0.218708,
        "test_combined": None,           # evaluate_retriever not run on v3 policy
        "test_hit_at_5": None,
        "policy_tau": 0.22,
        "val_sharpe": -0.073611,
        "mcnemar_da_p": None,
        "mcnemar_hit_p": None,
        "notes": "best_variant=learned_event_zeroed; full 4-block underperforms due to 25% zero event_vecs",
    },
    # ── v4: 93% event_vec, full 4-block model best ────────────────────────────
    {
        "id": "cem-rag-v4",
        "run_name": "v4 — LLM-populated event_state, 93% event_vec",
        "data_version": "real_optimizer_v3.json (regen after populate)",
        "retriever_version": "learned_retriever_v4.json",
        "policy_version": "policy_v4.json",
        "total_rows": 2886,
        "event_vec_coverage": 0.930,
        "val_hit_at_k": 0.9988,
        "seed_std": 0.0024,
        "trials": 10, "epochs": 40, "seeds": 5,
        "test_n": 305, "test_n_buy": 96, "test_n_sell": 40, "test_n_hold": 169,
        "test_da": 0.783607,
        "test_buy_da": 0.427083,
        "test_sell_da": 0.725,
        "test_coverage": 0.445902,
        "test_sharpe": -0.215985,
        "test_sortino": -0.32255,
        "test_max_drawdown": -0.970802,
        "test_brier": 1.141312,
        "test_ece": 0.200707,
        "test_combined": 0.2148,         # from evaluate_retriever.py learned_diagonal row
        "test_hit_at_5": 0.9502,
        "policy_tau": 0.22,
        "val_sharpe": -0.194614,
        "mcnemar_da_p": 0.7035,
        "mcnemar_hit_p": 0.3075,
        "notes": (
            "best_variant=learned_diagonal (full 4-block); event block now contributing positively. "
            "SELL-DA 57.5%→72.5%. Sharpe regressed due to higher BUY count (96 vs 83). "
            "Groq llama-3.1-8b-instant; 2373 LLM + 493 rule-based; TPD 500k/day."
        ),
    },
]


def main() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in .env")
        sys.exit(1)

    with httpx.Client() as client:
        for exp in EXPERIMENTS:
            r = client.post(
                f"{SUPABASE_URL}/rest/v1/cem_rag_experiments",
                headers=_headers(),
                content=json.dumps(exp),
                timeout=15,
            )
            if r.status_code in (200, 201):
                print(f"  ✓ upserted {exp['id']}")
            else:
                print(f"  ✗ {exp['id']}: {r.status_code} {r.text[:200]}")


if __name__ == "__main__":
    main()
