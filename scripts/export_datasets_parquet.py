"""
Export train/val/test parquet datasets from real_optimizer_v3.json.

Columns: date, split, future_return_1d, future_return_3d, future_return_7d,
         future_return_15d, future_return_30d, event_vec, factor_vec,
         indicator_vec, price_vec
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import pyarrow  # noqa: F401 – just verify it's present
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "pyarrow", "-q"])
    import pandas as pd

from stockmem.scripts.optimize_weights import load_rows
from stockmem.scripts.cem_dataset import label_rows

DATA_PATH = ROOT / "stockmem" / "data" / "real_optimizer_v3.json"
OUT_DIR = ROOT / "artifacts" / "datasets"

# ── load raw rows (missing 3d/15d in Row dataclass) ─────────────────────────
rows = load_rows(DATA_PATH)
print(f"Loaded {len(rows)} rows from {DATA_PATH.name}")

# ── label rows → split assignment ────────────────────────────────────────────
labeled = label_rows(rows)
print(f"Labeled {len(labeled)} rows")

# ── also load raw JSON for 3d/15d fields ─────────────────────────────────────
raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
raw_by_date: dict[str, dict] = {str(item["date"]): item for item in raw}

# ── build records ─────────────────────────────────────────────────────────────
records = []
for lr in labeled:
    r = lr.row
    raw_item = raw_by_date.get(r.date, {})
    records.append(
        {
            "date": r.date,
            "split": lr.split,
            "future_return_1d": r.future_return_1d,
            "future_return_3d": float(raw_item.get("future_return_3d") or 0.0),
            "future_return_7d": r.future_return_7d,
            "future_return_15d": float(raw_item.get("future_return_15d") or 0.0),
            "future_return_30d": r.future_return_30d,
            "event_vec": r.event_vec.tolist(),
            "factor_vec": r.factor_vec.tolist(),
            "indicator_vec": r.indicator_vec.tolist(),
            "price_vec": r.price_vec.tolist(),
        }
    )

df = pd.DataFrame(records)
print(f"\nDataFrame shape: {df.shape}")
print(f"Splits:\n{df['split'].value_counts().to_string()}")

# ── export ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

for split_name in ("train", "val", "test"):
    subset = df[df["split"] == split_name].copy()
    out_path = OUT_DIR / f"daily_records_{split_name}.parquet"
    subset.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  {split_name:5s}: {len(subset):5d} rows → {out_path}")

print("\nDone.")
