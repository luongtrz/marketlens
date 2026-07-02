"""PatchTST baseline for CEM-RAG comparison.

Architecture (Nie et al. 2023 — channel-independence variant):
  - Input: price_vec (60d) = [close_returns(20) | intraday_ranges(20) | volume_changes(20)]
    We treat each of the 3 channels as an independent time series of length 20.
  - Patch: P=5, stride=2 → (20-5)//2 + 1 = 8 patches per channel
  - Embed each patch with linear projection → d_model=64
  - Transformer encoder: 2 layers, 4 heads, ffn=128, dropout=0.1
  - Classification head: mean-pool patches → Linear(64→2) → softmax
  - Label: sign(future_return_7d), UP=1 DOWN=0, skip zeros

Splits:
  train: 2363 rows  val: 174  test: 305
  StandardScaler fit on train only (per channel)
  Early stop on val AUC (patience=10)

Output:
  artifacts/predictions/patchtst_test.jsonl
  main_table.csv row appended as "patchtst"

Usage:
    PYTHONPATH=/home/luong/marketlens python scripts/train_patchtst_baseline.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from stockmem.scripts.cem_dataset import label_rows
from stockmem.scripts.optimize_weights import load_rows

PRED_DIR  = ROOT / "artifacts/predictions"
METRICS   = ROOT / "artifacts/metrics/main_table.csv"
DATA_PATH = ROOT / "stockmem/data/real_optimizer_v3.json"

PRED_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-params ─────────────────────────────────────────────────────────────
SEQ_LEN   = 20   # time steps per channel in price_vec
N_CHAN    = 3    # channels: close_ret / intraday_range / volume_change
PATCH_LEN = 5
STRIDE    = 2
D_MODEL   = 64
N_HEADS   = 4
N_LAYERS  = 2
FFN_DIM   = 128
DROPOUT   = 0.1
EPOCHS    = 80
PATIENCE  = 12
LR        = 3e-4
BATCH     = 64
SEED      = 42

COST_PCT  = 0.10
PERIODS_PER_YEAR = 252 / 7
HORIZON   = "future_return_7d"


def set_seed(s: int):
    torch.manual_seed(s)
    np.random.seed(s)


# ── PatchTST model ───────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_patches, patch_len)
        return self.norm(self.proj(x))


class PatchTST(nn.Module):
    """Channel-independence PatchTST for binary classification."""

    def __init__(self, seq_len=SEQ_LEN, n_chan=N_CHAN, patch_len=PATCH_LEN,
                 stride=STRIDE, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        n_patches = (seq_len - patch_len) // stride + 1
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = n_patches
        self.n_chan    = n_chan

        self.embed = PatchEmbedding(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * n_chan, 2)
        self.drop = nn.Dropout(dropout)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_chan, seq_len)
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            patches.append(x[:, :, start:start + self.patch_len])
        # → (B, n_chan, n_patches, patch_len)
        return torch.stack(patches, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_chan, seq_len)
        B = x.shape[0]
        patches = self._patchify(x)              # (B, n_chan, n_patches, patch_len)

        # Process each channel independently
        chan_out = []
        for c in range(self.n_chan):
            cp = patches[:, c, :, :]             # (B, n_patches, patch_len)
            emb = self.embed(cp) + self.pos_embed # (B, n_patches, d_model)
            enc = self.encoder(emb)               # (B, n_patches, d_model)
            pooled = enc.mean(dim=1)              # (B, d_model)
            chan_out.append(pooled)

        fused = torch.cat(chan_out, dim=-1)       # (B, d_model * n_chan)
        return self.head(self.drop(fused))        # (B, 2)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_price_channels(lb) -> np.ndarray | None:
    """Return (N_CHAN, SEQ_LEN) from price_vec(60d) = [cr20|ir20|vc20]."""
    pv = getattr(lb.row, "price_vec", None)
    if pv is None:
        return None
    arr = np.asarray(pv, dtype=np.float32)
    if arr.shape != (60,):
        return None
    cr = arr[:20]
    ir = arr[20:40]
    vc = arr[40:60]
    return np.stack([cr, ir, vc])  # (3, 20)


# ── Build dataset ─────────────────────────────────────────────────────────────

def build(labeled) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X, y, meta = [], [], []
    for lb in labeled:
        ret7 = getattr(lb.row, HORIZON, None)
        if ret7 is None or float(ret7) == 0.0:
            continue
        channels = extract_price_channels(lb)
        if channels is None:
            continue
        X.append(channels)
        y.append(1 if float(ret7) > 0 else 0)
        meta.append({
            "date":  lb.row.date.isoformat() if hasattr(lb.row.date, "isoformat") else str(lb.row.date),
            "split": lb.split,
            "actual_return_7d":  float(ret7),
            "actual_return_1d":  float(getattr(lb.row, "future_return_1d",  None) or 0),
            "actual_return_3d":  float(getattr(lb.row, "future_return_3d",  None) or 0),
            "actual_return_15d": float(getattr(lb.row, "future_return_15d", None) or 0),
            "actual_return_30d": float(getattr(lb.row, "future_return_30d", None) or 0),
        })
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), meta


# ── Normalize per-channel ─────────────────────────────────────────────────────

def fit_scalers(X_tr: np.ndarray) -> list[StandardScaler]:
    scalers = []
    for c in range(N_CHAN):
        sc = StandardScaler()
        sc.fit(X_tr[:, c, :])
        scalers.append(sc)
    return scalers


def apply_scalers(X: np.ndarray, scalers: list[StandardScaler]) -> np.ndarray:
    out = X.copy()
    for c, sc in enumerate(scalers):
        out[:, c, :] = sc.transform(X[:, c, :])
    return out


# ── Train / eval helpers ──────────────────────────────────────────────────────

def make_loader(X, y, shuffle=False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


@torch.no_grad()
def predict_proba(model, loader) -> np.ndarray:
    model.eval()
    probs = []
    for xb, _ in loader:
        logits = model(xb)
        p = torch.softmax(logits, dim=-1)[:, 1].numpy()
        probs.append(p)
    return np.concatenate(probs)


def train_epoch(model, loader, opt, criterion) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / n


# ── Tau tuning ────────────────────────────────────────────────────────────────

def tune_tau(p_up: np.ndarray, y: np.ndarray) -> float:
    best_tau, best_sell_da = 0.0, 0.0
    n = len(y)
    for tau_int in range(0, 46, 2):
        tau = tau_int / 100.0
        p_down = 1.0 - p_up
        sells = (p_down - p_up) >= tau
        buys  = (p_up - p_down) >= tau
        n_sell = sells.sum()
        if n_sell < max(3, 0.03 * n):
            continue
        sell_da = ((y == 0) & sells).sum() / max(1, n_sell)
        if sell_da > best_sell_da:
            best_sell_da, best_tau = sell_da, tau
    return best_tau


def apply_tau(p_up: np.ndarray, tau: float) -> np.ndarray:
    """Returns signal array: 1=BUY, -1=SELL, 0=HOLD."""
    p_down = 1.0 - p_up
    sig = np.zeros(len(p_up), dtype=np.int8)
    sig[p_up - p_down >= tau]  =  1
    sig[p_down - p_up >= tau]  = -1
    return sig


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(sigs: np.ndarray, p_up: np.ndarray, y: np.ndarray,
                    rets: np.ndarray) -> dict:
    dir_mask = sigs != 0
    n_dir = dir_mask.sum()
    n = len(sigs)

    da = float(((sigs == 1) & (y == 1) | (sigs == -1) & (y == 0)).sum()) / max(1, n_dir)
    coverage = float(n_dir) / max(1, n)

    y_pred_bin = np.where(sigs == 1, 1, 0)
    bal  = balanced_accuracy_score(y, y_pred_bin)
    f1   = f1_score(y, y_pred_bin, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(y, y_pred_bin)
    brier = float(np.mean((p_up - y) ** 2))

    trade_rets = np.where(sigs ==  1, rets - COST_PCT,
                 np.where(sigs == -1, -rets - COST_PCT, 0.0))
    mean_r = trade_rets.mean() * PERIODS_PER_YEAR
    std_r  = trade_rets.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR) if n > 1 else 1e-9
    sharpe = float(mean_r / std_r) if std_r > 1e-9 else 0.0

    neg = trade_rets[trade_rets < 0]
    down_std = neg.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR) if len(neg) > 1 else 1e-9
    sortino = float(mean_r / down_std) if down_std > 1e-9 else 0.0

    cum, peak, max_dd = 1.0, 1.0, 0.0
    for r in trade_rets:
        cum *= (1 + r / 100)
        if cum > peak:
            peak = cum
        dd = (cum - peak) / peak
        if dd < max_dd:
            max_dd = dd

    sell_da = float(((sigs == -1) & (y == 0)).sum()) / max(1, (sigs == -1).sum())
    buy_da  = float(((sigs ==  1) & (y == 1)).sum()) / max(1, (sigs ==  1).sum())

    return {
        "da": round(da, 6), "balanced_acc": round(bal, 6), "macro_f1": round(f1, 6),
        "mcc": round(mcc, 6), "coverage": round(coverage, 6),
        "sharpe": round(sharpe, 6), "sortino": round(sortino, 6),
        "max_dd": round(max_dd, 6), "hit_at_5": 0.0, "brier": round(brier, 6),
        "sell_da": round(sell_da, 6), "buy_da": round(buy_da, 6),
        "n_buy": int((sigs == 1).sum()), "n_sell": int((sigs == -1).sum()),
        "n_hold": int((sigs == 0).sum()),
    }


# ── Write JSONL ───────────────────────────────────────────────────────────────

def write_jsonl(sigs, p_up, meta_te, y_te, path: Path):
    with path.open("w") as f:
        for i, m in enumerate(meta_te):
            pu = float(p_up[i])
            sig_str = "BUY" if sigs[i] == 1 else ("SELL" if sigs[i] == -1 else "HOLD")
            row = {
                "date": m["date"], "symbol": "BTC",
                "signal": sig_str, "confidence": float(max(pu, 1 - pu)),
                "p_up": pu, "p_down": float(1 - pu), "p_hold": 0.0,
                "actual_return_1d":  m["actual_return_1d"],
                "actual_return_3d":  m["actual_return_3d"],
                "actual_return_7d":  m["actual_return_7d"],
                "actual_return_15d": m["actual_return_15d"],
                "actual_return_30d": m["actual_return_30d"],
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(meta_te)} rows → {path}")


def append_main_table(name: str, m: dict):
    header = "retriever,da,balanced_acc,macro_f1,mcc,coverage,sharpe,sortino,max_dd,hit_at_5,brier\n"
    row = (f"{name},{m['da']},{m['balanced_acc']},{m['macro_f1']},{m['mcc']},"
           f"{m['coverage']},{m['sharpe']},{m['sortino']},{m['max_dd']},"
           f"{m['hit_at_5']},{m['brier']}\n")
    txt = METRICS.read_text() if METRICS.exists() else header
    lines = [l for l in txt.splitlines(keepends=True) if not l.startswith(name + ",")]
    METRICS.write_text("".join(lines) + row)
    print(f"Updated main_table.csv → {name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    print("Loading data...")
    rows = load_rows(DATA_PATH)
    labeled = label_rows(rows)

    print("Building dataset (price channels only)...")
    X_all, y_all, meta_all = build(labeled)
    splits = np.array([m["split"] for m in meta_all])
    print(f"  Total rows: {len(X_all)}, shape: {X_all.shape}")

    tr = splits == "train"
    va = splits == "val"
    te = splits == "test"
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_va, y_va = X_all[va], y_all[va]
    X_te, y_te = X_all[te], y_all[te]
    meta_va = [m for m, s in zip(meta_all, splits) if s == "val"]
    meta_te = [m for m, s in zip(meta_all, splits) if s == "test"]
    rets_te = np.array([m["actual_return_7d"] for m in meta_te], dtype=np.float64)

    print(f"  train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    # Normalize
    scalers = fit_scalers(X_tr)
    X_tr_s = apply_scalers(X_tr, scalers)
    X_va_s = apply_scalers(X_va, scalers)
    X_te_s = apply_scalers(X_te, scalers)

    # Class weights
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]))

    tr_loader = make_loader(X_tr_s, y_tr, shuffle=True)
    va_loader = make_loader(X_va_s, y_va)
    te_loader = make_loader(X_te_s, y_te)

    model = PatchTST()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    best_auc, best_state, patience_cnt = 0.0, None, 0
    print(f"\nTraining PatchTST ({EPOCHS} epochs, patience={PATIENCE})...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, tr_loader, opt, criterion)
        sched.step()
        p_va = predict_proba(model, va_loader)
        try:
            auc = roc_auc_score(y_va, p_va)
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_state, patience_cnt = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  loss={loss:.4f}  val_auc={auc:.4f}  best={best_auc:.4f}")
        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"\nBest val AUC: {best_auc:.4f}")
    model.load_state_dict(best_state)

    # Val tau tuning
    p_up_va = predict_proba(model, va_loader)
    tau = tune_tau(p_up_va, y_va)
    sigs_va = apply_tau(p_up_va, tau)
    n_sell_va = (sigs_va == -1).sum()
    sell_da_va = float(((sigs_va == -1) & (y_va == 0)).sum()) / max(1, n_sell_va)
    print(f"Best tau: {tau:.2f}, val SELL-DA: {sell_da_va:.3f}, val n_sell: {n_sell_va}")

    # Test
    p_up_te = predict_proba(model, te_loader)
    sigs_te = apply_tau(p_up_te, tau)
    m = compute_metrics(sigs_te, p_up_te, y_te, rets_te)

    print(f"\nTest results:")
    print(f"  DA={m['da']:.3f}  SELL-DA={m['sell_da']:.3f}  BUY-DA={m['buy_da']:.3f}")
    print(f"  MCC={m['mcc']:.4f}  Sharpe={m['sharpe']:.4f}  cov={m['coverage']:.3f}")
    print(f"  n_buy={m['n_buy']}  n_sell={m['n_sell']}  n_hold={m['n_hold']}")

    write_jsonl(sigs_te, p_up_te, meta_te, y_te, PRED_DIR / "patchtst_test.jsonl")
    append_main_table("patchtst", m)
    print("\nDone.")


if __name__ == "__main__":
    main()
