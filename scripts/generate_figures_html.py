"""
Generate two HTML figures from artifacts/predictions/*_test.jsonl:

1. equity_curve.html   — cumulative return curves (dark theme, canvas)
2. coverage_precision.html — coverage vs directional_da sweep (dark theme, canvas)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "artifacts" / "predictions"
SCRATCHPAD = Path("/tmp/claude-1000/-home-luong-marketlens/f90cef3b-d927-41d1-b502-952756cef3cc/scratchpad")
SCRATCHPAD.mkdir(parents=True, exist_ok=True)

COST_BPS = 0.10  # 10 bps

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def compute_equity(records: list[dict]) -> list[tuple[str, float]]:
    """Return (date, cumulative_pct) starting at 100."""
    records = sorted(records, key=lambda r: r["date"])
    cum = 100.0
    curve: list[tuple[str, float]] = [("", 100.0)]  # sentinel start
    for rec in records:
        sig = rec.get("signal", "HOLD")
        ret7d = rec.get("actual_return_7d")
        if ret7d is None:
            step = 0.0
        elif sig == "BUY":
            step = ret7d - COST_BPS
        elif sig == "SELL":
            step = -ret7d - COST_BPS
        else:
            step = 0.0
        cum *= (1 + step / 100.0)
        curve.append((rec["date"], round(cum, 4)))
    # drop sentinel start
    return curve[1:]


# ── Model configuration ──────────────────────────────────────────────────────

EQUITY_MODELS = {
    "cem_rag": "cem_rag_test.jsonl",
    "knn_returns": "knn_returns_test.jsonl",
    "fixed_knn": "fixed_knn_test.jsonl",
    "xgboost_price_only": "xgboost_price_only_test.jsonl",
}

# Models with p_up/p_down for coverage-precision sweep
PROB_MODELS = {
    "cem_rag": "cem_rag_test.jsonl",
    "cem_rag_full": "cem_rag_full_test.jsonl",
    "xgboost": "xgboost_test.jsonl",
    "xgboost_price_only": "xgboost_price_only_test.jsonl",
    "xgboost_no_event": "xgboost_no_event_test.jsonl",
    "xgboost_event_only": "xgboost_event_only_test.jsonl",
    "ablation_no_event": "ablation_no_event_test.jsonl",
    "ablation_no_factor": "ablation_no_factor_test.jsonl",
    "ablation_no_policy": "ablation_no_policy_test.jsonl",
    "ablation_no_retrieval": "ablation_no_retrieval_test.jsonl",
    "ablation_price_only": "ablation_price_only_test.jsonl",
    "ablation_fixed_ret": "ablation_fixed_ret_test.jsonl",
}

# ── Color palette ─────────────────────────────────────────────────────────────
PALETTE = [
    "#4fc3f7",  # sky blue
    "#81c784",  # green
    "#ffb74d",  # amber
    "#f06292",  # pink
    "#ce93d8",  # purple
    "#4db6ac",  # teal
    "#fff176",  # yellow
    "#ff8a65",  # deep orange
    "#80cbc4",  # teal light
    "#ef9a9a",  # red light
    "#b0bec5",  # blue-grey
    "#a5d6a7",  # green light
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. EQUITY CURVE
# ─────────────────────────────────────────────────────────────────────────────

equity_series: dict[str, list[tuple[str, float]]] = {}
for model_name, fname in EQUITY_MODELS.items():
    path = PRED_DIR / fname
    if path.exists():
        recs = load_jsonl(path)
        equity_series[model_name] = compute_equity(recs)
        print(f"  equity {model_name}: {len(recs)} records")
    else:
        print(f"  SKIP (not found): {fname}")

# Buy-and-hold: use actual_return_7d as daily signal = BUY always
bah_path = PRED_DIR / "cem_rag_test.jsonl"  # use this as date source
if bah_path.exists():
    bah_recs = load_jsonl(bah_path)
    bah_series: list[tuple[str, float]] = []
    cum = 100.0
    for rec in sorted(bah_recs, key=lambda r: r["date"]):
        ret7d = rec.get("actual_return_7d")
        if ret7d is None:
            step = 0.0
        else:
            step = ret7d - COST_BPS  # always BUY
        cum *= (1 + step / 100.0)
        bah_series.append((rec["date"], round(cum, 4)))
    equity_series["buy_and_hold"] = bah_series

# Collect all dates
all_dates: list[str] = sorted(
    {d for series in equity_series.values() for d, _ in series}
)

def series_to_aligned(series: list[tuple[str, float]], all_dates: list[str]) -> list[float | None]:
    d2v = {d: v for d, v in series}
    return [d2v.get(d) for d in all_dates]

aligned: dict[str, list[float | None]] = {
    name: series_to_aligned(s, all_dates) for name, s in equity_series.items()
}

model_names_eq = list(equity_series.keys())

equity_js_data = json.dumps({
    "dates": all_dates,
    "series": {name: aligned[name] for name in model_names_eq},
    "colors": {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(model_names_eq)},
})

EQUITY_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Equity Curves — MarketLens</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', system-ui, sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; color: #f0f6fc; }}
  p.subtitle {{ font-size: 0.85rem; color: #8b949e; margin-bottom: 20px; }}
  #chart-wrap {{ position: relative; width: 100%; }}
  canvas {{ width: 100% !important; display: block; border-radius: 8px; background: #161b22; }}
  #legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }}
  .leg-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #c9d1d9; cursor: pointer; user-select: none; }}
  .leg-swatch {{ width: 28px; height: 3px; border-radius: 2px; }}
  #tooltip {{
    position: fixed; pointer-events: none;
    background: #21262d; border: 1px solid #30363d;
    border-radius: 6px; padding: 8px 12px;
    font-size: 0.78rem; color: #e6edf3; line-height: 1.6;
    display: none; z-index: 100; max-width: 220px;
  }}
</style>
</head>
<body>
<h1>Equity Curves</h1>
<p class="subtitle">Cumulative return (%) · cost = 10 bps per trade · BUY: +ret7d, SELL: −ret7d, HOLD: 0</p>
<div id="chart-wrap">
  <canvas id="cv"></canvas>
</div>
<div id="legend"></div>
<div id="tooltip"></div>

<script>
const DATA = {equity_js_data};

const MARGIN = {{ top: 24, right: 24, bottom: 60, left: 72 }};
const canvas = document.getElementById('cv');
const tooltip = document.getElementById('tooltip');
const legendEl = document.getElementById('legend');

// build legend
const names = Object.keys(DATA.series);
const hidden = new Set();
names.forEach((name, i) => {{
  const color = DATA.colors[name];
  const item = document.createElement('div');
  item.className = 'leg-item';
  item.innerHTML = `<div class="leg-swatch" style="background:${{color}}"></div><span>${{name.replace(/_/g,' ')}}</span>`;
  item.addEventListener('click', () => {{
    if (hidden.has(name)) hidden.delete(name); else hidden.add(name);
    item.style.opacity = hidden.has(name) ? '0.35' : '1';
    draw();
  }});
  legendEl.appendChild(item);
}});

function getSize() {{
  const wrap = document.getElementById('chart-wrap');
  return {{ w: wrap.clientWidth, h: Math.min(520, Math.max(340, wrap.clientWidth * 0.45)) }};
}}

function draw() {{
  const {{ w, h }} = getSize();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pw = w - MARGIN.left - MARGIN.right;
  const ph = h - MARGIN.top - MARGIN.bottom;

  // background
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  const dates = DATA.dates;
  const n = dates.length;
  if (n === 0) return;

  // y range across visible series
  let yMin = Infinity, yMax = -Infinity;
  names.forEach(name => {{
    if (hidden.has(name)) return;
    DATA.series[name].forEach(v => {{
      if (v !== null && v !== undefined) {{ yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }}
    }});
  }});
  if (!isFinite(yMin)) {{ yMin = 80; yMax = 120; }}
  const pad = (yMax - yMin) * 0.08 || 5;
  yMin -= pad; yMax += pad;

  const xScale = i => MARGIN.left + (i / (n - 1)) * pw;
  const yScale = v => MARGIN.top + ph - ((v - yMin) / (yMax - yMin)) * ph;

  // grid lines
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  const nGridY = 6;
  for (let g = 0; g <= nGridY; g++) {{
    const y = MARGIN.top + (g / nGridY) * ph;
    ctx.beginPath(); ctx.moveTo(MARGIN.left, y); ctx.lineTo(MARGIN.left + pw, y); ctx.stroke();
    const val = yMax - (g / nGridY) * (yMax - yMin);
    ctx.fillStyle = '#6e7681'; ctx.font = '11px system-ui'; ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(0), MARGIN.left - 8, y + 4);
  }}

  // baseline at 100
  const y100 = yScale(100);
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(MARGIN.left, y100); ctx.lineTo(MARGIN.left + pw, y100); ctx.stroke();
  ctx.setLineDash([]);

  // x-axis labels (approx 8)
  const step = Math.max(1, Math.floor(n / 8));
  ctx.fillStyle = '#6e7681'; ctx.font = '11px system-ui'; ctx.textAlign = 'center';
  for (let i = 0; i < n; i += step) {{
    const x = xScale(i);
    ctx.fillText(dates[i].slice(0, 10), x, h - MARGIN.bottom + 18);
  }}

  // axes
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(MARGIN.left, MARGIN.top); ctx.lineTo(MARGIN.left, MARGIN.top + ph);
  ctx.lineTo(MARGIN.left + pw, MARGIN.top + ph);
  ctx.stroke();

  // series lines
  names.forEach(name => {{
    if (hidden.has(name)) return;
    const vals = DATA.series[name];
    const color = DATA.colors[name];
    ctx.strokeStyle = color;
    ctx.lineWidth = name === 'cem_rag' ? 2.5 : (name === 'buy_and_hold' ? 1.5 : 1.8);
    ctx.setLineDash(name === 'buy_and_hold' ? [6, 4] : []);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {{
      const v = vals[i];
      if (v === null || v === undefined) {{ started = false; continue; }}
      const x = xScale(i), y = yScale(v);
      if (!started) {{ ctx.moveTo(x, y); started = true; }}
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    ctx.setLineDash([]);
  }});

  // axis label
  ctx.save();
  ctx.translate(14, MARGIN.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#8b949e'; ctx.font = '12px system-ui'; ctx.textAlign = 'center';
  ctx.fillText('Cumulative Return (%)', 0, 0);
  ctx.restore();
}}

// tooltip
canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const {{ w, h }} = getSize();
  const pw = w - MARGIN.left - MARGIN.right;
  const n = DATA.dates.length;
  if (n === 0) return;
  const idx = Math.round(((mx - MARGIN.left) / pw) * (n - 1));
  if (idx < 0 || idx >= n) {{ tooltip.style.display = 'none'; return; }}
  const date = DATA.dates[idx];
  let html = `<strong>${{date}}</strong><br>`;
  names.forEach(name => {{
    if (hidden.has(name)) return;
    const v = DATA.series[name][idx];
    if (v !== null && v !== undefined) {{
      const color = DATA.colors[name];
      html += `<span style="color:${{color}}">${{name.replace(/_/g,' ')}}</span>: ${{v.toFixed(1)}}%<br>`;
    }}
  }});
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY - 20) + 'px';
}});
canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

window.addEventListener('resize', draw);
draw();
</script>
</body>
</html>
"""

out_eq = SCRATCHPAD / "equity_curve.html"
out_eq.write_text(EQUITY_HTML, encoding="utf-8")
print(f"\nWrote: {out_eq}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. COVERAGE-PRECISION (directional DA) CURVE
# ─────────────────────────────────────────────────────────────────────────────

def compute_coverage_da_sweep(records: list[dict]) -> list[tuple[float, float]]:
    """Sweep tau from 0 to 0.5 in 51 steps; return [(coverage, directional_da), ...]."""
    # filter records with actual_return_7d
    valid = [r for r in records if r.get("actual_return_7d") is not None]
    if not valid:
        return []
    total = len(valid)
    result: list[tuple[float, float]] = []
    for tau_int in range(51):
        tau = tau_int / 100.0  # 0.00 to 0.50
        active = []
        for r in valid:
            p_up = r.get("p_up", 0.0) or 0.0
            p_down = r.get("p_down", 0.0) or 0.0
            # take signal if max(p_up, p_down) > 0.5 + tau
            max_p = max(p_up, p_down)
            if max_p >= 0.5 + tau:
                active.append(r)
        coverage = len(active) / total if total > 0 else 0.0
        if not active:
            result.append((round(coverage, 4), 0.0))
            continue
        correct = 0
        for r in active:
            p_up = r.get("p_up", 0.0) or 0.0
            p_down = r.get("p_down", 0.0) or 0.0
            ret7d = r.get("actual_return_7d", 0.0) or 0.0
            pred_dir = 1 if p_up >= p_down else -1
            actual_dir = 1 if ret7d > 0 else -1
            if pred_dir == actual_dir:
                correct += 1
        da = correct / len(active)
        result.append((round(coverage, 4), round(da, 4)))
    return result

cp_series: dict[str, list[tuple[float, float]]] = {}
for model_name, fname in PROB_MODELS.items():
    path = PRED_DIR / fname
    if not path.exists():
        print(f"  SKIP (not found): {fname}")
        continue
    recs = load_jsonl(path)
    # check if it has p_up
    if not any("p_up" in r for r in recs):
        print(f"  SKIP (no p_up): {fname}")
        continue
    curve = compute_coverage_da_sweep(recs)
    cp_series[model_name] = curve
    print(f"  coverage-DA {model_name}: {len(recs)} records, tau sweep done")

cp_js_data = json.dumps({
    "series": {
        name: {"coverage": [p[0] for p in pts], "da": [p[1] for p in pts]}
        for name, pts in cp_series.items()
    },
    "colors": {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(cp_series.keys())},
})

CP_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Coverage–Precision Curve — MarketLens</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', system-ui, sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; color: #f0f6fc; }}
  p.subtitle {{ font-size: 0.85rem; color: #8b949e; margin-bottom: 20px; }}
  #chart-wrap {{ position: relative; width: 100%; }}
  canvas {{ width: 100% !important; display: block; border-radius: 8px; background: #161b22; }}
  #legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }}
  .leg-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #c9d1d9; cursor: pointer; user-select: none; }}
  .leg-swatch {{ width: 28px; height: 3px; border-radius: 2px; }}
  #tooltip {{
    position: fixed; pointer-events: none;
    background: #21262d; border: 1px solid #30363d;
    border-radius: 6px; padding: 8px 12px;
    font-size: 0.78rem; color: #e6edf3; line-height: 1.6;
    display: none; z-index: 100; max-width: 240px;
  }}
</style>
</head>
<body>
<h1>Coverage–Precision Curve</h1>
<p class="subtitle">X: coverage (fraction of test days with active signal) · Y: directional DA · tau swept 0 → 0.50</p>
<div id="chart-wrap">
  <canvas id="cv"></canvas>
</div>
<div id="legend"></div>
<div id="tooltip"></div>

<script>
const DATA = {cp_js_data};

const MARGIN = {{ top: 24, right: 24, bottom: 60, left: 72 }};
const canvas = document.getElementById('cv');
const tooltip = document.getElementById('tooltip');
const legendEl = document.getElementById('legend');

const names = Object.keys(DATA.series);
const hidden = new Set();
names.forEach((name, i) => {{
  const color = DATA.colors[name];
  const item = document.createElement('div');
  item.className = 'leg-item';
  item.innerHTML = `<div class="leg-swatch" style="background:${{color}}"></div><span>${{name.replace(/_/g,' ')}}</span>`;
  item.addEventListener('click', () => {{
    if (hidden.has(name)) hidden.delete(name); else hidden.add(name);
    item.style.opacity = hidden.has(name) ? '0.35' : '1';
    draw();
  }});
  legendEl.appendChild(item);
}});

function getSize() {{
  const wrap = document.getElementById('chart-wrap');
  return {{ w: wrap.clientWidth, h: Math.min(520, Math.max(340, wrap.clientWidth * 0.5)) }};
}}

function draw() {{
  const {{ w, h }} = getSize();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pw = w - MARGIN.left - MARGIN.right;
  const ph = h - MARGIN.top - MARGIN.bottom;

  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  // fixed axes: x [0,1], y [0,1]
  const xScale = v => MARGIN.left + v * pw;
  const yScale = v => MARGIN.top + ph - v * ph;

  // grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  for (let g = 0; g <= 5; g++) {{
    const gv = g / 5;
    const gy = yScale(gv);
    ctx.beginPath(); ctx.moveTo(MARGIN.left, gy); ctx.lineTo(MARGIN.left + pw, gy); ctx.stroke();
    ctx.fillStyle = '#6e7681'; ctx.font = '11px system-ui'; ctx.textAlign = 'right';
    ctx.fillText((gv * 100).toFixed(0) + '%', MARGIN.left - 8, gy + 4);
    const gx = xScale(gv);
    ctx.beginPath(); ctx.moveTo(gx, MARGIN.top); ctx.lineTo(gx, MARGIN.top + ph); ctx.stroke();
    ctx.fillStyle = '#6e7681'; ctx.font = '11px system-ui'; ctx.textAlign = 'center';
    ctx.fillText((gv * 100).toFixed(0) + '%', gx, MARGIN.top + ph + 18);
  }}

  // random baseline at y=0.5
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(MARGIN.left, yScale(0.5)); ctx.lineTo(MARGIN.left + pw, yScale(0.5)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#6e7681'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
  ctx.fillText('random (50%)', MARGIN.left + 4, yScale(0.5) - 4);

  // axes border
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(MARGIN.left, MARGIN.top); ctx.lineTo(MARGIN.left, MARGIN.top + ph);
  ctx.lineTo(MARGIN.left + pw, MARGIN.top + ph);
  ctx.stroke();

  // series
  names.forEach(name => {{
    if (hidden.has(name)) return;
    const s = DATA.series[name];
    const color = DATA.colors[name];
    ctx.strokeStyle = color;
    ctx.lineWidth = name.startsWith('cem_rag') ? 2.5 : 1.8;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.coverage.length; i++) {{
      const x = xScale(s.coverage[i]), y = yScale(s.da[i]);
      if (!started) {{ ctx.moveTo(x, y); started = true; }}
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    // dot at tau=0 (last point, highest coverage)
    const last = s.coverage.length - 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(xScale(s.coverage[last]), yScale(s.da[last]), 4, 0, Math.PI * 2);
    ctx.fill();
    // dot at start (highest tau, lowest coverage)
    ctx.beginPath();
    ctx.arc(xScale(s.coverage[0]), yScale(s.da[0]), 3, 0, Math.PI * 2);
    ctx.fill();
  }});

  // axis labels
  ctx.fillStyle = '#8b949e'; ctx.font = '12px system-ui'; ctx.textAlign = 'center';
  ctx.fillText('Coverage', MARGIN.left + pw / 2, h - MARGIN.bottom + 36);
  ctx.save();
  ctx.translate(14, MARGIN.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Directional DA', 0, 0);
  ctx.restore();
}}

// hover tooltip (find nearest point by coverage distance)
canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const {{ w, h }} = getSize();
  const pw = w - MARGIN.left - MARGIN.right;
  const ph = h - MARGIN.top - MARGIN.bottom;
  const cx = (mx - MARGIN.left) / pw;  // coverage value under cursor
  if (cx < -0.05 || cx > 1.05) {{ tooltip.style.display = 'none'; return; }}
  let html = `<strong>Coverage ≈ ${{(cx * 100).toFixed(0)}}%</strong><br>`;
  names.forEach(name => {{
    if (hidden.has(name)) return;
    const s = DATA.series[name];
    // find closest coverage
    let best = 0, bestDist = Infinity;
    for (let i = 0; i < s.coverage.length; i++) {{
      const d = Math.abs(s.coverage[i] - cx);
      if (d < bestDist) {{ bestDist = d; best = i; }}
    }}
    const color = DATA.colors[name];
    html += `<span style="color:${{color}}">${{name.replace(/_/g,' ')}}</span>: DA ${{(s.da[best]*100).toFixed(1)}}%<br>`;
  }});
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY - 20) + 'px';
}});
canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

window.addEventListener('resize', draw);
draw();
</script>
</body>
</html>
"""

out_cp = SCRATCHPAD / "coverage_precision.html"
out_cp.write_text(CP_HTML, encoding="utf-8")
print(f"Wrote: {out_cp}")
print("\nAll done.")
