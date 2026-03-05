"""
ES/SPX Options Market Microstructure Dashboard Generator
=========================================================
Reads the options order book CSV and generates a self-contained interactive
HTML dashboard with multiple synchronized visualizations.

Data columns:
  - timestamp: high-frequency timestamps (~15s window)
  - Side: Bid / Ask
  - future_strike: ES futures strike price (0.25 increments)
  - MBO_pulling_stacking: net order flow (+stacking / -pulling)
  - current_es_price, spx_strike, spx_price: underlying prices
  - t: time to expiry (years)
  - call/put Greeks: delta, gamma, theta, vega, vanna, vomma, charm, rho
  - MBO_1..MBO_14: individual order sizes at each depth level

Usage:
    python3 generate_dashboard.py
    # Opens dashboard.html in the browser
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import webbrowser
import os

CSV_PATH = "/Users/fabricioponssamano/finance_cpsc481/fixed_output 2.csv"
OUTPUT_PATH = "/Users/fabricioponssamano/finance_cpsc481/dashboard.html"

# ── Load & preprocess ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["future_strike"] = df["future_strike"].astype(float)
df["MBO_pulling_stacking"] = df["MBO_pulling_stacking"].astype(float)

mbo_cols = [f"MBO_{i}" for i in range(1, 15)]
for c in mbo_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df["total_depth"] = df[mbo_cols].sum(axis=1)

# Assign a sequential snapshot index per unique timestamp
ts_order = sorted(df["timestamp"].unique())
ts_map = {t: i for i, t in enumerate(ts_order)}
df["snap_idx"] = df["timestamp"].map(ts_map)

n_snaps = len(ts_order)
print(f"  {len(df)} rows, {n_snaps} snapshots, strikes {df['future_strike'].min()}-{df['future_strike'].max()}")

# ── Precompute per-snapshot aggregates ─────────────────────────────────────────
print("Precomputing aggregates...")

# 1) For each snapshot: bid/ask depth profile by strike
strikes_sorted = sorted(df["future_strike"].unique())
strike_to_idx = {s: i for i, s in enumerate(strikes_sorted)}

# Sample every Nth snapshot for the animated heatmap (keep it responsive)
SNAP_STEP = max(1, n_snaps // 80)
sampled_snaps = list(range(0, n_snaps, SNAP_STEP))

# ── Build animated order book heatmap data ─────────────────────────────────────
print("Building order book heatmap frames...")

ask_frames = []
bid_frames = []
imbalance_frames = []
pull_stack_frames = []

for snap in sampled_snaps:
    snap_df = df[df["snap_idx"] == snap]

    ask_depth = np.zeros(len(strikes_sorted))
    bid_depth = np.zeros(len(strikes_sorted))
    pull_stack = np.zeros(len(strikes_sorted))
    pull_count = np.zeros(len(strikes_sorted))

    for _, row in snap_df.iterrows():
        idx = strike_to_idx[row["future_strike"]]
        if row["Side"] == "Ask":
            ask_depth[idx] = row["total_depth"]
        else:
            bid_depth[idx] = row["total_depth"]
        pull_stack[idx] += row["MBO_pulling_stacking"]
        pull_count[idx] += 1

    # Avoid division by zero
    pull_count[pull_count == 0] = 1
    pull_stack_avg = pull_stack / pull_count

    ask_frames.append(ask_depth.tolist())
    bid_frames.append(bid_depth.tolist())
    imbalance_frames.append((bid_depth - ask_depth).tolist())
    pull_stack_frames.append(pull_stack_avg.tolist())

# ── Greeks profile (use first snapshot as default, update via JS) ──────────────
print("Building Greeks profiles...")

# Aggregate Greeks by strike (one row per spx_strike)
first_snap_df = df[df["snap_idx"] == 0]
greeks_df = first_snap_df.drop_duplicates(subset=["spx_strike"]).sort_values("spx_strike")

# Build Greeks data for all sampled snapshots
greeks_by_snap = {}
for snap in sampled_snaps:
    sdf = df[df["snap_idx"] == snap].drop_duplicates(subset=["spx_strike"]).sort_values("spx_strike")
    greeks_by_snap[snap] = {
        "spx_strike": sdf["spx_strike"].tolist(),
        "call_delta": sdf["call_delta"].tolist(),
        "put_delta": sdf["put_delta"].tolist(),
        "call_gamma": sdf["call_gamma"].tolist(),
        "call_vega": sdf["call_vega"].tolist(),
        "call_theta": sdf["call_theta"].tolist(),
        "call_vanna": sdf["call_vanna"].tolist(),
        "call_vomma": sdf["call_vomma"].tolist(),
        "spx_price": sdf["spx_price"].iloc[0] if len(sdf) > 0 else 0,
    }

# ── Time series: aggregate pulling/stacking over time ─────────────────────────
print("Building time series...")

ts_agg = df.groupby("snap_idx").agg(
    net_pull_stack=("MBO_pulling_stacking", "sum"),
    mean_pull_stack=("MBO_pulling_stacking", "mean"),
    total_ask_depth=pd.NamedAgg(column="total_depth", aggfunc=lambda x: x[df.loc[x.index, "Side"] == "Ask"].sum()),
    total_bid_depth=pd.NamedAgg(column="total_depth", aggfunc=lambda x: x[df.loc[x.index, "Side"] == "Bid"].sum()),
    es_price=("current_es_price", "first"),
).reset_index()

ts_agg["imbalance_ratio"] = (ts_agg["total_bid_depth"] - ts_agg["total_ask_depth"]) / (
    ts_agg["total_bid_depth"] + ts_agg["total_ask_depth"] + 1e-9
)

# ── 3D Greeks surface data ─────────────────────────────────────────────────────
print("Building 3D surface data...")

# For the 3D surface: delta across (strike, time)
surface_strikes = greeks_by_snap[sampled_snaps[0]]["spx_strike"]
surface_snaps = sampled_snaps[:40]  # limit for performance
surface_delta = []
surface_gamma = []
surface_vega = []

for snap in surface_snaps:
    gd = greeks_by_snap[snap]
    surface_delta.append(gd["call_delta"])
    surface_gamma.append(gd["call_gamma"])
    surface_vega.append(gd["call_vega"])

# ── MBO depth profile heatmap data ────────────────────────────────────────────
print("Building MBO depth profiles...")

# For a selected snapshot, show the full MBO_1..MBO_14 profile
# Pre-build for first snapshot, JS will handle interactivity
def build_mbo_heatmap(snap_idx, side="Ask"):
    sdf = df[(df["snap_idx"] == snap_idx) & (df["Side"] == side)].sort_values("future_strike")
    strikes = sdf["future_strike"].values
    matrix = sdf[mbo_cols].values.T  # 14 x N_strikes
    return strikes.tolist(), matrix.tolist()

ask_strikes_mbo, ask_mbo_matrix = build_mbo_heatmap(0, "Ask")
bid_strikes_mbo, bid_mbo_matrix = build_mbo_heatmap(0, "Bid")

# Pre-build MBO heatmaps for all sampled snapshots
mbo_data = {}
for snap in sampled_snaps:
    a_s, a_m = build_mbo_heatmap(snap, "Ask")
    b_s, b_m = build_mbo_heatmap(snap, "Bid")
    mbo_data[snap] = {"ask_strikes": a_s, "ask_matrix": a_m, "bid_strikes": b_s, "bid_matrix": b_m}

# ── Build the HTML dashboard ──────────────────────────────────────────────────
print("Generating HTML dashboard...")

# Prepare timestamp labels
snap_labels = {snap: str(ts_order[snap])[-15:-6] for snap in sampled_snaps}

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ES/SPX Options Market Microstructure Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0a0e17;
    color: #c8d6e5;
    overflow-x: hidden;
  }}
  .header {{
    background: linear-gradient(135deg, #0f1923 0%, #1a2332 100%);
    padding: 18px 32px;
    border-bottom: 1px solid #1e3a5f;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .header h1 {{
    font-size: 20px;
    font-weight: 600;
    color: #e8f0fe;
    letter-spacing: 0.5px;
  }}
  .header .subtitle {{
    font-size: 12px;
    color: #6b8299;
    margin-top: 2px;
  }}
  .header .stats {{
    display: flex;
    gap: 24px;
  }}
  .stat-box {{
    text-align: center;
  }}
  .stat-box .val {{
    font-size: 18px;
    font-weight: 700;
    color: #4fc3f7;
  }}
  .stat-box .lbl {{
    font-size: 10px;
    color: #6b8299;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .controls {{
    background: #0f1923;
    padding: 12px 32px;
    border-bottom: 1px solid #1e3a5f;
    display: flex;
    align-items: center;
    gap: 20px;
  }}
  .controls label {{
    font-size: 12px;
    color: #6b8299;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .controls input[type=range] {{
    flex: 1;
    accent-color: #4fc3f7;
    height: 6px;
  }}
  .controls .time-display {{
    font-family: 'Courier New', monospace;
    font-size: 14px;
    color: #4fc3f7;
    min-width: 120px;
  }}
  .controls button {{
    background: #1e3a5f;
    border: 1px solid #2d5a8e;
    color: #4fc3f7;
    padding: 6px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .controls button:hover {{ background: #2d5a8e; }}
  .controls button.active {{ background: #4fc3f7; color: #0a0e17; }}
  .tabs {{
    display: flex;
    background: #0f1923;
    border-bottom: 2px solid #1e3a5f;
    padding: 0 32px;
  }}
  .tab {{
    padding: 12px 24px;
    cursor: pointer;
    color: #6b8299;
    font-size: 13px;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .tab:hover {{ color: #c8d6e5; }}
  .tab.active {{
    color: #4fc3f7;
    border-bottom-color: #4fc3f7;
  }}
  .tab-content {{
    display: none;
    padding: 16px 32px;
  }}
  .tab-content.active {{ display: block; }}
  .grid {{
    display: grid;
    gap: 16px;
  }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .panel {{
    background: #0f1923;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    overflow: hidden;
  }}
  .panel-title {{
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 600;
    color: #4fc3f7;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #1e3a5f;
    background: rgba(79,195,247,0.05);
  }}
  .plot {{ width: 100%; }}
  .greek-toggles {{
    display: flex;
    gap: 8px;
    padding: 8px 16px;
    flex-wrap: wrap;
  }}
  .greek-btn {{
    background: #1a2332;
    border: 1px solid #2d5a8e;
    color: #6b8299;
    padding: 4px 12px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 11px;
    transition: all 0.2s;
  }}
  .greek-btn:hover {{ border-color: #4fc3f7; color: #c8d6e5; }}
  .greek-btn.active {{ background: #4fc3f7; color: #0a0e17; border-color: #4fc3f7; }}
  .legend-bar {{
    display: flex;
    gap: 16px;
    padding: 8px 16px;
    font-size: 11px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }}
  @media (max-width: 1200px) {{
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>ES/SPX Options Market Microstructure</h1>
    <div class="subtitle">Order Book Depth | Greeks Landscape | Flow Analysis</div>
  </div>
  <div class="stats">
    <div class="stat-box">
      <div class="val">{len(strikes_sorted)}</div>
      <div class="lbl">Strikes</div>
    </div>
    <div class="stat-box">
      <div class="val">{n_snaps}</div>
      <div class="lbl">Snapshots</div>
    </div>
    <div class="stat-box">
      <div class="val" id="es-price">{df['current_es_price'].iloc[0]/100:.2f}</div>
      <div class="lbl">ES Price</div>
    </div>
    <div class="stat-box">
      <div class="val" id="spx-price">{df['spx_price'].iloc[0]:.2f}</div>
      <div class="lbl">SPX Price</div>
    </div>
  </div>
</div>

<div class="controls">
  <label>Time</label>
  <input type="range" id="time-slider" min="0" max="{len(sampled_snaps)-1}" value="0" step="1">
  <span class="time-display" id="time-label">{snap_labels[sampled_snaps[0]]}</span>
  <button id="play-btn" onclick="togglePlay()">Play</button>
  <button id="speed-btn" onclick="cycleSpeed()">1x</button>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab(0)">Order Book</div>
  <div class="tab" onclick="switchTab(1)">Greeks</div>
  <div class="tab" onclick="switchTab(2)">Flow Analysis</div>
  <div class="tab" onclick="switchTab(3)">3D Surface</div>
  <div class="tab" onclick="switchTab(4)">MBO Depth</div>
</div>

<!-- Tab 0: Order Book -->
<div class="tab-content active" id="tab-0">
  <div class="grid grid-2">
    <div class="panel">
      <div class="panel-title">Bid Depth by Strike</div>
      <div id="plot-bid-depth" class="plot"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Ask Depth by Strike</div>
      <div id="plot-ask-depth" class="plot"></div>
    </div>
  </div>
  <div class="grid grid-2" style="margin-top:16px;">
    <div class="panel">
      <div class="panel-title">Order Book Imbalance (Bid - Ask)</div>
      <div id="plot-imbalance" class="plot"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Pulling / Stacking Flow</div>
      <div id="plot-pullstack" class="plot"></div>
    </div>
  </div>
</div>

<!-- Tab 1: Greeks -->
<div class="tab-content" id="tab-1">
  <div class="greek-toggles">
    <button class="greek-btn active" onclick="toggleGreek('call_delta',this)">Call Delta</button>
    <button class="greek-btn active" onclick="toggleGreek('put_delta',this)">Put Delta</button>
    <button class="greek-btn" onclick="toggleGreek('call_gamma',this)">Gamma</button>
    <button class="greek-btn" onclick="toggleGreek('call_vega',this)">Vega</button>
    <button class="greek-btn" onclick="toggleGreek('call_theta',this)">Theta</button>
    <button class="greek-btn" onclick="toggleGreek('call_vanna',this)">Vanna</button>
    <button class="greek-btn" onclick="toggleGreek('call_vomma',this)">Vomma</button>
  </div>
  <div class="panel">
    <div class="panel-title">Options Greeks by Strike</div>
    <div id="plot-greeks" class="plot"></div>
  </div>
</div>

<!-- Tab 2: Flow Analysis -->
<div class="tab-content" id="tab-2">
  <div class="grid grid-2">
    <div class="panel">
      <div class="panel-title">Net Pulling/Stacking Over Time</div>
      <div id="plot-flow-ts" class="plot"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Bid/Ask Imbalance Ratio Over Time</div>
      <div id="plot-imb-ts" class="plot"></div>
    </div>
  </div>
  <div class="grid grid-2" style="margin-top:16px;">
    <div class="panel">
      <div class="panel-title">Total Bid vs Ask Depth Over Time</div>
      <div id="plot-depth-ts" class="plot"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Cumulative Net Flow</div>
      <div id="plot-cumflow" class="plot"></div>
    </div>
  </div>
</div>

<!-- Tab 3: 3D Surface -->
<div class="tab-content" id="tab-3">
  <div class="grid grid-2">
    <div class="panel">
      <div class="panel-title">Call Delta Surface (Strike x Time)</div>
      <div id="plot-3d-delta" class="plot"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Call Gamma Surface (Strike x Time)</div>
      <div id="plot-3d-gamma" class="plot"></div>
    </div>
  </div>
</div>

<!-- Tab 4: MBO Depth -->
<div class="tab-content" id="tab-4">
  <div class="grid grid-2">
    <div class="panel">
      <div class="panel-title">Ask Side - Order Book Depth Levels (MBO 1-14)</div>
      <div id="plot-mbo-ask" class="plot"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Bid Side - Order Book Depth Levels (MBO 1-14)</div>
      <div id="plot-mbo-bid" class="plot"></div>
    </div>
  </div>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────────
const strikes = {json.dumps(strikes_sorted)};
const sampledSnaps = {json.dumps(sampled_snaps)};
const snapLabels = {json.dumps(snap_labels)};
const askFrames = {json.dumps(ask_frames)};
const bidFrames = {json.dumps(bid_frames)};
const imbalanceFrames = {json.dumps(imbalance_frames)};
const pullStackFrames = {json.dumps(pull_stack_frames)};
const greeksBySnap = {json.dumps(greeks_by_snap)};
const tsAgg = {{
  snapIdx: {ts_agg["snap_idx"].tolist()},
  netPullStack: {ts_agg["net_pull_stack"].tolist()},
  meanPullStack: {ts_agg["mean_pull_stack"].tolist()},
  totalAskDepth: {ts_agg["total_ask_depth"].tolist()},
  totalBidDepth: {ts_agg["total_bid_depth"].tolist()},
  imbalanceRatio: {ts_agg["imbalance_ratio"].tolist()},
}};
const surfaceStrikes = {json.dumps(surface_strikes)};
const surfaceDelta = {json.dumps(surface_delta)};
const surfaceGamma = {json.dumps(surface_gamma)};
const mboData = {json.dumps(mbo_data)};

// ── State ─────────────────────────────────────────────────────────────────────
let currentFrame = 0;
let playing = false;
let playInterval = null;
let playSpeed = 1;
const speeds = [1, 2, 4, 8];
let speedIdx = 0;
let visibleGreeks = {{'call_delta': true, 'put_delta': true, 'call_gamma': false, 'call_vega': false, 'call_theta': false, 'call_vanna': false, 'call_vomma': false}};

const plotLayout = {{
  paper_bgcolor: '#0f1923',
  plot_bgcolor: '#0a0e17',
  font: {{ color: '#c8d6e5', family: 'Segoe UI, system-ui, sans-serif', size: 11 }},
  margin: {{ t: 20, b: 40, l: 55, r: 20 }},
  xaxis: {{ gridcolor: '#1e3a5f', zerolinecolor: '#2d5a8e' }},
  yaxis: {{ gridcolor: '#1e3a5f', zerolinecolor: '#2d5a8e' }},
  height: 320,
}};

const plotConfig = {{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d','select2d'] }};

// ── Tab 0: Order Book ─────────────────────────────────────────────────────────
function initOrderBook() {{
  const bidTrace = {{
    x: strikes, y: bidFrames[0], type: 'bar',
    marker: {{ color: 'rgba(76,175,80,0.7)', line: {{ width: 0 }} }},
    name: 'Bid Depth'
  }};
  Plotly.newPlot('plot-bid-depth', [bidTrace], {{
    ...plotLayout,
    xaxis: {{ ...plotLayout.xaxis, title: 'Strike' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Total Contracts' }},
  }}, plotConfig);

  const askTrace = {{
    x: strikes, y: askFrames[0], type: 'bar',
    marker: {{ color: 'rgba(244,67,54,0.7)', line: {{ width: 0 }} }},
    name: 'Ask Depth'
  }};
  Plotly.newPlot('plot-ask-depth', [askTrace], {{
    ...plotLayout,
    xaxis: {{ ...plotLayout.xaxis, title: 'Strike' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Total Contracts' }},
  }}, plotConfig);

  const imbTrace = {{
    x: strikes, y: imbalanceFrames[0], type: 'bar',
    marker: {{
      color: imbalanceFrames[0].map(v => v >= 0 ? 'rgba(76,175,80,0.8)' : 'rgba(244,67,54,0.8)'),
      line: {{ width: 0 }}
    }},
    name: 'Imbalance'
  }};
  Plotly.newPlot('plot-imbalance', [imbTrace], {{
    ...plotLayout,
    xaxis: {{ ...plotLayout.xaxis, title: 'Strike' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Bid - Ask' }},
  }}, plotConfig);

  const psTrace = {{
    x: strikes, y: pullStackFrames[0], type: 'bar',
    marker: {{
      color: pullStackFrames[0].map(v => v >= 0 ? 'rgba(79,195,247,0.8)' : 'rgba(255,152,0,0.8)'),
      line: {{ width: 0 }}
    }},
    name: 'Pull/Stack'
  }};
  Plotly.newPlot('plot-pullstack', [psTrace], {{
    ...plotLayout,
    xaxis: {{ ...plotLayout.xaxis, title: 'Strike' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Avg Pull(-)/Stack(+)' }},
  }}, plotConfig);
}}

function updateOrderBook(frame) {{
  Plotly.restyle('plot-bid-depth', {{ y: [bidFrames[frame]] }});
  Plotly.restyle('plot-ask-depth', {{ y: [askFrames[frame]] }});
  Plotly.restyle('plot-imbalance', {{
    y: [imbalanceFrames[frame]],
    'marker.color': [imbalanceFrames[frame].map(v => v >= 0 ? 'rgba(76,175,80,0.8)' : 'rgba(244,67,54,0.8)')]
  }});
  Plotly.restyle('plot-pullstack', {{
    y: [pullStackFrames[frame]],
    'marker.color': [pullStackFrames[frame].map(v => v >= 0 ? 'rgba(79,195,247,0.8)' : 'rgba(255,152,0,0.8)')]
  }});
}}

// ── Tab 1: Greeks ─────────────────────────────────────────────────────────────
const greekColors = {{
  call_delta: '#4fc3f7',
  put_delta: '#ff7043',
  call_gamma: '#66bb6a',
  call_vega: '#ab47bc',
  call_theta: '#ffa726',
  call_vanna: '#26c6da',
  call_vomma: '#ef5350',
}};
const greekNames = {{
  call_delta: 'Call Delta',
  put_delta: 'Put Delta',
  call_gamma: 'Gamma',
  call_vega: 'Vega',
  call_theta: 'Theta',
  call_vanna: 'Vanna',
  call_vomma: 'Vomma',
}};

function getGreeksTraces(snap) {{
  const snapKey = sampledSnaps[snap].toString();
  const gd = greeksBySnap[snapKey];
  if (!gd) return [];
  const traces = [];
  for (const [key, visible] of Object.entries(visibleGreeks)) {{
    if (visible) {{
      traces.push({{
        x: gd.spx_strike,
        y: gd[key],
        type: 'scatter',
        mode: 'lines',
        name: greekNames[key],
        line: {{ color: greekColors[key], width: 2 }},
      }});
    }}
  }}
  // Add SPX price reference line
  traces.push({{
    x: [gd.spx_price, gd.spx_price],
    y: [-1, 1],
    type: 'scatter',
    mode: 'lines',
    name: 'SPX Price',
    line: {{ color: '#ffffff', width: 1, dash: 'dot' }},
    showlegend: true,
  }});
  return traces;
}}

function initGreeks() {{
  const traces = getGreeksTraces(0);
  Plotly.newPlot('plot-greeks', traces, {{
    ...plotLayout,
    height: 450,
    xaxis: {{ ...plotLayout.xaxis, title: 'SPX Strike' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Greek Value' }},
    legend: {{ bgcolor: 'rgba(15,25,35,0.8)', bordercolor: '#1e3a5f', borderwidth: 1 }},
    showlegend: true,
  }}, plotConfig);
}}

function updateGreeks(frame) {{
  const traces = getGreeksTraces(frame);
  Plotly.react('plot-greeks', traces, {{
    ...plotLayout,
    height: 450,
    xaxis: {{ ...plotLayout.xaxis, title: 'SPX Strike' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Greek Value' }},
    legend: {{ bgcolor: 'rgba(15,25,35,0.8)', bordercolor: '#1e3a5f', borderwidth: 1 }},
    showlegend: true,
  }}, plotConfig);
}}

function toggleGreek(key, btn) {{
  visibleGreeks[key] = !visibleGreeks[key];
  btn.classList.toggle('active');
  updateGreeks(currentFrame);
}}

// ── Tab 2: Flow Analysis ─────────────────────────────────────────────────────
function initFlowAnalysis() {{
  // Net pull/stack over time
  const cumFlow = [];
  let cum = 0;
  for (const v of tsAgg.netPullStack) {{ cum += v; cumFlow.push(cum); }}

  Plotly.newPlot('plot-flow-ts', [{{
    x: tsAgg.snapIdx, y: tsAgg.netPullStack, type: 'scatter', mode: 'lines',
    line: {{ color: '#4fc3f7', width: 1.5 }},
    fill: 'tozeroy',
    fillcolor: 'rgba(79,195,247,0.15)',
  }}], {{
    ...plotLayout,
    xaxis: {{ ...plotLayout.xaxis, title: 'Snapshot' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Net Flow' }},
  }}, plotConfig);

  // Imbalance ratio
  Plotly.newPlot('plot-imb-ts', [{{
    x: tsAgg.snapIdx, y: tsAgg.imbalanceRatio, type: 'scatter', mode: 'lines',
    line: {{ color: '#ab47bc', width: 1.5 }},
    fill: 'tozeroy',
    fillcolor: 'rgba(171,71,188,0.15)',
  }}], {{
    ...plotLayout,
    xaxis: {{ ...plotLayout.xaxis, title: 'Snapshot' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Imbalance Ratio' }},
    shapes: [{{
      type: 'line', x0: tsAgg.snapIdx[0], x1: tsAgg.snapIdx[tsAgg.snapIdx.length-1],
      y0: 0, y1: 0, line: {{ color: '#ffffff', width: 1, dash: 'dot' }}
    }}]
  }}, plotConfig);

  // Total bid vs ask depth
  Plotly.newPlot('plot-depth-ts', [
    {{ x: tsAgg.snapIdx, y: tsAgg.totalBidDepth, type: 'scatter', mode: 'lines',
       name: 'Bid Depth', line: {{ color: '#66bb6a', width: 1.5 }} }},
    {{ x: tsAgg.snapIdx, y: tsAgg.totalAskDepth, type: 'scatter', mode: 'lines',
       name: 'Ask Depth', line: {{ color: '#ef5350', width: 1.5 }} }},
  ], {{
    ...plotLayout,
    xaxis: {{ ...plotLayout.xaxis, title: 'Snapshot' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Total Contracts' }},
    legend: {{ bgcolor: 'rgba(15,25,35,0.8)', bordercolor: '#1e3a5f', borderwidth: 1 }},
    showlegend: true,
  }}, plotConfig);

  // Cumulative net flow
  Plotly.newPlot('plot-cumflow', [{{
    x: tsAgg.snapIdx, y: cumFlow, type: 'scatter', mode: 'lines',
    line: {{ color: '#ffa726', width: 2 }},
    fill: 'tozeroy',
    fillcolor: 'rgba(255,167,38,0.15)',
  }}], {{
    ...plotLayout,
    xaxis: {{ ...plotLayout.xaxis, title: 'Snapshot' }},
    yaxis: {{ ...plotLayout.yaxis, title: 'Cumulative Net Flow' }},
  }}, plotConfig);

  // Add time cursor lines
  addTimeCursor('plot-flow-ts');
  addTimeCursor('plot-imb-ts');
  addTimeCursor('plot-depth-ts');
  addTimeCursor('plot-cumflow');
}}

function addTimeCursor(plotId) {{
  // Will be updated via updateFlowCursor
}}

function updateFlowCursor(frame) {{
  const snapIdx = sampledSnaps[frame];
  const shape = {{
    type: 'line', x0: snapIdx, x1: snapIdx,
    y0: 0, y1: 1, yref: 'paper',
    line: {{ color: '#4fc3f7', width: 2, dash: 'dash' }}
  }};
  ['plot-flow-ts', 'plot-imb-ts', 'plot-depth-ts', 'plot-cumflow'].forEach(id => {{
    const el = document.getElementById(id);
    if (el && el.layout) {{
      const existingShapes = (el.layout.shapes || []).filter(s => s._custom !== true);
      shape._custom = true;
      Plotly.relayout(id, {{ shapes: [...existingShapes, shape] }});
    }}
  }});
}}

// ── Tab 3: 3D Surface ─────────────────────────────────────────────────────────
function init3DSurface() {{
  const layout3d = {{
    paper_bgcolor: '#0f1923',
    plot_bgcolor: '#0a0e17',
    font: {{ color: '#c8d6e5', family: 'Segoe UI, system-ui, sans-serif', size: 11 }},
    margin: {{ t: 20, b: 20, l: 20, r: 20 }},
    height: 500,
    scene: {{
      xaxis: {{ title: 'Strike', gridcolor: '#1e3a5f', backgroundcolor: '#0a0e17' }},
      yaxis: {{ title: 'Time Step', gridcolor: '#1e3a5f', backgroundcolor: '#0a0e17' }},
      zaxis: {{ title: 'Delta', gridcolor: '#1e3a5f', backgroundcolor: '#0a0e17' }},
      bgcolor: '#0a0e17',
    }},
  }};

  Plotly.newPlot('plot-3d-delta', [{{
    z: surfaceDelta,
    x: surfaceStrikes,
    type: 'surface',
    colorscale: 'Viridis',
    contours: {{ z: {{ show: true, usecolormap: true, highlightcolor: '#4fc3f7', project: {{ z: true }} }} }},
  }}], layout3d, plotConfig);

  const layout3dGamma = JSON.parse(JSON.stringify(layout3d));
  layout3dGamma.scene.zaxis.title = 'Gamma';

  Plotly.newPlot('plot-3d-gamma', [{{
    z: surfaceGamma,
    x: surfaceStrikes,
    type: 'surface',
    colorscale: 'Plasma',
    contours: {{ z: {{ show: true, usecolormap: true, highlightcolor: '#66bb6a', project: {{ z: true }} }} }},
  }}], layout3dGamma, plotConfig);
}}

// ── Tab 4: MBO Depth Heatmap ──────────────────────────────────────────────────
function initMBODepth() {{
  const snapKey = sampledSnaps[0].toString();
  const md = mboData[snapKey];
  const yLabels = Array.from({{length: 14}}, (_, i) => 'Level ' + (i+1));

  Plotly.newPlot('plot-mbo-ask', [{{
    z: md.ask_matrix,
    x: md.ask_strikes,
    y: yLabels,
    type: 'heatmap',
    colorscale: [
      [0, '#0a0e17'], [0.1, '#1a237e'], [0.3, '#4fc3f7'],
      [0.5, '#66bb6a'], [0.7, '#ffa726'], [1, '#ef5350']
    ],
    colorbar: {{ title: 'Size', tickfont: {{ color: '#c8d6e5' }}, titlefont: {{ color: '#c8d6e5' }} }},
  }}], {{
    ...plotLayout,
    height: 400,
    xaxis: {{ ...plotLayout.xaxis, title: 'Strike' }},
    yaxis: {{ ...plotLayout.yaxis, title: '' }},
  }}, plotConfig);

  Plotly.newPlot('plot-mbo-bid', [{{
    z: md.bid_matrix,
    x: md.bid_strikes,
    y: yLabels,
    type: 'heatmap',
    colorscale: [
      [0, '#0a0e17'], [0.1, '#1b5e20'], [0.3, '#66bb6a'],
      [0.5, '#4fc3f7'], [0.7, '#ffa726'], [1, '#ef5350']
    ],
    colorbar: {{ title: 'Size', tickfont: {{ color: '#c8d6e5' }}, titlefont: {{ color: '#c8d6e5' }} }},
  }}], {{
    ...plotLayout,
    height: 400,
    xaxis: {{ ...plotLayout.xaxis, title: 'Strike' }},
    yaxis: {{ ...plotLayout.yaxis, title: '' }},
  }}, plotConfig);
}}

function updateMBODepth(frame) {{
  const snapKey = sampledSnaps[frame].toString();
  const md = mboData[snapKey];
  if (!md) return;
  Plotly.restyle('plot-mbo-ask', {{ z: [md.ask_matrix], x: [md.ask_strikes] }});
  Plotly.restyle('plot-mbo-bid', {{ z: [md.bid_matrix], x: [md.bid_strikes] }});
}}

// ── Tab Management ────────────────────────────────────────────────────────────
let activeTab = 0;
const tabInited = [false, false, false, false, false];

function switchTab(idx) {{
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', i === idx));
  document.querySelectorAll('.tab-content').forEach((t, i) => t.classList.toggle('active', i === idx));
  activeTab = idx;
  if (!tabInited[idx]) {{
    tabInited[idx] = true;
    if (idx === 1) initGreeks();
    if (idx === 2) initFlowAnalysis();
    if (idx === 3) init3DSurface();
    if (idx === 4) initMBODepth();
  }}
  // Trigger resize for proper rendering
  setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
}}

// ── Playback ──────────────────────────────────────────────────────────────────
function togglePlay() {{
  playing = !playing;
  const btn = document.getElementById('play-btn');
  if (playing) {{
    btn.textContent = 'Pause';
    btn.classList.add('active');
    playInterval = setInterval(() => {{
      currentFrame = (currentFrame + 1) % sampledSnaps.length;
      document.getElementById('time-slider').value = currentFrame;
      onFrameChange(currentFrame);
    }}, 200 / playSpeed);
  }} else {{
    btn.textContent = 'Play';
    btn.classList.remove('active');
    clearInterval(playInterval);
  }}
}}

function cycleSpeed() {{
  speedIdx = (speedIdx + 1) % speeds.length;
  playSpeed = speeds[speedIdx];
  document.getElementById('speed-btn').textContent = playSpeed + 'x';
  if (playing) {{
    clearInterval(playInterval);
    playInterval = setInterval(() => {{
      currentFrame = (currentFrame + 1) % sampledSnaps.length;
      document.getElementById('time-slider').value = currentFrame;
      onFrameChange(currentFrame);
    }}, 200 / playSpeed);
  }}
}}

function onFrameChange(frame) {{
  currentFrame = frame;
  document.getElementById('time-label').textContent = snapLabels[sampledSnaps[frame]] || '';
  if (activeTab === 0) updateOrderBook(frame);
  if (activeTab === 1) updateGreeks(frame);
  if (activeTab === 2) updateFlowCursor(frame);
  if (activeTab === 4) updateMBODepth(frame);
}}

document.getElementById('time-slider').addEventListener('input', function() {{
  onFrameChange(parseInt(this.value));
}});

// ── Init ──────────────────────────────────────────────────────────────────────
initOrderBook();
tabInited[0] = true;

</script>
</body>
</html>
"""

with open(OUTPUT_PATH, "w") as f:
    f.write(html)

print(f"\nDashboard saved to: {OUTPUT_PATH}")
print("Opening in browser...")
webbrowser.open("file://" + OUTPUT_PATH)
