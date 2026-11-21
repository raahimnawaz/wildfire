"""
California wildfire risk map — an operations view of one day's ignition scores.

Reads a scored parquet (output of ``wildfire.dashboard``) and writes one
self-contained HTML page built around three ideas:

  * The answer comes first. A header carries the day's risk level and the
    health of the inputs; the panel opens with one sentence saying how many
    cells are on alert and where they are.
  * Colour is reserved for signal. Cells below the operational threshold are
    drawn in quiet slate so the state reads as a calm surface; only alert
    cells are warm, and they shimmer (two summed sinusoids per cell, a
    fundamental and a golden-ratio overtone). The five largest alert
    clusters get a slow breathing ring.
  * Every number can be traced. Click a cell, county or hotspot and the panel
    becomes an inspector showing the inputs and how much each one moved the
    model's score (XGBoost per-feature contributions, which sum exactly to
    the cell's log-odds).

Map: probabilities rasterized onto a 2 km grid clipped to California, rows
evenly spaced in Web Mercator so cells are square and line up with the
basemap. GEE samples ~37% of cells per window; unsampled land cells take the
mean of sampled cells in the surrounding 3x3 block (5x5 if empty) and are
labelled as filled.

Risk level (header) is a display band on the share of scored cells at or
above the validation-tuned threshold: Low < 0.1%, Moderate < 1%,
Elevated < 5%, Severe >= 5%. The bands are presentation choices, not model
outputs, and the page says so.

Usage:
    python -m wildfire.heatmap --date 2025-09-15
    python -m wildfire.heatmap --input wildfire_output/2025-09-15/ignition_2025-09-15.parquet
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import zlib
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import region
from .config import CFG

# Colour ramp in multiples of the operational threshold t. Below t the ramp
# stays in dark slate (quiet); at t it jumps to amber and runs to red by 3t,
# so crossing the threshold is a visible event, not a gradual drift.
RAMP = [
    (0.0,   "#10161e"),
    (0.5,   "#1b2835"),
    (0.999, "#3a5670"),
    (1.0,   "#f2b134"),
    (2.0,   "#ff6a1a"),
    (3.0,   "#ff2d4a"),
]

# (upper bound on alert share, label, colour)
LEVELS = [
    (0.001, "Low",      "#9fb3c8"),
    (0.01,  "Moderate", "#7fdcff"),
    (0.05,  "Elevated", "#f2b134"),
    (1.01,  "Severe",   "#ff2d4a"),
]

FEATURE_LABELS = {
    "temp_c": ("Temperature", "°C", 1),
    "dewpoint_c": ("Dew point", "°C", 1),
    "rh_pct": ("Relative humidity", "%", 0),
    "vpd_kpa": ("Vapour pressure deficit", "kPa", 2),
    "wind_speed_ms": ("Wind speed", "m/s", 1),
    "wind_dir_deg": ("Wind direction", "°", 0),
    "soil_moist_surf": ("Surface soil moisture", "m³/m³", 2),
    "soil_moist_root": ("Root-zone soil moisture", "m³/m³", 2),
    "kbdi_index": ("Drought index (KBDI)", "", 0),
    "ndvi": ("Vegetation (NDVI)", "", 2),
    "elevation_m": ("Elevation", "m", 0),
    "slope_deg": ("Slope", "°", 1),
    "aspect_deg": ("Aspect", "°", 0),
    "month": ("Month", "", 0),
    "day_of_year": ("Day of year", "", 0),
}

MAX_ZOOM = 11      # 2 km cells are ~15 px here; closer adds pixels, not information
EMPTY = 0xFFFF
N_HOTSPOTS = 5
CONTRIB_SCALE = 16  # int8 log-odds contributions, 1/16 resolution, +-8 range


def _hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def ramp_color(p: float, threshold: float) -> str:
    """Hex colour for probability p on the threshold-anchored ramp."""
    x = p / threshold if threshold > 0 else 0.0
    for (x0, c0), (x1, c1) in zip(RAMP, RAMP[1:]):
        if x <= x1:
            f = max(0.0, (x - x0) / (x1 - x0))
            a, b = _hex_rgb(c0), _hex_rgb(c1)
            return "#" + "".join(f"{round(u + (v - u) * f):02x}" for u, v in zip(a, b))
    return RAMP[-1][1]


def risk_level(alert_share: float) -> tuple[str, str]:
    for bound, label, colour in LEVELS:
        if alert_share < bound:
            return label, colour
    return LEVELS[-1][1], LEVELS[-1][2]


# ─────────────────────────────────────────────────────────────────────────────
# Rasterization
# ─────────────────────────────────────────────────────────────────────────────

def _merc(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _inv_merc(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def _box_mean(values: np.ndarray, weights: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean over a (2r+1)^2 window, and the window's total weight."""
    from numpy.lib.stride_tricks import sliding_window_view
    k = 2 * r + 1
    vw = sliding_window_view(np.pad(values * weights, r), (k, k)).sum(axis=(-1, -2))
    w = sliding_window_view(np.pad(weights, r), (k, k)).sum(axis=(-1, -2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return vw / w, w


def rasterize(scored: pd.DataFrame) -> dict:
    """Bin scored samples onto the California 2 km grid.

    Returns the grid geometry plus per-cell arrays (row-major, top row first):
    p, temp, wind, kbdi (NaN where blank), flag (0 blank, 1 sampled,
    2 filled), county index (-1 outside the state), and ``cell_of_sample``,
    the flat grid index of each scored row (-1 if off-grid).
    """
    w, s, e, n = CFG.region_bbox
    lat0 = (s + n) / 2
    dlon = (CFG.cell_scale_m / 1000) / (111.32 * math.cos(math.radians(lat0)))
    dy = math.radians(dlon)  # equal Mercator steps in x and y -> square cells
    y_top = float(_merc(n))
    ncols = math.ceil((e - w) / dlon)
    nrows = math.ceil((y_top - float(_merc(s))) / dy)
    south = float(_inv_merc(y_top - nrows * dy))
    east = w + ncols * dlon

    lon = scored["longitude"].to_numpy(dtype=np.float64)
    lat = scored["latitude"].to_numpy(dtype=np.float64)
    col = np.floor((lon - w) / dlon).astype(int)
    row = np.floor((y_top - _merc(lat)) / dy).astype(int)
    ok = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    cell_of_sample = np.where(ok, row * ncols + col, -1)
    idx = cell_of_sample[ok]
    size = nrows * ncols

    count = np.bincount(idx, minlength=size).astype(np.float64)
    sampled = count > 0

    def cell_mean(column: str) -> np.ndarray:
        vals = scored[column].to_numpy(dtype=np.float64)[ok] if column in scored else np.zeros(idx.size)
        tot = np.bincount(idx, weights=vals, minlength=size)
        out = np.full(size, np.nan)
        out[sampled] = tot[sampled] / count[sampled]
        return out.reshape(nrows, ncols)

    fields = {
        "p": cell_mean("fire_probability"),
        "temp": cell_mean("temp_c"),
        "wind": cell_mean("wind_speed_ms"),
        "kbdi": cell_mean("kbdi_index"),
    }
    sampled = sampled.reshape(nrows, ncols)

    rr, cc = np.mgrid[0:nrows, 0:ncols]
    c_lon = w + (cc + 0.5) * dlon
    c_lat = _inv_merc(y_top - (rr + 0.5) * dy)
    county = region.locate(c_lon.ravel(), c_lat.ravel()).reshape(nrows, ncols)

    flag = np.where(sampled, 1, 0).astype(np.uint8)
    weight = sampled.astype(np.float64)
    for r in (1, 2):
        todo = (county >= 0) & (flag == 0)
        if not todo.any():
            break
        for arr in fields.values():
            mean, wsum = _box_mean(np.nan_to_num(arr), weight, r)
            fill = todo & (wsum > 0)
            arr[fill] = mean[fill]
        flag[todo & ~np.isnan(fields["p"])] = 2

    return {
        "bounds": {"west": w, "south": south, "east": east, "north": n},
        "nrows": nrows, "ncols": ncols, "dlon": dlon, "dy": dy, "y_top": y_top,
        "flag": flag, "county": county, "cell_of_sample": cell_of_sample, **fields,
    }


def cell_center(grid: dict, i: int) -> tuple[float, float]:
    r, c = divmod(int(i), grid["ncols"])
    lon = grid["bounds"]["west"] + (c + 0.5) * grid["dlon"]
    lat = float(_inv_merc(grid["y_top"] - (r + 0.5) * grid["dy"]))
    return lon, lat


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: drivers, counties, hotspots
# ─────────────────────────────────────────────────────────────────────────────

def contributions(scored: pd.DataFrame) -> np.ndarray | None:
    """Per-row, per-feature log-odds contributions from the shipped model
    (columns in CFG.feature_columns order, bias dropped). None if the model
    or any feature column is unavailable."""
    if not CFG.model_path().exists() or not set(CFG.feature_columns) <= set(scored.columns):
        return None
    try:
        import xgboost as xgb
        from .predict import _load_model
        model, _ = _load_model()
        X = scored[CFG.feature_columns].astype("float32").to_numpy()
        dm = xgb.DMatrix(X, feature_names=list(CFG.feature_columns))
        return model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    except Exception as exc:  # explanation is optional; the map is not
        print(f"[heatmap] drivers skipped: {exc}")
        return None


def county_table(scored: pd.DataFrame, sample_county: np.ndarray, threshold: float) -> list[dict]:
    names = region.county_names()
    df = pd.DataFrame({"c": sample_county, "p": scored["fire_probability"].to_numpy(),
                       "lon": scored["longitude"].to_numpy(), "lat": scored["latitude"].to_numpy()})
    df = df[df["c"] >= 0]
    rows = []
    for c, g in df.groupby("c"):
        top = g.nlargest(5, "p")
        rows.append({
            "i": int(c), "name": names[c], "cells": int(len(g)),
            "alerts": int((g["p"] >= threshold).sum()),
            "share": float((g["p"] >= threshold).mean()),
            "peak": float(g["p"].max()), "mean": float(g["p"].mean()),
            "top": [[round(float(r.lat), 4), round(float(r.lon), 4), round(float(r.p), 4)]
                    for r in top.itertuples()],
        })
    rows.sort(key=lambda r: (-r["alerts"], -r["peak"]))
    return rows


def hotspots(grid: dict, threshold: float, k: int = N_HOTSPOTS) -> list[dict]:
    """The k largest connected groups of alert cells (8-neighbour), ranked by
    summed probability, so one broad cluster outranks a lone hot pixel."""
    from scipy import ndimage

    p = np.nan_to_num(grid["p"])
    labels, n = ndimage.label(p >= threshold, structure=np.ones((3, 3)))
    if n == 0:
        return []
    ids = np.arange(1, n + 1)
    mass = ndimage.sum(p, labels, ids)
    names = region.county_names()
    out = []
    for lab in ids[np.argsort(-mass)][:k]:
        cells = np.flatnonzero(labels.ravel() == lab)
        rows, cols = np.divmod(cells, grid["ncols"])
        cty = grid["county"].ravel()[cells]
        cty = cty[cty >= 0]
        counties = [names[c] for c, _ in sorted(zip(*np.unique(cty, return_counts=True)),
                                                key=lambda t: -t[1])][:3]
        w, e = cols.min(), cols.max() + 1
        north_y = grid["y_top"] - rows.min() * grid["dy"]
        south_y = grid["y_top"] - (rows.max() + 1) * grid["dy"]
        west = grid["bounds"]["west"]
        clon, clat = cell_center(grid, int(rows.mean().round()) * grid["ncols"] + int(cols.mean().round()))
        out.append({
            "cells": [int(c) for c in cells],
            "n": int(cells.size),
            "peak": float(p.ravel()[cells].max()),
            "counties": counties,
            "center": [round(clat, 4), round(clon, 4)],
            "bounds": [[float(_inv_merc(south_y)), west + w * grid["dlon"]],
                       [float(_inv_merc(north_y)), west + e * grid["dlon"]]],
            "radius_m": float(max(e - w, rows.max() - rows.min() + 1) * CFG.cell_scale_m * 0.6 + 2500),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Packing
# ─────────────────────────────────────────────────────────────────────────────

def _pack_dense(grid: dict) -> str:
    """Per grid cell: p, temp, wind, kbdi (16-bit), flag, county (bytes)."""
    p = grid["p"]
    parts = [
        np.where(np.isnan(p), EMPTY, np.round(np.nan_to_num(p) * 10000)).astype("<u2"),
        np.round(np.nan_to_num(grid["temp"]) * 10).astype("<i2"),
        np.round(np.clip(np.nan_to_num(grid["wind"]), 0, 6000) * 10).astype("<u2"),
        np.round(np.clip(np.nan_to_num(grid["kbdi"]), 0, 65000)).astype("<u2"),
        grid["flag"].astype(np.uint8),
        np.where(grid["county"] < 0, 255, grid["county"]).astype(np.uint8),
    ]
    raw = b"".join(a.ravel().tobytes() for a in parts)
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def _pack_sampled(grid: dict, scored: pd.DataFrame, contrib: np.ndarray | None) -> tuple[str, dict]:
    """Per sampled cell: feature values (uint16, per-feature linear scale)
    and contributions (int8). Layout: uint32 count, uint32 cell index[count],
    uint16 values[count*F], int8 contribs[count*F] (zeros if unavailable)."""
    cos = grid["cell_of_sample"]
    keep = cos >= 0
    cells, inverse = np.unique(cos[keep], return_inverse=True)
    m, F = cells.size, len(CFG.feature_columns)
    cnt = np.bincount(inverse, minlength=m).astype(np.float64)

    def per_cell(mat: np.ndarray) -> np.ndarray:
        out = np.empty((m, mat.shape[1]))
        for j in range(mat.shape[1]):
            out[:, j] = np.bincount(inverse, weights=mat[:, j], minlength=m) / cnt
        return out

    vals = per_cell(scored.loc[keep, CFG.feature_columns].to_numpy(dtype=np.float64)
                    if set(CFG.feature_columns) <= set(scored.columns) else np.zeros((keep.sum(), F)))
    lo, hi = vals.min(axis=0), vals.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = np.round((vals - lo) / span * 65535).astype("<u2")
    if contrib is not None:
        cq = np.clip(np.round(per_cell(contrib[keep]) * CONTRIB_SCALE), -127, 127).astype(np.int8)
    else:
        cq = np.zeros((m, F), dtype=np.int8)
    raw = (np.array([m], dtype="<u4").tobytes() + cells.astype("<u4").tobytes()
           + q.tobytes() + cq.tobytes())
    info = {"lo": lo.tolist(), "span": span.tolist(), "hasDrivers": contrib is not None}
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii"), info


# ─────────────────────────────────────────────────────────────────────────────
# Panel pieces
# ─────────────────────────────────────────────────────────────────────────────

def _histogram_svg(probs: pd.Series, threshold: float, width: int = 340, height: int = 84) -> str:
    """Square-root-scaled histogram, coloured with the map ramp, threshold marked."""
    x_max = max(0.1, math.ceil(float(probs.max()) * 10) / 10)
    n_bins = 34
    counts, edges = np.histogram(probs.clip(upper=x_max), bins=n_bins, range=(0, x_max))
    top = math.sqrt(counts.max() or 1)
    pad, base = 2, height - 16
    bw = (width - 2 * pad) / n_bins
    bars = []
    for i, c in enumerate(counts):
        h = (base - 6) * math.sqrt(c) / top
        mid = (edges[i] + edges[i + 1]) / 2
        bars.append(f'<rect x="{pad + i * bw:.1f}" y="{base - h:.1f}" width="{bw - 1:.1f}" '
                    f'height="{h:.1f}" fill="{ramp_color(mid, threshold)}"/>')
    tx = pad + (width - 2 * pad) * min(threshold / x_max, 1)
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="Distribution of ignition probabilities">'
        f'{"".join(bars)}'
        f'<line x1="{pad}" y1="{base}" x2="{width - pad}" y2="{base}" class="axis"/>'
        f'<line x1="{tx:.1f}" y1="2" x2="{tx:.1f}" y2="{base}" class="thr"/>'
        f'<text x="{tx + 4:.1f}" y="10" class="thr-l">threshold {threshold:.3f}</text>'
        f'<text x="{pad}" y="{height - 2}" class="tick">0</text>'
        f'<text x="{width - pad}" y="{height - 2}" class="tick" text-anchor="end">{x_max:.1f}</text>'
        f'</svg>'
    )


def _feature_importance_rows(meta: dict, top_n: int = 6) -> str:
    fi = meta.get("feature_importance") or {}
    if not fi:
        return '<p class="muted">No feature importances in model metadata.</p>'
    items = list(fi.items())[:top_n]
    max_v = max(v for _, v in items) or 1.0
    return "".join(
        f'<div class="fi"><span>{html.escape(FEATURE_LABELS.get(k, (k,))[0])}</span>'
        f'<i style="width:{100 * v / max_v:.1f}%"></i><b>{v:.2f}</b></div>'
        for k, v in items
    )


def _join(names: list[str]) -> str:
    names = [html.escape(n) for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _fmt_day(d: str) -> str:
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%b %-d")
    except (TypeError, ValueError):
        return str(d)


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg: #000; --panel: #06070a; --raise: #0b0d12; --line: #15181e; --line-2: #22262e;
  --text: #f2f4f7; --muted: #8b94a1; --dim: #5d6572;
  --blue: #7fdcff; --amber: #f2b134; --red: #ff2d4a; --level: #9fb3c8;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }
html, body { height: 100%; background: var(--bg); color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-size: 13px; line-height: 1.45; font-feature-settings: 'tnum' 1, 'cv11' 1;
  -webkit-font-smoothing: antialiased; }
button { font: inherit; color: inherit; background: none; border: 0; cursor: pointer; }
.app { display: grid; grid-template-rows: 52px minmax(0, 1fr); height: 100vh; }
header { display: flex; align-items: center; gap: 16px; padding: 0 18px;
  border-bottom: 1px solid var(--line); background: var(--panel); min-width: 0; }
.brand { font-weight: 600; letter-spacing: -.005em; white-space: nowrap; }
.brand span { color: var(--muted); font-weight: 400; margin-left: 8px; }
.badges { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;
  min-width: 0; }
.badge { display: inline-flex; align-items: center; gap: 6px; height: 24px; padding: 0 9px;
  border: 1px solid var(--line-2); border-radius: 12px; font-size: 11.5px; color: var(--muted);
  white-space: nowrap; }
.badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: #3ecf8e; }
.badge.warn { color: var(--amber); border-color: #f2b13444; }
.badge.warn::before { background: var(--amber); }
.badge.info::before { background: var(--dim); }
.body { display: grid; grid-template-columns: minmax(0, 1fr) 400px; min-height: 0; }
.mapwrap { position: relative; height: 100%; min-height: 0; }
#map { position: absolute; inset: 0; background: var(--bg); }
.leaflet-container { background: var(--bg); font-family: inherit; }
.basemap { filter: brightness(0.5) contrast(1.1) saturate(0.6); }
.leaflet-control-layers, .leaflet-bar a { background: var(--raise); color: var(--text);
  border-color: var(--line-2) !important; }
.leaflet-control-layers { border: 1px solid var(--line-2) !important; border-radius: 6px; }
.leaflet-control-layers-expanded { padding: 8px 12px; }
.leaflet-control-layers-separator { border-top-color: var(--line-2); }
.leaflet-bar a:hover { background: #151820; }
.leaflet-control-scale-line { background: rgba(0,0,0,.6); color: var(--muted); border-color: var(--dim); }
.leaflet-tooltip.hs { background: var(--raise); color: var(--text); border: 1px solid var(--line-2);
  box-shadow: none; font-size: 12px; }
.leaflet-tooltip.hs::before { display: none; }
.hs-ring { animation: breathe 3.6s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
@keyframes breathe { 0%, 100% { stroke-opacity: .12; } 50% { stroke-opacity: .55; } }
.motion-off .hs-ring { animation: none; stroke-opacity: .35; }
.readout { position: absolute; left: 14px; bottom: 30px; z-index: 800; min-width: 210px;
  background: rgba(6, 7, 10, .9); border: 1px solid var(--line-2); border-radius: 6px;
  padding: 9px 11px; font-size: 12px; pointer-events: none; }
.readout .t { font-weight: 600; }
.readout .s { color: var(--muted); }
.panel { background: var(--panel); border-left: 1px solid var(--line); overflow-y: auto; min-height: 0;
  scroll-behavior: smooth; overscroll-behavior: contain; }
.panel::-webkit-scrollbar { width: 6px; }
.panel::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 3px; }
.view { padding: 20px 22px 28px; }
.level { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; color: var(--level);
  text-shadow: 0 0 12px color-mix(in srgb, var(--level) 55%, transparent); }
.level::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: var(--level);
  box-shadow: 0 0 10px var(--level); }
.situation { font-size: 18px; line-height: 1.4; font-weight: 500; margin-top: 10px; letter-spacing: -.01em; }
.situation em { font-style: normal; color: var(--amber); }
.asof { color: var(--muted); margin-top: 8px; font-size: 12px; }
.kpis { display: grid; grid-template-columns: repeat(3, 1fr); margin-top: 18px;
  border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.kpi { padding: 10px 12px; background: var(--raise); }
.kpi + .kpi { border-left: 1px solid var(--line); }
.kpi .l { color: var(--muted); font-size: 11px; }
.kpi .v { font-size: 20px; font-weight: 600; margin-top: 1px; }
h2 { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--blue); margin: 26px 0 8px; display: flex; justify-content: space-between; align-items: baseline; }
h2 small { color: var(--dim); font-weight: 400; letter-spacing: 0; text-transform: none; font-size: 11px; }
.list { border-top: 1px solid var(--line); }
.item { display: grid; grid-template-columns: 1fr auto; gap: 2px 12px; width: 100%; text-align: left;
  padding: 9px 4px; border-bottom: 1px solid var(--line); align-items: center; }
.item:hover, .item:focus-visible { background: var(--raise); outline: none; }
.item .n { font-weight: 500; }
.item .r { text-align: right; font-weight: 600; }
.item .d { color: var(--muted); font-size: 12px; }
.item .meter { grid-column: 1 / -1; height: 3px; background: var(--line); border-radius: 2px; overflow: hidden; }
.item .meter i { display: block; height: 100%; background: var(--amber); }
.more { color: var(--muted); font-size: 12px; padding: 8px 4px; }
.more:hover { color: var(--text); }
.legend .bar { display: flex; height: 8px; border-radius: 2px; overflow: hidden; }
.legend .bar i { flex: 1; }
.legend .scale { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; margin-top: 5px; }
.legend .keys { display: flex; justify-content: space-between; font-size: 11px; color: var(--dim); margin-top: 2px; }
.chart { width: 100%; height: auto; display: block; }
.chart .axis { stroke: var(--line-2); }
.chart .thr { stroke: var(--amber); stroke-dasharray: 3 3; }
.chart .thr-l { fill: var(--amber); font-size: 10px; }
.chart .tick { fill: var(--dim); font-size: 10px; }
.toggle { display: flex; align-items: center; gap: 8px; margin-top: 12px; color: var(--muted); font-size: 12px; cursor: pointer; }
.fi { display: grid; grid-template-columns: 150px 1fr 36px; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; }
.fi i { height: 6px; border-radius: 3px; background: var(--line-2); }
.fi b { font-weight: 500; color: var(--muted); text-align: right; }
.meta { margin-top: 14px; color: var(--muted); font-size: 11.5px; line-height: 1.7; }
.meta b { color: var(--text); font-weight: 500; }
.back { color: var(--muted); font-size: 12px; margin-bottom: 14px; display: inline-flex; gap: 6px; }
.back:hover { color: var(--text); }
.kind { color: var(--muted); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.title { font-size: 20px; font-weight: 600; margin-top: 2px; letter-spacing: -.01em; }
.big { display: flex; align-items: baseline; gap: 10px; margin-top: 12px; }
.big .v { font-size: 34px; font-weight: 600; letter-spacing: -.02em; }
.big .x { color: var(--muted); }
.facts { display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin-top: 14px;
  border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.facts div { padding: 8px 12px; background: var(--raise); border-bottom: 1px solid var(--line); }
.facts div:nth-child(odd) { border-right: 1px solid var(--line); }
.facts dt { color: var(--muted); font-size: 11px; }
.facts dd { font-weight: 500; }
.drv-note { color: var(--dim); font-size: 11.5px; margin-bottom: 8px; }
.drv { display: grid; grid-template-columns: minmax(0, 1fr) 96px 40px; gap: 10px; align-items: center; padding: 5px 0; font-size: 12px; }
.drv .lab { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.drv .lab small { color: var(--muted); margin-left: 4px; }
.drv .lab .fb { color: var(--amber); }
.drv .track { position: relative; height: 8px; }
.drv .track::before { content: ''; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--line-2); }
.drv .track i { position: absolute; top: 0; height: 100%; border-radius: 2px; }
.drv .num { text-align: right; color: var(--muted); }
.note { color: var(--dim); font-size: 11.5px; margin-top: 10px; }
@media (max-width: 900px) {
  html, body { overflow: auto; }
  .app { display: block; height: auto; }
  header { height: auto; min-height: 52px; flex-wrap: wrap; padding: 10px 16px; }
  .badges { margin-left: 0; justify-content: flex-start; }
  .body { display: block; }
  .mapwrap { height: 62vh; }
  .panel { border-left: 0; border-top: 1px solid var(--line); overflow: visible; }
  .view { padding: 18px 16px 24px; }
}
@media (prefers-reduced-motion: reduce) { .hs-ring { animation: none; stroke-opacity: .35; } }
"""

# Runs in the page after Leaflet loads. Template fields: __CONFIG__, __COUNTIES__.
_JS = r"""
(function () {
  const C = __CONFIG__;
  const COUNTIES = __COUNTIES__;
  const B = C.bounds;
  const F = C.features.length;
  const $ = s => document.querySelector(s);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ── Map ──
  const map = L.map('map', {
    zoomControl: true, attributionControl: true, maxZoom: C.maxZoom, minZoom: 5,
    zoomSnap: 0.25, zoomDelta: 0.5, wheelPxPerZoomLevel: 110, wheelDebounceTime: 20,
    maxBounds: L.latLngBounds([B.south - 3, B.west - 6], [B.north + 3, B.east + 6]),
  });
  map.attributionControl.setPrefix(false);
  const home = () => { map.invalidateSize(); map.fitBounds([[B.south, B.west], [B.north, B.east]], { padding: [12, 12] }); };
  home();
  window.addEventListener('load', home, { once: true });
  L.control.scale({ imperial: false }).addTo(map);

  const dark = L.tileLayer(
    'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 16, className: 'basemap', attribution: 'Basemap &copy; Esri' }).addTo(map);
  const sat = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 18, className: 'basemap', attribution: 'Imagery &copy; Esri' });

  map.createPane('risk').style.zIndex = 390;
  map.createPane('counties').style.zIndex = 395;
  map.createPane('marks').style.zIndex = 420;
  map.getPane('counties').style.pointerEvents = 'none';

  // One canvas pixel per 2 km cell, stretched over the grid bounds.
  const canvas = document.createElement('canvas');
  canvas.width = C.ncols; canvas.height = C.nrows;
  canvas.style.imageRendering = 'pixelated';
  const ctx = canvas.getContext('2d');
  const gb = L.latLngBounds([B.south, B.west], [B.north, B.east]);
  const GridLayer = L.Layer.extend({
    onAdd(m) { canvas.classList.add('leaflet-zoom-animated'); m.getPane('risk').appendChild(canvas); this._reset(); },
    onRemove() { canvas.remove(); },
    getEvents() { return { zoomanim: this._anim, viewreset: this._reset, zoomend: this._reset }; },
    _reset() {
      const tl = map.latLngToLayerPoint(gb.getNorthWest()), br = map.latLngToLayerPoint(gb.getSouthEast());
      L.DomUtil.setPosition(canvas, tl);
      canvas.style.width = (br.x - tl.x) + 'px'; canvas.style.height = (br.y - tl.y) + 'px';
    },
    _anim(e) {
      L.DomUtil.setTransform(canvas, map._latLngBoundsToNewLayerBounds(gb, e.zoom, e.center).min, map.getZoomScale(e.zoom));
    },
  });
  const grid = new GridLayer().addTo(map);

  const countyStyle = { color: '#ffffff', weight: 0.6, opacity: 0.22, fill: false };
  const countyLayers = {};
  const counties = L.geoJSON(COUNTIES, {
    pane: 'counties', interactive: false, style: countyStyle,
    onEachFeature: (f, layer) => { countyLayers[f.properties.name] = layer; },
  }).addTo(map);

  const marks = L.layerGroup().addTo(map);
  const hsLayer = L.layerGroup().addTo(map);
  L.control.layers({ 'Dark': dark, 'Satellite': sat },
    { 'Ignition risk (2 km)': grid, 'County lines': counties, 'Hotspots': hsLayer },
    { collapsed: true, position: 'topright' }).addTo(map);

  // ── Colour ──
  function rampRGB(p) {
    const x = C.threshold > 0 ? p / C.threshold : 0, R = C.ramp;
    for (let i = 0; i < R.length - 1; i++) {
      const [x0, c0] = R[i], [x1, c1] = R[i + 1];
      if (x <= x1) { const f = Math.max(0, (x - x0) / (x1 - x0)); return c0.map((v, k) => v + (c1[k] - v) * f); }
    }
    return R[R.length - 1][1];
  }
  const hex = p => '#' + rampRGB(p).map(v => Math.round(v).toString(16).padStart(2, '0')).join('');

  // ── Decode ──
  const inflate = b64 => new Response(new Blob([Uint8Array.from(atob(b64), ch => ch.charCodeAt(0))])
    .stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();

  Promise.all([inflate(C.dense), inflate(C.sampled)]).then(([buf, sbuf]) => {
    const n = C.nrows * C.ncols;
    const P = new Uint16Array(buf, 0, n), T = new Int16Array(buf, 2 * n, n);
    const W = new Uint16Array(buf, 4 * n, n), K = new Uint16Array(buf, 6 * n, n);
    const FL = new Uint8Array(buf, 8 * n, n), CTY = new Uint8Array(buf, 9 * n, n);
    const m = new Uint32Array(sbuf, 0, 1)[0];
    const SI = new Uint32Array(sbuf, 4, m);
    const SV = new Uint16Array(sbuf, 4 + 4 * m, m * F);
    const SC = new Int8Array(sbuf, 4 + 4 * m + 2 * m * F, m * F);
    const slot = new Map(); for (let k = 0; k < m; k++) slot.set(SI[k], k);
    const featVal = (k, j) => C.lo[j] + SV[k * F + j] / 65535 * C.span[j];

    // Paint
    const img = ctx.createImageData(C.ncols, C.nrows), base = img.data;
    for (let i = 0; i < n; i++) {
      if (P[i] === 0xFFFF) continue;
      const p = P[i] / 10000, [r, g, b] = rampRGB(p), o = 4 * i;
      base[o] = r; base[o + 1] = g; base[o + 2] = b;
      base[o + 3] = p >= C.threshold ? 255 : 215;
    }
    ctx.putImageData(img, 0, 0);

    // Shimmer: every alert cell glints on two summed sinusoids (a fundamental
    // and a golden-ratio overtone) with its own phase; hotspot rings breathe.
    const hot = [];
    for (let i = 0; i < n; i++) if (P[i] !== 0xFFFF && P[i] / 10000 >= C.threshold) hot.push(i);
    const phase = new Float32Array(hot.length);
    let seed = 12345;
    const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    hot.forEach((i, k) => { phase[k] = 0.18 * (Math.floor(i / C.ncols) + i % C.ncols) + 2 * Math.PI * rand(); });
    const frame = ctx.createImageData(C.ncols, C.nrows);
    const OMEGA = 1.6, A1 = 0.30, A2 = 0.16, PHI = 1.618;
    let running = !reduceMotion && hot.length > 0, last = 0;
    function tick(now) {
      if (!running) return;
      if (now - last > 33) {
        last = now;
        const t = now / 1000, out = frame.data;
        out.set(base);
        for (let k = 0; k < hot.length; k++) {
          const o = 4 * hot[k], s = A1 * Math.sin(OMEGA * t + phase[k]) + A2 * Math.sin(PHI * OMEGA * t + 1.5 * phase[k]);
          if (s > 0) { out[o] += (255 - out[o]) * s; out[o + 1] += (255 - out[o + 1]) * s; out[o + 2] += (255 - out[o + 2]) * s; }
          else { const d = 1 + 0.35 * s; out[o] *= d; out[o + 1] *= d; out[o + 2] *= d; }
        }
        ctx.putImageData(frame, 0, 0);
      }
      requestAnimationFrame(tick);
    }
    const motion = $('#motion');
    motion.checked = running; motion.disabled = hot.length === 0;
    document.body.classList.toggle('motion-off', !running);
    motion.addEventListener('change', () => {
      running = motion.checked;
      document.body.classList.toggle('motion-off', !running);
      if (running) requestAnimationFrame(tick); else ctx.putImageData(img, 0, 0);
    });
    if (running) requestAnimationFrame(tick);

    // Hotspot rings
    C.hotspots.forEach((h, k) => {
      L.circle(h.center, { radius: h.radius_m, pane: 'marks', color: '#ffd08a', weight: 1, fill: false, className: 'hs-ring' })
        .bindTooltip(`Hotspot ${k + 1} · ${esc(h.counties.join(', '))} · ${h.n} cells`, { className: 'hs', direction: 'top' })
        .on('click', () => showHotspot(k)).addTo(hsLayer);
    });

    // ── Cell lookup ──
    const mercY = lat => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
    function cellAt(ll) {
      const col = Math.floor((ll.lng - B.west) / C.dlon), row = Math.floor((C.yTop - mercY(ll.lat)) / C.dy);
      if (col < 0 || row < 0 || col >= C.ncols || row >= C.nrows) return -1;
      const i = row * C.ncols + col;
      return P[i] === 0xFFFF ? -1 : i;
    }
    function cellBounds(i) {
      const row = Math.floor(i / C.ncols), col = i % C.ncols;
      const lat = y => (2 * Math.atan(Math.exp(y)) - Math.PI / 2) * 180 / Math.PI;
      return [[lat(C.yTop - (row + 1) * C.dy), B.west + col * C.dlon], [lat(C.yTop - row * C.dy), B.west + (col + 1) * C.dlon]];
    }
    const countyName = i => CTY[i] === 255 ? 'California coast' : C.countyNames[CTY[i]] + ' County';

    // Hover
    const box = $('#readout');
    map.on('mousemove', e => {
      const i = cellAt(e.latlng);
      if (i < 0) { box.hidden = true; return; }
      const p = P[i] / 10000;
      box.hidden = false;
      box.innerHTML = `<div class="t">${esc(countyName(i))}</div>` +
        `<div><span style="color:${hex(p)};font-weight:600">${p.toFixed(3)}</span>` +
        `<span class="s"> · ${(p / C.threshold).toFixed(1)}× threshold · click for details</span></div>`;
    });
    map.on('mouseout', () => { box.hidden = true; });
    map.on('click', e => { const i = cellAt(e.latlng); if (i >= 0) showCell(i); });

    // ── Inspector ──
    const overview = $('#overview'), inspector = $('#inspector'), panel = $('.panel');
    function openInspector(htmlStr) {
      inspector.innerHTML = `<button class="back" id="back">← Overview</button>` + htmlStr;
      overview.hidden = true; inspector.hidden = false; panel.scrollTo({ top: 0, behavior: 'smooth' });
      $('#back').addEventListener('click', closeInspector);
      inspector.querySelectorAll('[data-cell]').forEach(el => el.addEventListener('click', () => {
        const ll = L.latLng(+el.dataset.lat, +el.dataset.lon), i = cellAt(ll);
        map.flyTo(ll, 10, { duration: 1 }); if (i >= 0) showCell(i);
      }));
    }
    function closeInspector() {
      marks.clearLayers(); resetCountyStyle();
      inspector.hidden = true; overview.hidden = false;
    }
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && !inspector.hidden) closeInspector(); });
    function resetCountyStyle() { Object.values(countyLayers).forEach(l => l.setStyle(countyStyle)); }

    function fmtVal(j, v) {
      const [label, unit, dp] = C.features[j];
      return `${v.toFixed(dp)}${unit ? ' ' + unit : ''}`;
    }
    function driversHTML(contrib, values) {
      if (!C.hasDrivers) return '';
      const order = [...contrib.keys()].sort((a, b) => Math.abs(contrib[b]) - Math.abs(contrib[a])).slice(0, 7);
      const maxAbs = Math.max(0.5, ...order.map(j => Math.abs(contrib[j])));
      const rows = order.map(j => {
        const c = contrib[j], w = 50 * Math.abs(c) / maxAbs, up = c > 0;
        const fb = C.staleFeatures.includes(C.featureKeys[j]);
        return `<div class="drv"><div class="lab">${esc(C.features[j][0])}` +
          (values ? `<small>${fmtVal(j, values[j])}</small>` : '') +
          (fb ? `<small class="fb">fallback</small>` : '') + `</div>` +
          `<div class="track"><i style="${up ? 'left:50%' : 'right:50%'};width:${w}%;background:${up ? '#f2b134' : '#3a5670'}"></i></div>` +
          `<div class="num">${up ? '+' : '−'}${Math.abs(c).toFixed(2)}</div></div>`;
      }).join('');
      return `<h2>What drives this score</h2><div class="drv-note">How much each input pushed the model's log-odds; right raises risk, left lowers it.</div>${rows}`;
    }
    function meanDrivers(cells) {
      const sum = new Float64Array(F); let cnt = 0;
      cells.forEach(i => { const k = slot.get(i); if (k === undefined) return; cnt++; for (let j = 0; j < F; j++) sum[j] += SC[k * F + j] / C.contribScale; });
      return cnt ? { c: Array.from(sum, v => v / cnt), n: cnt } : null;
    }

    function showCell(i) {
      marks.clearLayers(); resetCountyStyle();
      L.rectangle(cellBounds(i), { pane: 'marks', color: '#ffffff', weight: 2, fill: false }).addTo(marks);
      const p = P[i] / 10000, k = slot.get(i), alert = p >= C.threshold;
      const [[s, w], [nn, e]] = cellBounds(i);
      let drv = '', facts;
      if (k !== undefined) {
        const values = Array.from({ length: F }, (_, j) => featVal(k, j));
        const contrib = Array.from({ length: F }, (_, j) => SC[k * F + j] / C.contribScale);
        const pick = ['temp_c', 'rh_pct', 'wind_speed_ms', 'vpd_kpa', 'ndvi', 'elevation_m'].map(f => C.featureKeys.indexOf(f)).filter(j => j >= 0);
        facts = pick.map(j => `<div><dt>${esc(C.features[j][0])}</dt><dd>${fmtVal(j, values[j])}</dd></div>`).join('');
        drv = driversHTML(contrib, values);
      } else {
        facts = `<div><dt>Temperature</dt><dd>${(T[i] / 10).toFixed(1)} °C</dd></div><div><dt>Wind speed</dt><dd>${(W[i] / 10).toFixed(1)} m/s</dd></div>`;
        drv = `<div class="note">This cell was not sampled in this window; its values are the mean of nearby sampled cells, so it has no model explanation of its own. Click a neighbouring cell for one.</div>`;
      }
      openInspector(
        `<div class="kind">2 km cell · ${FL[i] === 1 ? 'sampled' : 'filled from neighbours'}</div>` +
        `<div class="title">${esc(countyName(i))}</div>` +
        `<div class="big"><span class="v" style="color:${hex(p)}">${p.toFixed(3)}</span>` +
        `<span class="x">${(p / C.threshold).toFixed(2)}× threshold${alert ? ' · <b style="color:var(--amber)">alert</b>' : ''}</span></div>` +
        `<dl class="facts">${facts}</dl>` + drv +
        `<div class="note">${((s + nn) / 2).toFixed(4)}, ${((w + e) / 2).toFixed(4)} · 7-day ignition probability</div>`);
    }

    function groupHTML(cells, kind, title, stats, topRows) {
      const md = meanDrivers(cells);
      const facts = stats.map(([l, v]) => `<div><dt>${l}</dt><dd>${v}</dd></div>`).join('');
      const tops = topRows.length ? `<h2>Highest cells</h2><div class="list">` + topRows.map(([lat, lon, p]) =>
        `<button class="item" data-cell data-lat="${lat}" data-lon="${lon}"><span class="n">${lat.toFixed(3)}, ${lon.toFixed(3)}</span>` +
        `<span class="r" style="color:${hex(p)}">${p.toFixed(3)}</span></button>`).join('') + `</div>` : '';
      const drv = md ? driversHTML(md.c, null).replace('What drives this score', `What drives risk here <small>mean over ${md.n.toLocaleString()} sampled cells</small>`) : '';
      return `<div class="kind">${kind}</div><div class="title">${esc(title)}</div><dl class="facts">${facts}</dl>${tops}${drv}`;
    }

    function showCounty(ci) {
      const c = C.counties.find(r => r.i === ci); if (!c) return;
      marks.clearLayers(); resetCountyStyle();
      const layer = countyLayers[c.name];
      if (layer) { layer.setStyle({ color: '#ffffff', weight: 2, opacity: 0.95 }); map.flyToBounds(layer.getBounds(), { padding: [30, 30], duration: 1 }); }
      const cells = []; for (let i = 0; i < n; i++) if (CTY[i] === ci && P[i] !== 0xFFFF && P[i] / 10000 >= (c.alerts ? C.threshold : 0)) cells.push(i);
      openInspector(groupHTML(cells, 'County', `${c.name} County`, [
        ['Alert cells', `${c.alerts.toLocaleString()} of ${c.cells.toLocaleString()}`],
        ['Share on alert', `${(100 * c.share).toFixed(1)}%`],
        ['Peak probability', `<span style="color:${hex(c.peak)}">${c.peak.toFixed(3)}</span>`],
        ['Mean probability', c.mean.toFixed(4)],
      ], c.top));
    }

    function showHotspot(k) {
      const h = C.hotspots[k];
      marks.clearLayers(); resetCountyStyle();
      map.flyToBounds(h.bounds, { padding: [60, 60], duration: 1, maxZoom: 10 });
      L.rectangle(h.bounds, { pane: 'marks', color: '#ffd08a', weight: 1, dashArray: '4 4', fill: false }).addTo(marks);
      openInspector(groupHTML(h.cells, `Hotspot ${k + 1}`, h.counties.join(', '), [
        ['Alert cells', h.n.toLocaleString()],
        ['Area', `~${(h.n * C.cellKm2).toLocaleString()} km²`],
        ['Peak probability', `<span style="color:${hex(h.peak)}">${h.peak.toFixed(3)}</span>`],
        ['Counties', h.counties.length],
      ], []));
    }

    document.querySelectorAll('[data-county]').forEach(el => el.addEventListener('click', () => showCounty(+el.dataset.county)));
    document.querySelectorAll('[data-hotspot]').forEach(el => el.addEventListener('click', () => showHotspot(+el.dataset.hotspot)));
    const more = $('#more-counties');
    if (more) more.addEventListener('click', () => {
      document.querySelectorAll('.item.extra').forEach(el => { el.hidden = !el.hidden; });
      more.textContent = more.textContent.startsWith('Show') ? 'Show fewer' : `Show all ${C.counties.length} counties`;
    });
  });
})();
"""


def _page(grid: dict, scored: pd.DataFrame, meta: dict, target_date: str,
          threshold: float, summary: dict) -> str:
    probs = scored["fire_probability"]
    n_total = len(scored)
    alerts = int((probs >= threshold).sum())
    share = alerts / n_total if n_total else 0.0
    peak_p = float(probs.max()) if n_total else 0.0
    level, level_colour = risk_level(share)

    names = region.county_names()
    sample_county = region.locate(scored["longitude"], scored["latitude"])
    counties = county_table(scored, sample_county, threshold)
    hs = hotspots(grid, threshold)
    contrib = contributions(scored)
    sampled_b64, sinfo = _pack_sampled(grid, scored, contrib)

    # Inputs that are a constant fallback in this window, not data.
    stale = []
    if "kbdi_index" in scored and len(scored) and scored["kbdi_index"].round().eq(300).all():
        stale.append("kbdi_index")

    metrics = meta.get("metrics") or {}
    test_metrics = metrics.get("test") or metrics
    auc = test_metrics.get("roc_auc", float("nan"))
    ap = test_metrics.get("average_precision", float("nan"))
    data_info = meta.get("data") or {}
    years = lambda k: "–".join(str(y) for y in data_info.get(k) or []) or "—"
    prod_years = data_info.get("production_years") or []
    prod_label = f"{prod_years[0]}–{prod_years[-1]}" if prod_years else years("train_years")
    trained_region = data_info.get("region")

    # Situation sentence
    peak_row = scored.loc[probs.idxmax()] if n_total else None
    peak_cty = region.locate([peak_row["longitude"]], [peak_row["latitude"]])[0] if n_total else -1
    peak_where = f" in {html.escape(names[peak_cty])} County" if peak_cty >= 0 else ""
    hot_counties = [c["name"] for c in counties if c["alerts"] > 0][:3]
    if alerts:
        situation = (f"<em>{alerts:,} cells</em> are above the alert threshold, "
                     f"{100 * share:.1f}% of the state, concentrated in {_join(hot_counties)}. "
                     f"Highest probability {peak_p:.3f}{peak_where}.")
    else:
        situation = (f"No cells are above the alert threshold. "
                     f"Highest probability {peak_p:.3f}{peak_where}.")

    window = summary.get("data_window") or {}
    badges = []
    if window.get("end"):
        lag = (date.fromisoformat(target_date) - date.fromisoformat(window["end"])).days
        cls = "warn" if lag > 2 else ""
        lag_txt = f" · {lag} days behind" if lag > 0 else ""
        badges.append((cls, f"Weather {_fmt_day(window.get('start'))} – {_fmt_day(window['end'])}{lag_txt}"))
    else:
        badges.append(("info", "Weather window not recorded"))
    if stale:
        badges.append(("warn", "KBDI missing · fallback used"))
    if summary.get("rescored"):
        badges.append(("info", "Re-scored with current model"))
    badges.append(("info", f"Model fit: {html.escape(trained_region)}" if trained_region
                   else "Model fit: CA · OR · WA"))
    badge_html = "".join(f'<span class="badge {c}">{t}</span>' for c, t in badges)

    # Overview lists
    hs_html = "".join(
        f'<button class="item" data-hotspot="{k}"><span class="n">{k + 1}. {html.escape(_join(h["counties"]))}</span>'
        f'<span class="r" style="color:{ramp_color(h["peak"], threshold)}">{h["peak"]:.3f}</span>'
        f'<span class="d">{h["n"]:,} alert cells · ~{h["n"] * (CFG.cell_scale_m / 1000) ** 2:,.0f} km²</span><span class="d"></span></button>'
        for k, h in enumerate(hs)
    ) or '<p class="muted" style="padding:8px 4px">No alert clusters today.</p>'
    max_share = max((c["share"] for c in counties), default=0) or 1
    county_html = "".join(
        f'<button class="item{" extra" if k >= 10 else ""}" data-county="{c["i"]}"{" hidden" if k >= 10 else ""}>'
        f'<span class="n">{html.escape(c["name"])}</span>'
        f'<span class="r" style="color:{ramp_color(c["peak"], threshold)}">{c["peak"]:.3f}</span>'
        f'<span class="d">{c["alerts"]:,} alert cells · {100 * c["share"]:.1f}% of sampled</span><span class="d">peak</span>'
        f'<span class="meter"><i style="width:{100 * c["share"] / max_share:.1f}%"></i></span></button>'
        for k, c in enumerate(counties)
    )
    legend_cells = "".join(f'<i style="background:{ramp_color(threshold * x / 20, threshold)}"></i>'
                           for x in range(0, 61))

    config = {
        "bounds": grid["bounds"], "nrows": grid["nrows"], "ncols": grid["ncols"],
        "dlon": grid["dlon"], "dy": grid["dy"], "yTop": grid["y_top"],
        "threshold": threshold, "maxZoom": MAX_ZOOM,
        "ramp": [[x, list(_hex_rgb(c))] for x, c in RAMP],
        "countyNames": names,
        "counties": [{k: v for k, v in c.items()} for c in counties],
        "hotspots": hs,
        "featureKeys": list(CFG.feature_columns),
        "features": [list(FEATURE_LABELS.get(f, (f, "", 2))) for f in CFG.feature_columns],
        "staleFeatures": stale,
        "contribScale": CONTRIB_SCALE,
        "cellKm2": (CFG.cell_scale_m / 1000) ** 2,
        "dense": _pack_dense(grid), "sampled": sampled_b64, **sinfo,
    }
    js = (_JS.replace("__CONFIG__", json.dumps(config, separators=(",", ":")))
             .replace("__COUNTIES__", json.dumps(region.counties(), separators=(",", ":"))))
    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stale_note = ("<br><b style='color:var(--amber)'>KBDI</b> the KBDI source returned no images for this "
                  "window, so every cell was scored with the 300 fallback. The model reads that as moist "
                  "ground: it lowers every cell's score.") if stale else ""
    model_region = html.escape(trained_region) if trained_region else "California, Oregon and Washington"

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>California Wildfire Risk — {target_date}</title>
<meta name="description" content="{level} wildfire ignition risk in California, {target_date}: {alerts:,} alert cells.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>{_CSS}</style>
</head>
<body style="--level:{level_colour}">
<div class="app">
  <header>
    <div class="brand">California Wildfire Risk<span>{datetime.strptime(target_date, "%Y-%m-%d").strftime("%a %b %-d, %Y")}</span></div>
    <div class="badges">{badge_html}</div>
  </header>
  <div class="body">
    <div class="mapwrap">
      <div id="map" aria-label="California ignition risk map"></div>
      <div id="readout" class="readout" hidden></div>
    </div>
    <aside class="panel">
      <section id="overview" class="view">
        <div class="level">{level} risk</div>
        <p class="situation">{situation}</p>
        <div class="asof">7-day ignition probability · scored {n_total:,} cells · generated {gen_ts}</div>
        <div class="kpis">
          <div class="kpi"><div class="l">Alert cells</div><div class="v" style="color:{'var(--amber)' if alerts else 'var(--text)'}">{alerts:,}</div></div>
          <div class="kpi"><div class="l">Peak</div><div class="v" style="color:{ramp_color(peak_p, threshold)}">{peak_p:.3f}</div></div>
          <div class="kpi"><div class="l">Threshold</div><div class="v">{threshold:.3f}</div></div>
        </div>

        <h2>Hotspots <small>largest alert clusters</small></h2>
        <div class="list">{hs_html}</div>

        <h2>Counties <small>by alert cells</small></h2>
        <div class="list">{county_html}</div>
        {f'<button class="more" id="more-counties">Show all {len(counties)} counties</button>' if len(counties) > 10 else ''}

        <h2>Map scale</h2>
        <div class="legend">
          <div class="bar">{legend_cells}</div>
          <div class="scale"><span>0</span><span style="color:var(--amber)">{threshold:.3f}</span><span>{2 * threshold:.2f}</span><span>≥ {3 * threshold:.2f}</span></div>
          <div class="keys"><span>below threshold</span><span>alert</span></div>
        </div>
        <label class="toggle"><input type="checkbox" id="motion"> Shimmer alert cells</label>

        <h2>Distribution <small>cells per bin, √ scale</small></h2>
        {_histogram_svg(probs, threshold)}

        <h2>Model</h2>
        {_feature_importance_rows(meta)}
        <div class="meta">
          <b>XGBoost</b> ignition classifier, production fit {prod_label} on {model_region} samples.
          Train {years("train_years")}, threshold tuned on {years("val_years")}, tested out of time on {years("test_years")}:
          ROC-AUC {auc:.3f}, AUC-PR {ap:.4f}.<br>
          <b>Inputs</b> ERA5-Land, SRTM, NASA SMAP, KBDI, MODIS NDVI · labels NASA FIRMS.{stale_note}<br>
          <b>Risk level</b> share of cells at or above the threshold: Low &lt; 0.1%, Moderate &lt; 1%,
          Elevated &lt; 5%, Severe ≥ 5%. These are display bands, not model outputs.<br>
          <b>Grid</b> {CFG.cell_scale_m / 1000:g} km cells · {int((grid["flag"] == 1).sum()):,} sampled ·
          {int((grid["flag"] == 2).sum()):,} filled from neighbours · county lines from Code for Germany click_that_hood.
        </div>
      </section>
      <section id="inspector" class="view" hidden></section>
    </aside>
  </div>
</div>
<script>{js}</script>
</body></html>
"""


def render(scored: pd.DataFrame, meta: dict, target_date: str, summary: dict,
           out_html: Path) -> None:
    from .predict import resolve_threshold

    threshold = float(summary.get("operational_threshold") or resolve_threshold(meta))
    scored = region.restrict(scored).reset_index(drop=True)
    grid = rasterize(scored)
    page = _page(grid, scored, meta, target_date, threshold, summary)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    print(f"[heatmap] wrote {out_html}  ({out_html.stat().st_size / 1024:.0f} KB, "
          f"{len(scored):,} California cells)")


def main():
    parser = argparse.ArgumentParser(description="Render the California wildfire risk map.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Scored parquet (default: derived from --date).")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date, used to locate the dashboard output dir.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (default: alongside input).")
    args = parser.parse_args()

    if args.input:
        in_path = args.input
        target_date = in_path.stem.replace("ignition_", "")
        out_dir = in_path.parent
    elif args.date:
        target_date = args.date
        out_dir = Path(__file__).resolve().parent.parent / "wildfire_output" / target_date
        in_path = out_dir / f"ignition_{target_date}.parquet"
    else:
        raise SystemExit("Pass --input <parquet> or --date YYYY-MM-DD.")

    if not in_path.exists():
        raise SystemExit(f"No scored parquet at {in_path}. Run wildfire.dashboard first.")

    summary_path = out_dir / f"summary_{target_date}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    meta_path = CFG.feature_meta_path()
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    render(pd.read_parquet(in_path), meta, target_date, summary,
           args.out or (out_dir / f"heatmap_{target_date}.html"))


if __name__ == "__main__":
    main()
