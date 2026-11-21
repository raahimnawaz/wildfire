"""
California wildfire risk map.

Reads a scored parquet (output of ``wildfire.dashboard``) and writes one
self-contained HTML page:

  * Map: the ignition probabilities rasterized onto a 2 km grid clipped to the
    California boundary and drawn as crisp cells on a dark basemap, so the
    risk colours the state itself rather than floating over it. County lines
    sit on top for orientation.
  * Alert cells (probability at or above the operational threshold) shimmer:
    each brightens toward white on two summed sinusoids (a fundamental and a
    golden-ratio overtone) with its own phase. Only alerts move, so the motion
    marks the alert set; the colour of every cell is the colour of its value.
  * Hover readout: county, probability, temperature, wind and KBDI under the
    cursor, and whether the cell was sampled or filled from neighbours.
  * Side panel: run statistics, risk distribution, top-10 cells (click to fly
    there) and feature importances.

GEE samples ~37% of the 2 km cells per window. Unsampled land cells take the
mean of sampled cells in the surrounding 3x3 block (5x5 if that is empty); the
readout labels them as filled. Cells with no sample within 2 cells stay blank.

The grid rows are spaced evenly in Web Mercator, so each cell is square on the
map and lines up exactly with the basemap at every zoom.

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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import region
from .config import CFG

# Colour ramp, anchored to the operational threshold t: blues below the
# threshold, amber at it, red by 3t. Stops are (multiple of t, colour).
RAMP = [
    (0.0,   "#0b1e36"),
    (1 / 3, "#12507a"),
    (2 / 3, "#1fa8cc"),
    (1.0,   "#ffc43d"),
    (2.0,   "#ff7a1a"),
    (3.0,   "#ff2d4a"),
]

MAX_ZOOM = 11  # 2 km cells are ~15 px here; closer adds pixels, not information
EMPTY = 0xFFFF


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
    2 filled) and county index (-1 outside the state).
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
    idx = row[ok] * ncols + col[ok]
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
        for key, arr in fields.items():
            mean, wsum = _box_mean(np.nan_to_num(arr), weight, r)
            fill = todo & (wsum > 0)
            arr[fill] = mean[fill]
        flag[todo & ~np.isnan(fields["p"])] = 2

    return {
        "bounds": {"west": w, "south": south, "east": east, "north": n},
        "nrows": nrows, "ncols": ncols, "dlon": dlon, "dy": dy, "y_top": y_top,
        "flag": flag, "county": county, **fields,
    }


def _pack(grid: dict) -> str:
    """Struct-of-arrays, zlib-compressed, base64: p, temp, wind, kbdi as
    little-endian 16-bit ints, then flag and county as bytes."""
    p = grid["p"]
    blank = np.isnan(p)
    parts = [
        np.where(blank, EMPTY, np.round(np.nan_to_num(p) * 10000)).astype("<u2"),
        np.round(np.nan_to_num(grid["temp"]) * 10).astype("<i2"),
        np.round(np.clip(np.nan_to_num(grid["wind"]), 0, 6000) * 10).astype("<u2"),
        np.round(np.clip(np.nan_to_num(grid["kbdi"]), 0, 65000)).astype("<u2"),
        grid["flag"].astype(np.uint8),
        np.where(grid["county"] < 0, 255, grid["county"]).astype(np.uint8),
    ]
    raw = b"".join(a.ravel().tobytes() for a in parts)
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
# Panel pieces
# ─────────────────────────────────────────────────────────────────────────────

def _histogram_svg(probs: pd.Series, threshold: float, width: int = 320, height: int = 96) -> str:
    """Square-root-scaled histogram, coloured with the map ramp, threshold marked."""
    x_max = max(0.1, math.ceil(float(probs.max()) * 10) / 10)
    n_bins = 30
    counts, edges = np.histogram(probs.clip(upper=x_max), bins=n_bins, range=(0, x_max))
    top = math.sqrt(counts.max() or 1)
    pad, base = 8, height - 16
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
        f'<line x1="{tx:.1f}" y1="4" x2="{tx:.1f}" y2="{base}" class="thr"/>'
        f'<text x="{tx + 4:.1f}" y="12" class="thr-l">alert {threshold:.3f}</text>'
        f'<text x="{pad}" y="{height - 2}" class="tick">0</text>'
        f'<text x="{width - pad}" y="{height - 2}" class="tick" text-anchor="end">{x_max:.1f}</text>'
        f'</svg>'
    )


def _feature_importance_svg(meta: dict, top_n: int = 8) -> str:
    fi = meta.get("feature_importance") or {}
    if not fi:
        return '<p class="muted">No feature importances in model metadata.</p>'
    items = list(fi.items())[:top_n]
    max_v = max(v for _, v in items) or 1.0
    width, row_h, label_w = 320, 24, 122
    rows = []
    for i, (name, v) in enumerate(items):
        bw = (width - label_w - 44) * (v / max_v)
        y = 6 + i * row_h
        rows.append(
            f'<text x="0" y="{y + 11}" class="fi-l">{html.escape(name)}</text>'
            f'<rect x="{label_w}" y="{y + 2}" width="{bw:.1f}" height="12" rx="2" fill="url(#fiGrad)"/>'
            f'<text x="{label_w + bw + 6:.1f}" y="{y + 12}" class="tick">{v:.3f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {row_h * len(items) + 6}" class="chart" role="img" '
        f'aria-label="Feature importances">'
        f'<defs><linearGradient id="fiGrad" x1="0" x2="1">'
        f'<stop offset="0" stop-color="#1fa8cc"/><stop offset="1" stop-color="#ff7a1a"/>'
        f'</linearGradient></defs>{"".join(rows)}</svg>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg: #000; --panel: #050608; --line: #16191f; --line-2: #22262e;
  --text: #f4f6f8; --muted: #8a93a0; --dim: #5b6370;
  --blue: #7fdcff; --red: #ff4d5e; --amber: #ffc43d;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; background: var(--bg); color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-size: 13px; line-height: 1.45; font-feature-settings: 'tnum' 1, 'cv11' 1;
  -webkit-font-smoothing: antialiased; }
.app { display: grid; grid-template-columns: minmax(0, 1fr) 380px; height: 100vh; }
.mapwrap { position: relative; height: 100%; min-height: 0; }
#map { position: absolute; inset: 0; background: var(--bg); }
.leaflet-container { background: var(--bg); font-family: inherit; }
.basemap { filter: brightness(0.55) contrast(1.1); }
.leaflet-control-layers, .leaflet-bar a { background: #0b0d11; color: var(--text);
  border-color: var(--line-2) !important; }
.leaflet-control-layers { border: 1px solid var(--line-2) !important; border-radius: 6px; }
.leaflet-control-layers-expanded { padding: 8px 12px; }
.leaflet-control-layers-separator { border-top-color: var(--line-2); }
.leaflet-bar a:hover { background: #151820; }
.leaflet-control-scale-line { background: rgba(0,0,0,.6); color: var(--muted);
  border-color: var(--dim); }
.readout { position: absolute; left: 16px; bottom: 16px; z-index: 800; min-width: 220px;
  background: rgba(5, 6, 8, .88); border: 1px solid var(--line-2); border-radius: 6px;
  padding: 10px 12px; font-size: 12px; pointer-events: none; backdrop-filter: blur(6px); }
.readout .county { font-weight: 600; margin-bottom: 4px; }
.readout dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 14px; }
.readout dt { color: var(--muted); }
.readout dd { text-align: right; }
.readout .note { color: var(--dim); font-size: 11px; margin-top: 4px; }
.panel { background: var(--panel); border-left: 1px solid var(--line); overflow-y: auto;
  padding: 22px 22px 28px; }
.panel::-webkit-scrollbar { width: 6px; }
.panel::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 3px; }
.eyebrow { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--red); text-shadow: 0 0 10px rgba(255, 77, 94, .45); }
h1 { font-size: 21px; font-weight: 600; letter-spacing: -.01em; margin-top: 4px; color: var(--red);
  text-shadow: 0 0 14px rgba(255, 77, 94, .40); }
.sub { color: var(--muted); margin-top: 2px; }
h2 { font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: var(--blue); text-shadow: 0 0 10px rgba(127, 220, 255, .35);
  margin: 26px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
.kpis { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.kpi { border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; background: #08090c; }
.kpi .l { color: var(--muted); font-size: 11px; }
.kpi .v { font-size: 20px; font-weight: 600; margin-top: 2px; }
.kpi.alert .v { color: var(--amber); }
.kpi.peak .v { color: var(--red); }
.legend .bar { height: 10px; border-radius: 2px; }
.legend .scale { display: flex; justify-content: space-between; color: var(--muted);
  font-size: 11px; margin-top: 4px; }
.chart { width: 100%; height: auto; display: block; }
.chart .axis { stroke: var(--line-2); }
.chart .thr { stroke: var(--amber); stroke-dasharray: 3 3; }
.chart .thr-l { fill: var(--amber); font-size: 10px; }
.chart .tick { fill: var(--dim); font-size: 10px; }
.chart .fi-l { fill: var(--text); font-size: 11px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { color: var(--muted); font-weight: 500; text-align: right; padding: 5px 4px;
  border-bottom: 1px solid var(--line-2); }
td { text-align: right; padding: 6px 4px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
td.p { font-weight: 600; }
tbody tr { cursor: pointer; }
tbody tr:hover td { background: #0d1016; }
tbody tr:focus-visible { outline: 1px solid var(--blue); }
.muted { color: var(--muted); }
.hint { color: var(--dim); font-size: 11px; margin-top: 6px; }
.toggle { display: flex; align-items: center; gap: 8px; margin-top: 12px; color: var(--muted);
  font-size: 12px; cursor: pointer; }
.meta { margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 11.5px; line-height: 1.7; }
.meta b { color: var(--text); font-weight: 500; }
@media (max-width: 900px) {
  html, body { overflow: auto; }
  .app { grid-template-columns: 1fr; height: auto; display: block; }
  .mapwrap { height: 70vh; }
  .panel { border-left: 0; border-top: 1px solid var(--line); padding: 18px 16px 24px; }
}
"""

# Runs in the page after Leaflet loads. Template fields: __CONFIG__, __COUNTIES__.
_JS = r"""
(function () {
  const C = __CONFIG__;
  const COUNTIES = __COUNTIES__;
  const B = C.bounds;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const map = L.map('map', {
    zoomControl: true, attributionControl: true, maxZoom: C.maxZoom, minZoom: 5,
    maxBounds: L.latLngBounds([B.south - 3, B.west - 6], [B.north + 3, B.east + 6]),
  });
  map.attributionControl.setPrefix(false);
  const frame = () => { map.invalidateSize(); map.fitBounds([[B.south, B.west], [B.north, B.east]], { padding: [12, 12] }); };
  frame();
  window.addEventListener('load', frame, { once: true });
  L.control.scale({ imperial: false }).addTo(map);

  const dark = L.tileLayer(
    'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 16, className: 'basemap', attribution: 'Basemap &copy; Esri' }).addTo(map);
  const sat = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 18, className: 'basemap', attribution: 'Imagery &copy; Esri' });

  // ── Risk grid: one canvas pixel per 2 km cell, stretched over its bounds ──
  const canvas = document.createElement('canvas');
  canvas.width = C.ncols; canvas.height = C.nrows;
  canvas.style.imageRendering = 'pixelated';
  const ctx = canvas.getContext('2d');
  const bounds = L.latLngBounds([B.south, B.west], [B.north, B.east]);

  const GridLayer = L.Layer.extend({
    onAdd(m) {
      canvas.classList.add('leaflet-zoom-animated');
      m.getPane('risk').appendChild(canvas);
      this._reset();
    },
    onRemove() { canvas.remove(); },
    getEvents() { return { zoomanim: this._animateZoom, viewreset: this._reset, zoomend: this._reset }; },
    _reset() {
      const tl = map.latLngToLayerPoint(bounds.getNorthWest());
      const br = map.latLngToLayerPoint(bounds.getSouthEast());
      L.DomUtil.setPosition(canvas, tl);
      canvas.style.width = (br.x - tl.x) + 'px';
      canvas.style.height = (br.y - tl.y) + 'px';
    },
    _animateZoom(e) {
      const scale = map.getZoomScale(e.zoom);
      const offset = map._latLngBoundsToNewLayerBounds(bounds, e.zoom, e.center).min;
      L.DomUtil.setTransform(canvas, offset, scale);
    },
  });
  map.createPane('risk').style.zIndex = 390;
  map.createPane('counties').style.zIndex = 395;
  map.getPane('counties').style.pointerEvents = 'none';
  const grid = new GridLayer();
  grid.addTo(map);

  const counties = L.geoJSON(COUNTIES, {
    pane: 'counties', interactive: false,
    style: { color: '#ffffff', weight: 0.7, opacity: 0.28, fill: false },
  }).addTo(map);

  L.control.layers(
    { 'Dark': dark, 'Satellite': sat },
    { 'Ignition risk (2 km)': grid, 'County lines': counties },
    { collapsed: true, position: 'topright' }).addTo(map);

  // ── Colour ramp, the same stops the Python side uses ──
  function rampRGB(p) {
    const x = C.threshold > 0 ? p / C.threshold : 0;
    const R = C.ramp;
    for (let i = 0; i < R.length - 1; i++) {
      const [x0, c0] = R[i], [x1, c1] = R[i + 1];
      if (x <= x1) {
        const f = Math.max(0, (x - x0) / (x1 - x0));
        return c0.map((v, k) => v + (c1[k] - v) * f);
      }
    }
    return R[R.length - 1][1];
  }

  // ── Decode the packed grid, paint, then animate the alert cells ──
  const raw = Uint8Array.from(atob(C.data), ch => ch.charCodeAt(0));
  new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate')))
    .arrayBuffer().then(buf => {
      const n = C.nrows * C.ncols;
      const P = new Uint16Array(buf, 0, n), T = new Int16Array(buf, 2 * n, n);
      const W = new Uint16Array(buf, 4 * n, n), K = new Uint16Array(buf, 6 * n, n);
      const F = new Uint8Array(buf, 8 * n, n), CTY = new Uint8Array(buf, 9 * n, n);

      const img = ctx.createImageData(C.ncols, C.nrows);
      const base = img.data;
      const hot = [];
      for (let i = 0; i < n; i++) {
        if (P[i] === 0xFFFF) continue;
        const p = P[i] / 10000;
        const [r, g, b] = rampRGB(p);
        const o = 4 * i;
        base[o] = r; base[o + 1] = g; base[o + 2] = b;
        base[o + 3] = Math.round(255 * Math.min(0.92, 0.55 + 0.37 * Math.min(1, p / C.threshold)));
        if (p >= C.threshold) hot.push(i);
      }
      ctx.putImageData(img, 0, 0);

      // Harmonic shimmer: lightness(t) = A1 sin(wt + phi) + A2 sin(1.618 wt + 1.5 phi),
      // phi = slow spatial wave + per-cell random phase, so neighbouring alerts
      // glint at different moments.
      const OMEGA = 1.6, A1 = 0.30, A2 = 0.16, PHI = 1.618;
      const phase = new Float32Array(hot.length);
      let seed = 12345;
      const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
      hot.forEach((i, k) => {
        const row = Math.floor(i / C.ncols), col = i % C.ncols;
        phase[k] = 0.18 * (row + col) + 2 * Math.PI * rand();
      });
      const frame = ctx.createImageData(C.ncols, C.nrows);
      let running = !reduceMotion && hot.length > 0, last = 0;
      function tick(now) {
        if (!running) return;
        if (now - last > 33) {
          last = now;
          const t = now / 1000, out = frame.data;
          out.set(base);
          for (let k = 0; k < hot.length; k++) {
            const o = 4 * hot[k], ph = phase[k];
            const s = A1 * Math.sin(OMEGA * t + ph) + A2 * Math.sin(PHI * OMEGA * t + 1.5 * ph);
            if (s > 0) {
              out[o] += (255 - out[o]) * s; out[o + 1] += (255 - out[o + 1]) * s;
              out[o + 2] += (255 - out[o + 2]) * s; out[o + 3] = 255;
            } else {
              const d = 1 + 0.35 * s;
              out[o] *= d; out[o + 1] *= d; out[o + 2] *= d;
            }
          }
          ctx.putImageData(frame, 0, 0);
        }
        requestAnimationFrame(tick);
      }
      const toggle = document.getElementById('shimmer');
      toggle.checked = running;
      toggle.disabled = hot.length === 0;
      toggle.addEventListener('change', () => {
        running = toggle.checked;
        if (running) requestAnimationFrame(tick); else ctx.putImageData(img, 0, 0);
      });
      if (running) requestAnimationFrame(tick);

      // ── Hover readout ──
      const box = document.getElementById('readout');
      function mercY(lat) { return Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)); }
      map.on('mousemove', e => {
        const col = Math.floor((e.latlng.lng - B.west) / C.dlon);
        const row = Math.floor((C.yTop - mercY(e.latlng.lat)) / C.dy);
        const i = row * C.ncols + col;
        if (col < 0 || row < 0 || col >= C.ncols || row >= C.nrows || (CTY[i] === 255 && P[i] === 0xFFFF)) {
          box.hidden = true; return;
        }
        const has = P[i] !== 0xFFFF;
        const p = P[i] / 10000;
        box.hidden = false;
        box.innerHTML =
          `<div class="county">${CTY[i] === 255 ? 'California coast' : C.counties[CTY[i]] + ' County'}</div>` +
          (has ? `<dl><dt>Ignition p</dt><dd style="color:${p >= C.threshold ? '#ffc43d' : '#f4f6f8'}">${p.toFixed(3)}${p >= C.threshold ? ' · alert' : ''}</dd>` +
                 `<dt>Temp</dt><dd>${(T[i] / 10).toFixed(1)} °C</dd>` +
                 `<dt>Wind</dt><dd>${(W[i] / 10).toFixed(1)} m/s</dd>` +
                 `<dt>KBDI</dt><dd>${C.kbdiStale ? 'no data' : K[i]}</dd></dl>` +
                 `<div class="note">${F[i] === 1 ? 'sampled cell' : 'filled from neighbouring samples'} · ` +
                 `${e.latlng.lat.toFixed(3)}, ${e.latlng.lng.toFixed(3)}</div>`
              : `<div class="note">no sample within 4 km</div>`);
      });
      map.on('mouseout', () => { box.hidden = true; });
    });

  // ── Top-10 rows fly the map to the cell ──
  document.querySelectorAll('#top10 tbody tr').forEach(tr => {
    const go = () => {
      const lat = +tr.dataset.lat, lon = +tr.dataset.lon;
      map.flyTo([lat, lon], 9, { duration: 1.2 });
      const ring = L.circleMarker([lat, lon], { radius: 22, color: '#7fdcff', weight: 2, fill: false })
        .addTo(map);
      setTimeout(() => map.removeLayer(ring), 2400);
    };
    tr.addEventListener('click', go);
    tr.addEventListener('keydown', ev => { if (ev.key === 'Enter') go(); });
  });
})();
"""


def _page(grid: dict, scored: pd.DataFrame, meta: dict, target_date: str,
          threshold: float) -> str:
    probs = scored["fire_probability"]
    alerts = (scored["fire_alert"] == 1) if "fire_alert" in scored else probs >= threshold
    n_total, n_alerts = len(scored), int(alerts.sum())
    peak_p, mean_p = float(probs.max()), float(probs.mean())

    metrics = meta.get("metrics") or {}
    test_metrics = metrics.get("test") or metrics
    auc = test_metrics.get("roc_auc", float("nan"))
    ap = test_metrics.get("average_precision", float("nan"))
    data_info = meta.get("data") or {}
    years = lambda k: "–".join(str(y) for y in data_info.get(k) or []) or "—"
    prod_years = data_info.get("production_years") or []
    prod_label = f"{prod_years[0]}–{prod_years[-1]}" if prod_years else years("train_years")
    trained_region = data_info.get("region") or "West Coast (CA, OR, WA)"

    # _kbdi() falls back to 300 when the KBDI collection has no images for
    # the window; a column that is 300 everywhere is that fallback, not data.
    kbdi_stale = "kbdi_index" in scored and scored["kbdi_index"].round().eq(300).all()
    kbdi_cell = (lambda r: "—") if kbdi_stale else (lambda r: f'{getattr(r, "kbdi_index", 0):.0f}')
    names = region.county_names()
    top = scored.nlargest(10, "fire_probability")
    top_county = region.locate(top["longitude"], top["latitude"])
    top_rows = "".join(
        f'<tr tabindex="0" data-lat="{r.latitude:.4f}" data-lon="{r.longitude:.4f}">'
        f'<td>{html.escape(names[c]) if c >= 0 else "—"}</td>'
        f'<td class="p" style="color:{ramp_color(r.fire_probability, threshold)}">{r.fire_probability:.3f}</td>'
        f'<td>{getattr(r, "temp_c", 0):.1f}</td><td>{getattr(r, "wind_speed_ms", 0):.1f}</td>'
        f'<td>{kbdi_cell(r)}</td></tr>'
        for r, c in zip(top.itertuples(), top_county)
    )
    legend_stops = ", ".join(f"{c} {100 * x / RAMP[-1][0]:.1f}%" for x, c in RAMP)
    sampled = int((grid["flag"] == 1).sum())
    filled = int((grid["flag"] == 2).sum())

    config = {
        "bounds": grid["bounds"], "nrows": grid["nrows"], "ncols": grid["ncols"],
        "dlon": grid["dlon"], "dy": grid["dy"], "yTop": grid["y_top"],
        "threshold": threshold, "maxZoom": MAX_ZOOM, "counties": names,
        "kbdiStale": bool(kbdi_stale),
        "ramp": [[x, list(_hex_rgb(c))] for x, c in RAMP],
        "data": _pack(grid),
    }
    js = (_JS.replace("__CONFIG__", json.dumps(config))
             .replace("__COUNTIES__", json.dumps(region.counties(), separators=(",", ":"))))
    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>California Wildfire Risk — {target_date}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>{_CSS}</style>
</head>
<body>
<div class="app">
  <div class="mapwrap">
    <div id="map" aria-label="California ignition risk map"></div>
    <div id="readout" class="readout" hidden></div>
  </div>

  <aside class="panel">
    <div class="eyebrow">California Wildfire Predictor</div>
    <h1>Risk briefing · {target_date}</h1>
    <div class="sub">7-day ignition probability · generated {gen_ts}</div>

    <h2>Run statistics</h2>
    <div class="kpis">
      <div class="kpi"><div class="l">Cells scored</div><div class="v">{n_total:,}</div></div>
      <div class="kpi alert"><div class="l">Alert cells</div><div class="v">{n_alerts:,}</div></div>
      <div class="kpi peak"><div class="l">Peak probability</div><div class="v">{peak_p:.3f}</div></div>
      <div class="kpi"><div class="l">Mean probability</div><div class="v">{mean_p:.4f}</div></div>
      <div class="kpi"><div class="l">Alert threshold</div><div class="v">{threshold:.3f}</div></div>
      <div class="kpi"><div class="l">Test ROC-AUC</div><div class="v">{auc:.3f}</div></div>
    </div>

    <h2>Map scale</h2>
    <div class="legend">
      <div class="bar" style="background:linear-gradient(90deg, {legend_stops})"></div>
      <div class="scale"><span>0</span><span>{threshold / 3:.3f}</span><span>{2 * threshold / 3:.3f}</span>
        <span style="color:var(--amber)">{threshold:.3f} alert</span><span>{2 * threshold:.2f}</span><span>≥ {3 * threshold:.2f}</span></div>
    </div>
    <label class="toggle"><input type="checkbox" id="shimmer"> Shimmer alert cells</label>

    <h2>Risk distribution</h2>
    {_histogram_svg(probs, threshold)}
    <div class="hint">cells per probability bin, square-root scale</div>

    <h2>Top 10 cells</h2>
    <table id="top10">
      <thead><tr><th>County</th><th>p</th><th>°C</th><th>m/s</th><th>KBDI</th></tr></thead>
      <tbody>{top_rows}</tbody>
    </table>
    <div class="hint">click a row to fly to the cell</div>

    <h2>Feature importance</h2>
    {_feature_importance_svg(meta)}

    <div class="meta">
      <b>Model</b> XGBoost ignition classifier, production fit {prod_label}, trained on {html.escape(trained_region)} samples<br>
      <b>Evaluation</b> train {years("train_years")} · threshold tuned on validation {years("val_years")} · tested out of time {years("test_years")} · ROC-AUC {auc:.3f} · AUC-PR {ap:.4f}<br>
      <b>Features</b> ERA5-Land · SRTM · NASA SMAP · KBDI · MODIS NDVI<br>
      {'<b style="color:var(--amber)">KBDI</b> the KBDI source returned no images for this window, so every cell was scored with the 300 fallback<br>' if kbdi_stale else ''}
      <b>Labels</b> NASA FIRMS MODIS thermal anomalies<br>
      <b>Grid</b> {CFG.cell_scale_m / 1000:g} km cells · {sampled:,} sampled · {filled:,} filled from neighbours · county lines: Code for Germany click_that_hood
    </div>
  </aside>
</div>
<script>{js}</script>
</body></html>
"""


def render(scored: pd.DataFrame, meta: dict, target_date: str, summary: dict,
           out_html: Path) -> None:
    from .predict import resolve_threshold

    threshold = float(summary.get("operational_threshold") or resolve_threshold(meta))
    scored = region.restrict(scored)
    grid = rasterize(scored)
    page = _page(grid, scored, meta, target_date, threshold)
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
