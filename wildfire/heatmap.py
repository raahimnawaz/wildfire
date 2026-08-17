"""
High-tech satellite-style wildfire risk heatmap.

Reads a scored parquet (output of `wildfire.dashboard`) and produces a
self-contained HTML page with:
  - A dark Leaflet basemap (CartoDB DarkMatter, deep-ops aesthetic)
  - Per-cell heat circles colored by ignition probability
  - HUD-style side panel with:
      * Top-10 at-risk cells (sortable table)
      * Risk-band distribution histogram (inline SVG)
      * Feature importances from the trained XGBoost model
      * Run metadata (timestamp, model version, decision threshold)

Dependencies: folium (Leaflet wrapper), branca (colormaps).

Usage:
    # After running the dashboard
    python -m wildfire.heatmap --input wildfire_output/2025-09-15/ignition_2025-09-15.parquet

    # Or point at a specific output directory
    python -m wildfire.heatmap --date 2025-09-15
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap
from branca.element import MacroElement
from jinja2 import Template
import pandas as pd

from .config import CFG


# Zoom level above which the heat layer is hidden. A heatmap is a density
# estimate over a 2 km grid: it reads well at regional zoom, but zoomed in the
# kernels smear into big blurry washes that cover the basemap and imply spatial
# precision the model does not have. Past this level the crisp high-risk markers
# carry the information instead.
HEAT_HIDE_ABOVE_ZOOM = 9


class ZoomAwareLayer(MacroElement):
    """Hide a layer once the map is zoomed in past `hide_above`.

    Restores it on zooming back out, but respects the user: if they switch the
    layer off by hand in the layer control, it stays off until they switch it
    back on.
    """

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            var layer = {{ this.layer.get_name() }};
            var hideAbove = {{ this.hide_above }};
            var userOff = false;

            map.on('overlayremove', function (e) {
                if (e.layer === layer && map.getZoom() <= hideAbove) { userOff = true; }
            });
            map.on('overlayadd', function (e) {
                if (e.layer === layer) { userOff = false; }
            });

            function sync() {
                var tooClose = map.getZoom() > hideAbove;
                if (tooClose || userOff) {
                    if (map.hasLayer(layer)) { map.removeLayer(layer); }
                } else if (!map.hasLayer(layer)) {
                    map.addLayer(layer);
                }
            }
            map.on('zoomend', sync);
            sync();
        })();
        {% endmacro %}
    """)

    def __init__(self, layer, hide_above: int = HEAT_HIDE_ABOVE_ZOOM):
        super().__init__()
        self._name = "ZoomAwareLayer"
        self.layer = layer
        self.hide_above = int(hide_above)


# ─────────────────────────────────────────────────────────────────────────────
# Color scale — risk tier ramp (deep blue → cyan → amber → red → magenta-red)
# ─────────────────────────────────────────────────────────────────────────────
RISK_COLORS = ["#0a4a6e", "#00d4ff", "#ffd60a", "#ff6b00", "#ff1744"]


def _make_colormap(vmin: float = 0.0, vmax: float = 1.0) -> LinearColormap:
    cm = LinearColormap(RISK_COLORS, vmin=vmin, vmax=vmax)
    cm.caption = "Ignition probability (next 7 days)"
    return cm


def _build_map(scored: pd.DataFrame, cm: LinearColormap) -> folium.Map:
    # Center on the geographic centroid of sampled cells
    center = [float(scored["latitude"].mean()), float(scored["longitude"].mean())]

    fmap = folium.Map(
        location=center,
        zoom_start=6,
        tiles=None,                       # we'll add tile layers explicitly
        control_scale=True,
        prefer_canvas=True,               # much faster for 30k+ markers
        attributionControl=False,
    )

    # Dark "ops" basemap
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
        attr="© CARTO © OpenStreetMap",
        name="Dark Matter",
        max_zoom=19,
        control=True,
    ).add_to(fmap)

    # Optional satellite layer (toggleable)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="© Esri",
        name="Satellite (Esri)",
        max_zoom=19,
        control=True,
        show=False,
    ).add_to(fmap)

    # Smooth heatmap layer for the bulk of the cells — fast, satellite-feel
    heat_data = [
        [float(r["latitude"]), float(r["longitude"]), float(r["fire_probability"])]
        for _, r in scored.iterrows()
    ]
    heat = HeatMap(
        heat_data,
        name=f"Ignition heat (regional, zoom ≤ {HEAT_HIDE_ABOVE_ZOOM})",
        radius=10,           # tighter footprint reduces offshore bleed from coastal cells
        blur=12,
        min_opacity=0.35,
        max_zoom=HEAT_HIDE_ABOVE_ZOOM,   # reach full intensity by the hide threshold
        gradient={
            "0.0":  "#0a4a6e",
            "0.25": "#00d4ff",
            "0.5":  "#ffd60a",
            "0.75": "#ff6b00",
            "1.0":  "#ff1744",
        },
    )
    heat.add_to(fmap)
    # Drop the heat layer once zoomed past regional scale — the kernels smear
    # into a wash that hides the basemap, and the high-risk markers below are
    # the honest representation at that scale.
    ZoomAwareLayer(heat, HEAT_HIDE_ABOVE_ZOOM).add_to(fmap)

    # Crisp circle markers ONLY for the most at-risk cells (top 1% or anything >= 0.4)
    p_cutoff = max(0.4, float(scored["fire_probability"].quantile(0.99)))
    focal = scored[scored["fire_probability"] >= p_cutoff].copy()
    focal_layer = folium.FeatureGroup(name=f"High-risk cells (p ≥ {p_cutoff:.2f})", show=True)

    for _, row in focal.iterrows():
        p = float(row["fire_probability"])
        color = cm(p)
        popup_html = (
            f"<div style='font-family: \"DM Mono\", monospace; font-size: 11px; color: #c8d3e0;'>"
            f"<b style='color:#fff'>Cell ({row['longitude']:.3f}, {row['latitude']:.3f})</b><br>"
            f"<span style='color:#6c7a89'>Ignition p</span> "
            f"<b style='color:{color}'>{p:.3f}</b><br>"
            f"<span style='color:#6c7a89'>Temp</span> {row.get('temp_c', 0):.1f}°C<br>"
            f"<span style='color:#6c7a89'>Wind</span> {row.get('wind_speed_ms', 0):.1f} m/s<br>"
            f"<span style='color:#6c7a89'>KBDI</span> {row.get('kbdi_index', 0):.0f}<br>"
            f"<span style='color:#6c7a89'>RH</span>   {row.get('rh_pct', 0):.0f}%<br>"
            f"<span style='color:#6c7a89'>NDVI</span> {row.get('ndvi', 0):.2f}"
            f"</div>"
        )
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=4 + 8 * p,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(focal_layer)

        # Crosshair ring for visual emphasis on alerts
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=14 + 8 * p,
            color=color,
            fill=False,
            weight=1,
            opacity=0.5,
        ).add_to(focal_layer)

    focal_layer.add_to(fmap)
    cm.add_to(fmap)
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    return fmap


def _histogram_svg(probs: pd.Series, width: int = 280, height: int = 90) -> str:
    """Tiny inline-SVG histogram of fire probabilities — no JS, ships in the page."""
    n_bins = 20
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in probs:
        idx = min(int(v * n_bins), n_bins - 1)
        counts[idx] += 1
    max_c = max(counts) or 1
    bar_w = (width - 20) / n_bins

    bars = []
    for i, c in enumerate(counts):
        h = (height - 20) * (c / max_c)
        x = 10 + i * bar_w
        y = (height - 10) - h
        # Color band matching the colormap
        t = (i + 0.5) / n_bins
        if t < 0.25:    color = "#0a4a6e"
        elif t < 0.50:  color = "#00d4ff"
        elif t < 0.75:  color = "#ffd60a"
        elif t < 0.90:  color = "#ff6b00"
        else:           color = "#ff1744"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 0.5:.1f}" '
            f'height="{h:.1f}" fill="{color}" opacity="0.85"/>'
        )

    axis = (
        f'<line x1="10" y1="{height - 10}" x2="{width - 10}" y2="{height - 10}" '
        f'stroke="#3a4654" stroke-width="0.5"/>'
        f'<text x="10" y="{height - 1}" fill="#6c7a89" font-size="8" '
        f'font-family="DM Mono, monospace">0.0</text>'
        f'<text x="{width - 25}" y="{height - 1}" fill="#6c7a89" font-size="8" '
        f'font-family="DM Mono, monospace">1.0</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%; height:auto; background:#0d141c; border:1px solid #1f2733;">'
        f'{"".join(bars)}{axis}</svg>'
    )


def _feature_importance_bars(meta: dict, top_n: int = 8) -> str:
    """Inline-SVG horizontal bar chart of the top feature importances."""
    fi = meta.get("feature_importance") or {}
    if not fi:
        return '<div style="color:#6c7a89; font-size:11px;">No feature importance found.</div>'
    items = list(fi.items())[:top_n]
    max_v = max(v for _, v in items) or 1.0
    height = 28 * len(items)
    width = 280

    rows = []
    for i, (name, v) in enumerate(items):
        bar_w = (width - 130) * (v / max_v)
        y = 14 + i * 28
        rows.append(
            f'<text x="0" y="{y + 4}" fill="#c8d3e0" font-size="11" '
            f'font-family="DM Mono, monospace">{name[:18]}</text>'
            f'<rect x="125" y="{y - 8}" width="{bar_w:.1f}" height="14" '
            f'fill="url(#fiGrad)" rx="1"/>'
            f'<text x="{125 + bar_w + 4:.1f}" y="{y + 4}" fill="#6c7a89" font-size="10" '
            f'font-family="DM Mono, monospace">{v:.3f}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%; height:auto;">'
        f'<defs><linearGradient id="fiGrad" x1="0" x2="1">'
        f'<stop offset="0" stop-color="#00d4ff"/>'
        f'<stop offset="1" stop-color="#ff6b00"/></linearGradient></defs>'
        f'{"".join(rows)}</svg>'
    )


def _hud_shell(map_html: str, scored: pd.DataFrame, meta: dict,
               target_date: str, summary: dict) -> str:
    """Wrap the folium map in a HUD-style HTML shell."""
    n_total      = int(summary["n_cells_scored"])
    n_alerts     = int(summary["n_high_risk_cells"])
    peak_p       = float(summary["max_ignition_probability"])
    mean_p       = float(summary["mean_ignition_probability"])
    from .predict import resolve_threshold
    threshold    = resolve_threshold(meta)  # the value fire_alert actually uses
    metrics      = (meta.get("metrics") or {})
    test_metrics = metrics.get("test") or metrics  # new nested layout, legacy flat fallback
    auc          = test_metrics.get("roc_auc", float("nan"))
    ap           = test_metrics.get("average_precision", float("nan"))
    data_info    = meta.get("data") or {}
    train_year   = ("-".join(str(y) for y in data_info["train_years"])
                    if data_info.get("train_years") else meta.get("train_year", "—"))
    test_year    = ("-".join(str(y) for y in data_info["test_years"])
                    if data_info.get("test_years") else meta.get("test_year", "—"))
    prod_years   = data_info.get("production_years") or []
    prod_label   = (f"{prod_years[0]}-{prod_years[-1]}" if prod_years else train_year)

    top10 = scored.sort_values("fire_probability", ascending=False).head(10)
    top_rows = "".join(
        f"<tr class='clickable' data-lat='{r['latitude']:.4f}' data-lon='{r['longitude']:.4f}'>"
        f"<td>{r['longitude']:.3f}</td><td>{r['latitude']:.3f}</td>"
        f"<td class='p' style='color:{_band(r['fire_probability'])}'>"
        f"{r['fire_probability']:.3f}</td>"
        f"<td>{r.get('temp_c', 0):.1f}</td>"
        f"<td>{r.get('wind_speed_ms', 0):.1f}</td>"
        f"<td>{r.get('kbdi_index', 0):.0f}</td></tr>"
        for _, r in top10.iterrows()
    )

    hist_svg = _histogram_svg(scored["fire_probability"])
    fi_svg = _feature_importance_bars(meta)
    gen_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>WCWP // {target_date} // West Coast Wildfire Risk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; background: #050810; color: #c8d3e0; font-family: 'Inter', system-ui, sans-serif;
              font-size: 13px; overflow: hidden; }}
.app {{ display: grid; grid-template-columns: 1fr 360px; height: 100vh; }}
.map {{ position: relative; background: #050810; }}
.map .frame {{ position: absolute; inset: 12px; border: 1px solid #1f2733;
               border-radius: 2px; overflow: hidden; box-shadow: 0 0 0 1px #050810; }}
.map .frame .corner {{ position: absolute; width: 16px; height: 16px;
                       border-color: #00d4ff; border-style: solid; border-width: 0;
                       pointer-events: none; z-index: 1000; }}
.map .frame .corner.tl {{ top: -1px; left: -1px; border-top-width: 1px; border-left-width: 1px; }}
.map .frame .corner.tr {{ top: -1px; right: -1px; border-top-width: 1px; border-right-width: 1px; }}
.map .frame .corner.bl {{ bottom: -1px; left: -1px; border-bottom-width: 1px; border-left-width: 1px; }}
.map .frame .corner.br {{ bottom: -1px; right: -1px; border-bottom-width: 1px; border-right-width: 1px; }}
.map .frame > div {{ height: 100%; }}
.map iframe {{ border: 0; width: 100%; height: 100%; }}
.overlay {{ position: absolute; top: 24px; left: 24px; z-index: 1001;
            background: rgba(8, 13, 22, 0.85); border: 1px solid #1f2733;
            padding: 10px 14px; font-family: 'DM Mono', monospace; font-size: 11px;
            line-height: 1.6; backdrop-filter: blur(6px); }}
.overlay .k {{ color: #6c7a89; }}
.overlay .v {{ color: #00d4ff; }}
.overlay .alert {{ color: #ff6b00; }}
.panel {{ background: #0a121c; border-left: 1px solid #1f2733; overflow-y: auto;
          padding: 18px 18px 24px; }}
.panel::-webkit-scrollbar {{ width: 6px; }}
.panel::-webkit-scrollbar-thumb {{ background: #1f2733; border-radius: 3px; }}
.panel h1 {{ font-family: 'DM Mono', monospace; font-size: 11px; color: #6c7a89;
             font-weight: 400; letter-spacing: 0.18em; text-transform: uppercase;
             margin-bottom: 8px; }}
.panel h2 {{ font-family: 'DM Mono', monospace; font-size: 10px; color: #6c7a89;
             font-weight: 400; letter-spacing: 0.15em; text-transform: uppercase;
             margin: 22px 0 8px; border-bottom: 1px solid #1f2733; padding-bottom: 6px; }}
.banner {{ font-family: 'Inter', sans-serif; font-size: 17px; font-weight: 600;
           color: #ffffff; margin-bottom: 4px; letter-spacing: 0.02em; }}
.sub {{ font-family: 'DM Mono', monospace; font-size: 11px; color: #6c7a89;
        margin-bottom: 12px; }}
.kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.kpi {{ background: #0d141c; border: 1px solid #1f2733; padding: 10px 12px; }}
.kpi .l {{ font-family: 'DM Mono', monospace; font-size: 9px; color: #6c7a89;
           letter-spacing: 0.12em; text-transform: uppercase; }}
.kpi .v {{ font-family: 'DM Mono', monospace; font-size: 18px; color: #ffffff;
           margin-top: 4px; font-weight: 500; }}
.kpi.alert .v {{ color: #ff6b00; }}
.kpi.crit  .v {{ color: #ff1744; }}
.kpi.ok    .v {{ color: #00d4ff; }}
table {{ width: 100%; border-collapse: collapse; font-family: 'DM Mono', monospace;
         font-size: 10px; }}
th {{ color: #6c7a89; text-align: left; padding: 6px 4px; border-bottom: 1px solid #1f2733;
      font-weight: 400; letter-spacing: 0.05em; text-transform: uppercase; }}
td {{ padding: 5px 4px; border-bottom: 1px solid #131b25; color: #c8d3e0; }}
td.p {{ font-weight: 500; }}
tr.clickable {{ cursor: pointer; transition: background 0.1s; }}
tr.clickable:hover td {{ background: #1a2532; color: #00d4ff; }}
tr.clickable:active td {{ background: #00d4ff22; }}
.hint {{ font-family: 'DM Mono', monospace; font-size: 10px; color: #6c7a89;
        margin-top: 6px; font-style: italic; }}
.meta {{ font-family: 'DM Mono', monospace; font-size: 10px; color: #6c7a89;
         line-height: 1.7; margin-top: 18px; padding-top: 14px;
         border-top: 1px solid #1f2733; }}
.meta b {{ color: #c8d3e0; font-weight: 500; }}
.tag {{ display: inline-block; padding: 2px 7px; font-size: 9px; font-family: 'DM Mono', monospace;
        border: 1px solid #1f2733; color: #6c7a89; letter-spacing: 0.1em;
        text-transform: uppercase; margin-right: 4px; }}
.tag.live {{ color: #00d4ff; border-color: #00d4ff44; }}
.tag.warn {{ color: #ff6b00; border-color: #ff6b0044; }}
.divider {{ height: 1px; background: linear-gradient(90deg, transparent, #1f2733, transparent);
            margin: 18px 0; }}
.crosshair {{ position: absolute; pointer-events: none; }}
</style></head>
<body>
<div class="app">
  <div class="map">
    <div class="frame">
      <span class="corner tl"></span><span class="corner tr"></span>
      <span class="corner bl"></span><span class="corner br"></span>
      {map_html}
    </div>
    <div class="overlay">
      <div><span class="k">SYS</span> <span class="v">WCWP-1.0</span></div>
      <div><span class="k">TGT</span> <span class="v">{target_date}</span></div>
      <div><span class="k">CELLS</span> <span class="v">{n_total:,}</span></div>
      <div><span class="k">ALERTS</span> <span class="{'alert' if n_alerts > 0 else 'v'}">{n_alerts}</span></div>
      <div><span class="k">PEAK</span> <span class="alert">{peak_p:.3f}</span></div>
    </div>
  </div>

  <div class="panel">
    <h1>// West Coast Wildfire Predictor</h1>
    <div class="banner">Risk briefing &mdash; {target_date}</div>
    <div class="sub">{gen_ts}</div>
    <div>
      <span class="tag live">LIVE GEE</span>
      <span class="tag">XGBoost</span>
      <span class="tag">U-Net</span>
      {'<span class="tag warn">ALERTS ACTIVE</span>' if n_alerts > 0 else ''}
    </div>

    <h2>Run statistics</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="l">Cells scored</div><div class="v">{n_total:,}</div></div>
      <div class="kpi {'alert' if n_alerts > 0 else 'ok'}"><div class="l">High-risk alerts</div><div class="v">{n_alerts}</div></div>
      <div class="kpi crit"><div class="l">Peak ignition p</div><div class="v">{peak_p:.3f}</div></div>
      <div class="kpi"><div class="l">Mean ignition p</div><div class="v">{mean_p:.4f}</div></div>
      <div class="kpi ok"><div class="l">Threshold</div><div class="v">{threshold:.3f}</div></div>
      <div class="kpi"><div class="l">Model AUC</div><div class="v">{auc:.3f}</div></div>
    </div>

    <h2>Risk distribution</h2>
    {hist_svg}

    <h2>Top 10 at-risk cells</h2>
    <div class="hint">click any row to fly the map to that cell</div>
    <table id="top10">
      <thead><tr><th>Lon</th><th>Lat</th><th>p</th><th>°C</th><th>m/s</th><th>KBDI</th></tr></thead>
      <tbody>{top_rows}</tbody>
    </table>

    <h2>Feature importance</h2>
    {fi_svg}

    <div class="meta">
      <b>MODEL</b> XGBoost binary classifier, 500 trees, depth 6 · production fit {prod_label}<br>
      <b>EVAL</b> train {train_year} · threshold tuned on validation · tested out-of-time {test_year}<br>
      <b>FEATURES</b> ERA5-Land · SRTM · NASA SMAP · KBDI · MODIS NDVI<br>
      <b>LABELS</b> NASA FIRMS MODIS T21 thermal anomalies<br>
      <b>METRICS</b> ROC-AUC {auc:.3f} · AUC-PR {ap:.4f}<br>
      <b>FRAME</b> EPSG:4326 · ~{CFG.cell_scale_m / 1000:g} km cells · 7-day window
    </div>
  </div>
</div>

<script>
// Bridge between HUD table rows and the Leaflet map inside the iframe.
// Each row carries data-lat/data-lon. On click we postMessage to the iframe
// which has a listener that flies the Leaflet map to those coordinates.
document.addEventListener('DOMContentLoaded', () => {{
  const iframe = document.querySelector('.map iframe');
  document.querySelectorAll('#top10 tbody tr.clickable').forEach(row => {{
    row.addEventListener('click', () => {{
      const lat = parseFloat(row.dataset.lat);
      const lon = parseFloat(row.dataset.lon);
      if (!isNaN(lat) && !isNaN(lon) && iframe && iframe.contentWindow) {{
        iframe.contentWindow.postMessage({{action: 'flyTo', lat, lon, zoom: 9}}, '*');
      }}
    }});
  }});
}});
</script>
</body></html>
"""


def _band(p: float) -> str:
    if p < 0.25:   return "#00d4ff"
    if p < 0.50:   return "#ffd60a"
    if p < 0.75:   return "#ff6b00"
    return "#ff1744"


def render(scored: pd.DataFrame, meta: dict, target_date: str, summary: dict,
           out_html: Path) -> None:
    cm = _make_colormap()
    fmap = _build_map(scored, cm)
    map_html = fmap.get_root().render()
    page = _hud_shell(map_html=_to_iframe(map_html), scored=scored, meta=meta,
                      target_date=target_date, summary=summary)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    print(f"[heatmap] wrote {out_html}  ({out_html.stat().st_size / 1024:.0f} KB)")


_MAP_BRIDGE_JS = """
<script>
// Listen for messages from the parent HUD shell. The parent posts
// {action: 'flyTo', lat, lon, zoom} when a Top-10 row is clicked.
// We find the folium-generated Leaflet map by looking for the
// `map_<hash>` global that folium creates.
(function() {
  function findMap() {
    for (const key of Object.keys(window)) {
      if (key.startsWith('map_') && window[key] && typeof window[key].flyTo === 'function') {
        return window[key];
      }
    }
    return null;
  }
  window.addEventListener('message', function(evt) {
    const d = evt.data || {};
    if (d.action !== 'flyTo') return;
    const m = findMap();
    if (!m) return;
    try {
      m.flyTo([d.lat, d.lon], d.zoom || 9, {duration: 1.4});
      // pulse a temporary marker at the destination
      const ring = L.circleMarker([d.lat, d.lon], {
        radius: 24, color: '#00d4ff', weight: 2, fill: false, opacity: 0.9
      }).addTo(m);
      setTimeout(() => m.removeLayer(ring), 2200);
    } catch (e) {}
  });
})();
</script>
"""


def _to_iframe(map_html: str) -> str:
    """Inline the folium map via a sandboxed srcdoc iframe so its <head> and JS
    don't collide with our HUD shell. Injects a postMessage listener so the
    parent table rows can pan the map."""
    import html as _h
    # Inject the bridge before </body> in the folium HTML
    if "</body>" in map_html:
        map_html = map_html.replace("</body>", _MAP_BRIDGE_JS + "</body>", 1)
    else:
        map_html = map_html + _MAP_BRIDGE_JS
    return (f'<iframe srcdoc="{_h.escape(map_html, quote=True)}" '
            f'style="border:0; width:100%; height:100%;" '
            f'sandbox="allow-scripts allow-same-origin allow-popups"></iframe>')


def main():
    parser = argparse.ArgumentParser(description="Render the wildfire risk heatmap.")
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

    feature_meta_path = CFG.feature_meta_path()
    meta = json.loads(feature_meta_path.read_text()) if feature_meta_path.exists() else {}

    scored = pd.read_parquet(in_path)

    out_html = args.out or (out_dir / f"heatmap_{target_date}.html")
    render(scored, meta, target_date, summary, out_html)


if __name__ == "__main__":
    main()
