"""
Combined wildfire risk dashboard CLI.

For a target date (defaults to today), this:
  1. Pulls a fresh GEE snapshot of the West Coast covering a 7-day window
     ending on the target date.
  2. Scores every cell with the XGBoost ignition model -> per-cell ignition
     probability for the next 7 days.
  3. For high-risk cells, queries FIRMS for active hot spots and runs the
     U-Net spread model on those local patches -> per-pixel next-day spread
     masks.
  4. Writes a single .geojson + .parquet + .html dashboard summarizing both
     model outputs (ignition map + active-fire spread predictions).

Usage:
    # Default: today, both models
    python -m wildfire.dashboard

    # Specific date, ignition-only
    python -m wildfire.dashboard --date 2025-09-15 --ignition-only

    # Different output dir
    python -m wildfire.dashboard --out wildfire_output/risk_2025-09-15

Requires:
  * GEE auth + GEE_PROJECT set
  * trained XGBoost ignition model
  * trained U-Net spread model
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import CFG


def _ignition_scores(target_date: str) -> tuple[pd.DataFrame, tuple[str, str]]:
    """Pull a fresh GEE window and score with the XGBoost ignition model.

    ERA5-Land lags realtime by ~5 days, so a 7-day window ending today can
    come back thin or empty. If that happens, walk the window back in 2-day
    steps (up to 8 days) and report the window actually used.
    """
    from .data_extraction import init_ee, extract_window
    from .predict import score_dataframe

    init_ee()

    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    last_err: Optional[Exception] = None
    for shift_days in (0, 2, 4, 6, 8):
        end = target - timedelta(days=shift_days)
        start = end - timedelta(days=7)
        print(f"[ignition] pulling fresh GEE features for {start} .. {end} ...")
        try:
            df = extract_window(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        except Exception as e:  # GEE raises on fully-empty ERA5 windows
            print(f"[ignition]   window failed ({e}); trying an earlier window ...", file=sys.stderr)
            last_err = e
            continue
        if df.empty:
            print("[ignition]   window returned no rows (data latency); trying an earlier window ...")
            continue
        if shift_days:
            print(f"[ignition] note: freshest complete data window is {start} .. {end} "
                  f"({shift_days} days behind the target date)")
        print(f"[ignition] scoring {len(df):,} cells ...")
        scored = score_dataframe(df)
        return scored, (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    raise RuntimeError(
        f"No data returned from GEE for any window near {target_date}. "
        f"Check auth/quota. Last error: {last_err}"
    )


def _spread_predictions(scored: pd.DataFrame,
                        top_k: int = 50,
                        target_date: str = "") -> dict:
    """Run U-Net spread model on REAL GEE imagery patches around the top-risk cells.

    For each of the top-K highest-risk cells we pull a 64x64 NDWS-format
    imagery patch from Google Earth Engine (elevation, slope, aspect, NDVI,
    temperature, dewpoint, wind, KBDI, soil moisture, prior fire mask),
    feed it to the trained U-Net, and record the predicted next-day spread
    mask statistics.
    """
    try:
        import torch
    except ImportError:
        print("[spread] torch not installed; skipping spread predictions.")
        return {}

    from .spread.config import SPREAD_CFG
    from .spread.model import build_model
    from .data_extraction import extract_patch

    ckpt_path = SPREAD_CFG.model_path()
    if not ckpt_path.exists():
        print(f"[spread] no trained spread model at {ckpt_path}; skipping.")
        return {}

    top = scored.sort_values("fire_probability", ascending=False).head(top_k).reset_index(drop=True)
    if top.empty:
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["in_channels"], ckpt["lookback_days"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    s = SPREAD_CFG.patch_size
    T = ckpt["lookback_days"]

    # Date window for the imagery patches: 7-day lookback ending at target_date
    end = datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else datetime.utcnow().date()
    start = end - timedelta(days=7)
    day_start = start.strftime("%Y-%m-%d")
    day_end   = end.strftime("%Y-%m-%d")

    print(f"[spread] pulling real GEE imagery patches for {len(top)} top-risk cells "
          f"(this calls GEE per patch — slower but accurate) ...")

    spread_results = []
    for i, row in top.iterrows():
        lon, lat = float(row["longitude"]), float(row["latitude"])
        try:
            patch = extract_patch(lon, lat, day_start, day_end,
                                  size_px=s, pixel_km=1.0)            # (C, H, W)
        except Exception as e:
            print(f"  patch {i} skipped ({lon:.3f}, {lat:.3f}): {e}", file=sys.stderr)
            continue

        x = np.broadcast_to(patch[None], (T, *patch.shape)).copy()    # (T, C, H, W)
        x = torch.from_numpy(x[None]).to(device)                      # (1, T, C, H, W)
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]          # (H, W)

        spread_results.append({
            "longitude": lon,
            "latitude":  lat,
            "ignition_probability": float(row["fire_probability"]),
            "spread_prob_max":  float(prob.max()),
            "spread_prob_mean": float(prob.mean()),
            "spread_pixels_above_50pct": int((prob >= 0.5).sum()),
            "spread_mask_shape": list(prob.shape),
            "patch_source": "real_gee_imagery",
        })

    return {
        "target_date": target_date,
        "top_k": top_k,
        "patch_source": "real_gee_imagery",
        "patches": spread_results,
    }


def _resolve_meta_threshold() -> float:
    """The threshold actually applied to fire_alert (validation-tuned, from
    model metadata) — shown in the HTML so display matches behavior."""
    from .predict import resolve_threshold
    meta = {}
    if CFG.feature_meta_path().exists():
        meta = json.loads(CFG.feature_meta_path().read_text())
    return resolve_threshold(meta)


def _write_outputs(scored: pd.DataFrame, spread: dict, out_dir: Path, target_date: str,
                   data_window: tuple[str, str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tabular output — every cell with ignition probability
    parquet_path = out_dir / f"ignition_{target_date}.parquet"
    scored.to_parquet(parquet_path, index=False)

    # GeoJSON for QGIS / web maps
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r["longitude"]), float(r["latitude"])]},
                "properties": {
                    "fire_probability": float(r["fire_probability"]),
                    "fire_alert":       int(r["fire_alert"]),
                    "temp_c":           float(r.get("temp_c", 0.0)),
                    "wind_speed_ms":    float(r.get("wind_speed_ms", 0.0)),
                    "kbdi_index":       float(r.get("kbdi_index", 0.0)),
                },
            }
            for _, r in scored.iterrows()
        ],
    }
    (out_dir / f"ignition_{target_date}.geojson").write_text(json.dumps(geojson))

    # Summary JSON (always lightweight)
    target_month = datetime.strptime(target_date, "%Y-%m-%d").month
    summary = {
        "target_date": target_date,
        "data_window": {"start": data_window[0], "end": data_window[1]} if data_window else None,
        "operational_threshold": _resolve_meta_threshold(),
        "n_cells_scored": int(len(scored)),
        "n_high_risk_cells": int((scored["fire_alert"] == 1).sum()),
        "max_ignition_probability": float(scored["fire_probability"].max()),
        "mean_ignition_probability": float(scored["fire_probability"].mean()),
        "spread_predictions": spread,
    }
    if target_month not in CFG.model_coverage_months:
        summary["out_of_season_note"] = (
            f"Target month {target_month} is outside the May-Oct training coverage; "
            "the model is extrapolating on the calendar features."
        )
    (out_dir / f"summary_{target_date}.json").write_text(json.dumps(summary, indent=2, default=str))

    # Single-page HTML dashboard
    _write_html_dashboard(scored, spread, out_dir, target_date, summary)


def _write_html_dashboard(scored: pd.DataFrame, spread: dict, out_dir: Path,
                          target_date: str, summary: dict) -> None:
    top = scored.sort_values("fire_probability", ascending=False).head(20)
    top_rows = "".join(
        f"<tr><td>{r['longitude']:.3f}</td><td>{r['latitude']:.3f}</td>"
        f"<td>{r['fire_probability']:.3f}</td><td>{r.get('temp_c', 0):.1f}</td>"
        f"<td>{r.get('wind_speed_ms', 0):.1f}</td><td>{r.get('kbdi_index', 0):.0f}</td></tr>"
        for _, r in top.iterrows()
    )

    spread_section = ""
    if spread.get("patches"):
        spread_rows = "".join(
            f"<tr><td>{p['longitude']:.3f}</td><td>{p['latitude']:.3f}</td>"
            f"<td>{p['ignition_probability']:.3f}</td><td>{p['spread_prob_max']:.3f}</td>"
            f"<td>{p['spread_pixels_above_50pct']}</td></tr>"
            for p in spread["patches"][:20]
        )
        spread_section = f"""
        <h2>Top {len(spread['patches'])} fire-spread predictions (U-Net on high-risk cells)</h2>
        <table>
          <thead><tr><th>Lon</th><th>Lat</th><th>Ignition p</th><th>Max spread p</th><th>Fire pixels @50%</th></tr></thead>
          <tbody>{spread_rows}</tbody>
        </table>
        """

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>West Coast Wildfire Risk — {target_date}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1 {{ border-bottom: 2px solid #d33; padding-bottom: .3em; }}
h2 {{ color: #555; margin-top: 2em; }}
.card {{ display: inline-block; background: #f6f6f6; border-left: 4px solid #d33; padding: 1em 1.5em; margin: .5em 1em .5em 0; min-width: 180px; }}
.card .v {{ font-size: 1.8em; font-weight: 600; color: #d33; }}
.card .l {{ font-size: .85em; color: #666; text-transform: uppercase; letter-spacing: .05em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .9em; }}
th {{ background: #d33; color: white; padding: .5em; text-align: left; }}
td {{ padding: .4em .5em; border-bottom: 1px solid #eee; }}
tr:nth-child(even) {{ background: #fafafa; }}
.meta {{ color: #888; font-size: .85em; }}
</style>
</head><body>
<h1>West Coast wildfire risk — {target_date}</h1>
<p class="meta">Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}. Models: XGBoost ignition + U-Net spread.</p>

<div>
  <div class="card"><div class="v">{summary["n_cells_scored"]:,}</div><div class="l">Cells scored</div></div>
  <div class="card"><div class="v">{summary["n_high_risk_cells"]:,}</div><div class="l">High-risk alerts</div></div>
  <div class="card"><div class="v">{summary["max_ignition_probability"]:.3f}</div><div class="l">Peak ignition prob</div></div>
  <div class="card"><div class="v">{len(spread.get("patches", []))}</div><div class="l">Spread predictions</div></div>
</div>

<h2>Top 20 highest-risk cells</h2>
<table>
  <thead><tr><th>Lon</th><th>Lat</th><th>Ignition p</th><th>Temp (°C)</th><th>Wind (m/s)</th><th>KBDI</th></tr></thead>
  <tbody>{top_rows}</tbody>
</table>

{spread_section}

<h2>Files</h2>
<ul>
  <li><code>ignition_{target_date}.parquet</code> — every scored cell</li>
  <li><code>ignition_{target_date}.geojson</code> — for QGIS / web maps</li>
  <li><code>summary_{target_date}.json</code> — top-level metrics</li>
</ul>
<p class="meta">Operational threshold: {summary["operational_threshold"]:.3f} for ignition alerts
(tuned on the validation year, read from model metadata).
{f"Data window: {summary['data_window']['start']} → {summary['data_window']['end']}." if summary.get("data_window") else ""}
{f"<br><b>Note:</b> {summary['out_of_season_note']}" if summary.get("out_of_season_note") else ""}
<br>Spread predictions run U-Net on real GEE imagery patches around top-risk cells —
research/triage signal, not perimeter-accurate forecasts.</p>
</body></html>
"""
    (out_dir / f"dashboard_{target_date}.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Combined wildfire risk dashboard.")
    parser.add_argument("--date", type=str, default=datetime.utcnow().strftime("%Y-%m-%d"),
                        help="Target date (YYYY-MM-DD). Default: today UTC.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: wildfire_output/<date>/)")
    parser.add_argument("--ignition-only", action="store_true",
                        help="Skip spread predictions.")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Run spread model on top-K highest-risk cells.")
    args = parser.parse_args()

    out_dir = args.out or Path(__file__).resolve().parent.parent / "wildfire_output" / args.date

    scored, data_window = _ignition_scores(args.date)
    spread = {} if args.ignition_only else _spread_predictions(scored, top_k=args.top_k, target_date=args.date)
    _write_outputs(scored, spread, out_dir, args.date, data_window=data_window)

    # Render the HUD heatmap (satellite-style visualization)
    try:
        from .heatmap import render as render_heatmap
        meta = json.loads(CFG.feature_meta_path().read_text()) if CFG.feature_meta_path().exists() else {}
        summary = json.loads((out_dir / f"summary_{args.date}.json").read_text())
        render_heatmap(scored, meta, args.date, summary, out_dir / f"heatmap_{args.date}.html")
    except Exception as e:
        print(f"  [heatmap] skipped: {e}", file=sys.stderr)

    print(f"\n=== Dashboard complete for {args.date} ===")
    print(f"  high-risk alerts : {int((scored['fire_alert'] == 1).sum()):,} / {len(scored):,} cells")
    print(f"  peak ignition p  : {scored['fire_probability'].max():.3f}")
    if spread.get("patches"):
        print(f"  spread runs      : {len(spread['patches'])} patches")
    print(f"  output           : {out_dir}/heatmap_{args.date}.html")


if __name__ == "__main__":
    main()
