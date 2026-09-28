"""
Combined wildfire risk dashboard CLI.

For a target date (defaults to today), this:
  1. Pulls a fresh GEE snapshot of the West Coast covering a 7-day window
     ending on the target date.
  2. Scores every cell with the XGBoost ignition model -> per-cell
     probability of a NEW fire starting in the 7 days after the window
     (calibrated when the model artifact carries a calibration).
  3. Around cells that are burning now (FIRMS detections in the last two
     weeks), pulls NDWS-format patches from the same sources the spread
     U-Net was trained on and predicts next-day spread masks.
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
    from .data_extraction import drop_filled_rows, extract_window, init_ee
    from .predict import score_dataframe

    init_ee()

    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    last_err: Optional[Exception] = None
    for shift_days in (0, 2, 4, 6, 8):
        end = target - timedelta(days=shift_days)
        start = end - timedelta(days=7)
        print(f"[ignition] pulling fresh GEE features for {start} .. {end} ...")
        try:
            # labelled=False: the week being forecast hasn't happened yet.
            df = extract_window(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                                labelled=False)
        except Exception as e:  # GEE raises on fully-empty ERA5 windows; FeatureOutage on dead sources
            print(f"[ignition]   window failed ({e}); trying an earlier window ...", file=sys.stderr)
            last_err = e
            continue
        if df.empty:
            print("[ignition]   window returned no rows (data latency); trying an earlier window ...")
            continue
        if shift_days:
            print(f"[ignition] note: freshest complete data window is {start} .. {end} "
                  f"({shift_days} days behind the target date)")
        df, n_filled = drop_filled_rows(df)
        if n_filled:
            print(f"[ignition] dropped {n_filled:,} cells whose features are gap-fill "
                  "fallbacks (the model never trained on those values)")
        print(f"[ignition] scoring {len(df):,} cells ...")
        scored = score_dataframe(df)
        return scored, (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    raise RuntimeError(
        f"No data returned from GEE for any window near {target_date}. "
        f"Check auth/quota. Last error: {last_err}"
    )


def _active_fire_centers(scored: pd.DataFrame, top_k: int, patch_km: float) -> pd.DataFrame:
    """Patch centres over cells burning now (FIRMS in the last
    CFG.label_quiet_days), densest clusters first, at most one per patch.

    The spread model answers "given a fire, where is it tomorrow?" -- run on a
    patch with no fire it can only predict nothing.
    """
    empty = pd.DataFrame(columns=["longitude", "latitude", "burning_cells"])
    if "fire_recent" not in scored.columns:
        return empty
    pts = scored.loc[scored["fire_recent"] == 1, ["longitude", "latitude"]].to_numpy(float)
    if len(pts) == 0:
        return empty
    x_km = pts[:, 0] * 111.0 * np.cos(np.radians(pts[:, 1]))
    y_km = pts[:, 1] * 111.0
    half = patch_km / 2.0
    inside = ((np.abs(x_km[:, None] - x_km[None, :]) <= half)
              & (np.abs(y_km[:, None] - y_km[None, :]) <= half))
    density = inside.sum(axis=1)
    taken = np.zeros(len(pts), dtype=bool)
    centers = []
    for i in np.argsort(-density, kind="stable"):
        if taken[i]:
            continue
        centers.append((pts[i, 0], pts[i, 1], int(density[i])))
        taken |= inside[i]
        if len(centers) >= top_k:
            break
    return pd.DataFrame(centers, columns=["longitude", "latitude", "burning_cells"])


def _spread_predictions(scored: pd.DataFrame,
                        top_k: int = 50,
                        target_date: str = "") -> dict:
    """Run the U-Net spread model on active fires.

    Patches come from the datasets Next Day Wildfire Spread was built from
    (GRIDMET weather, GRIDMET PDSI, VIIRS NDVI, MOD14A1 fire mask), for a
    single day t -- the latest day with both weather and a fire mask -- so each
    channel means what it meant in training. The prediction is for day t+1.
    """
    try:
        import torch
    except ImportError:
        print("[spread] torch not installed; skipping spread predictions.")
        return {}

    from .spread.config import SPREAD_CFG
    from .spread.model import build_model
    from .data_extraction import extract_patch, ndws_input_day

    ckpt_path = SPREAD_CFG.model_path()
    if not ckpt_path.exists():
        print(f"[spread] no trained spread model at {ckpt_path}; skipping.")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("source") == "synthetic":
        print("[spread] checkpoint was trained on synthetic patches; skipping real imagery.")
        return {}
    trained_channels = ckpt.get("feature_channels")
    if trained_channels is not None and list(trained_channels) != list(SPREAD_CFG.feature_channels):
        print(f"[spread] checkpoint channels {trained_channels} differ from "
              f"{SPREAD_CFG.feature_channels}; skipping.")
        return {}

    s = SPREAD_CFG.patch_size
    centers = _active_fire_centers(scored, top_k, patch_km=s * SPREAD_CFG.cell_resolution_m / 1000.0)
    if centers.empty:
        print("[spread] no active fires in the scored cells; nothing to forecast.")
        return {"target_date": target_date, "patches": [], "note": "no active fires"}

    try:
        day = ndws_input_day(target_date or datetime.utcnow().strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"[spread] no complete NDWS inputs near {target_date}: {e}", file=sys.stderr)
        return {}

    model = build_model(ckpt["arch"], ckpt["in_channels"], ckpt["lookback_days"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    T = ckpt["lookback_days"]
    prev_idx = list(SPREAD_CFG.feature_channels).index("PrevFireMask")

    print(f"[spread] {len(centers)} active-fire patches, inputs from {day} "
          f"(predicting {day} + 1 day) ...")

    spread_results = []
    for i, row in centers.iterrows():
        lon, lat = float(row["longitude"]), float(row["latitude"])
        try:
            patch, valid_fraction = extract_patch(lon, lat, day, size_px=s, pixel_km=1.0)
        except Exception as e:
            print(f"  patch {i} skipped ({lon:.3f}, {lat:.3f}): {e}", file=sys.stderr)
            continue
        if valid_fraction < 0.8:
            print(f"  patch {i} skipped ({lon:.3f}, {lat:.3f}): only "
                  f"{valid_fraction:.0%} of it has weather data (coast / border)")
            continue

        x = np.broadcast_to(patch[None], (T, *patch.shape)).copy()    # (T, C, H, W)
        x = torch.from_numpy(x[None]).to(device)                      # (1, T, C, H, W)
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]          # (H, W)

        spread_results.append({
            "longitude": lon,
            "latitude":  lat,
            "burning_cells_nearby": int(row["burning_cells"]),
            "prev_fire_pixels": int((patch[prev_idx] == 1).sum()),
            "valid_fraction": float(valid_fraction),
            "spread_prob_max":  float(prob.max()),
            "spread_prob_mean": float(prob.mean()),
            "spread_pixels_above_50pct": int((prob >= 0.5).sum()),
            "spread_mask_shape": list(prob.shape),
        })

    return {
        "target_date": target_date,
        "input_day": day,
        "top_k": top_k,
        "patch_source": "ndws_sources_gee",
        "centred_on": f"active fires (FIRMS, last {CFG.label_quiet_days} days)",
        "patches": spread_results,
    }


def _load_meta() -> dict:
    if CFG.feature_meta_path().exists():
        return json.loads(CFG.feature_meta_path().read_text())
    return {}


def _resolve_meta_threshold() -> float:
    """The threshold actually applied to fire_alert (validation-tuned, from
    model metadata) — shown in the HTML so display matches behavior."""
    from .predict import resolve_threshold
    return resolve_threshold(_load_meta())


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
                    "erc":              float(r.get("erc", 0.0)),
                },
            }
            for _, r in scored.iterrows()
        ],
    }
    (out_dir / f"ignition_{target_date}.geojson").write_text(json.dumps(geojson))

    # Summary JSON (always lightweight)
    from .predict import probability_is_calibrated
    meta = _load_meta()
    target_month = datetime.strptime(target_date, "%Y-%m-%d").month
    summary = {
        "target_date": target_date,
        "data_window": {"start": data_window[0], "end": data_window[1]} if data_window else None,
        "forecast": (meta.get("label") or {}).get("definition"),
        "probability_calibrated": probability_is_calibrated(meta),
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
    calibrated = summary.get("probability_calibrated")
    p_label = "P(new fire, 7 d)" if calibrated else "Risk score (uncalibrated)"
    fmt_p = (lambda p: f"{p:.2%}") if calibrated else (lambda p: f"{p:.3f}")
    top = scored.sort_values("fire_probability", ascending=False).head(20)
    top_rows = "".join(
        f"<tr><td>{r['longitude']:.3f}</td><td>{r['latitude']:.3f}</td>"
        f"<td>{fmt_p(r['fire_probability'])}</td><td>{r.get('temp_c', 0):.1f}</td>"
        f"<td>{r.get('wind_speed_ms', 0):.1f}</td><td>{r.get('erc', 0):.0f}</td></tr>"
        for _, r in top.iterrows()
    )

    spread_section = ""
    if spread.get("patches"):
        spread_rows = "".join(
            f"<tr><td>{p['longitude']:.3f}</td><td>{p['latitude']:.3f}</td>"
            f"<td>{p['prev_fire_pixels']}</td><td>{p['spread_prob_max']:.3f}</td>"
            f"<td>{p['spread_pixels_above_50pct']}</td></tr>"
            for p in spread["patches"][:20]
        )
        spread_section = f"""
        <h2>Next-day spread on {len(spread['patches'])} active fires (U-Net, inputs from {spread.get('input_day')})</h2>
        <table>
          <thead><tr><th>Lon</th><th>Lat</th><th>Fire px (day t)</th><th>Max spread p</th><th>Fire pixels @50%</th></tr></thead>
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
  <div class="card"><div class="v">{fmt_p(summary["max_ignition_probability"])}</div><div class="l">Peak {p_label}</div></div>
  <div class="card"><div class="v">{len(spread.get("patches", []))}</div><div class="l">Spread predictions</div></div>
</div>

<h2>Top 20 highest-risk cells</h2>
<table>
  <thead><tr><th>Lon</th><th>Lat</th><th>{p_label}</th><th>Temp (°C)</th><th>Wind (m/s)</th><th>ERC</th></tr></thead>
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
<br>Spread predictions run the U-Net on NDWS-format patches around active fires —
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
