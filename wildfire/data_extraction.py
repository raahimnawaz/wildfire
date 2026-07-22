"""
Google Earth Engine extraction pipeline for the West Coast wildfire model.

Pulls the paper-recommended feature stack:
  * ERA5-Land (temperature, dewpoint, wind components, root-zone soil moisture)
  * SRTM elevation + derived slope/aspect
  * NASA SMAP surface soil moisture
  * UTOKYO KBDI drought index
  * MODIS NDVI (Sentinel-2 if you swap MODIS/006 for COPERNICUS/S2_SR)
  * NASA FIRMS MODIS T21 thermal anomalies (binary fire labels)

Authentication (one-time):
    earthengine authenticate
    # then in Python:
    import ee; ee.Initialize(project='your-gcp-project-id')

Runtime is dominated by the FeatureCollection.getInfo() call — expect ~10 min
per year over the full West Coast at the configured sampling density.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CFG


def _require_ee():
    try:
        import ee  # noqa: F401
        return ee
    except ImportError:
        raise ImportError(
            "earthengine-api is not installed. Install with:\n"
            "    pip install earthengine-api\n"
            "Then authenticate with:\n"
            "    earthengine authenticate"
        )


def init_ee(project: Optional[str] = None) -> None:
    """
    Initialize Earth Engine.

    Auth resolution order:
      1. GEE_SA_JSON env var holding a service-account key (JSON string or a
         path to one) — the non-interactive path used by CI.
      2. Cached interactive OAuth credentials from `earthengine authenticate`.
    Pass your GCP project id or set the GEE_PROJECT env var.
    """
    import os
    ee = _require_ee()
    project = project or os.environ.get("GEE_PROJECT")

    sa_json = os.environ.get("GEE_SA_JSON")
    if sa_json:
        import json as _json
        import tempfile
        if os.path.exists(sa_json):
            key_path = sa_json
            email = _json.loads(open(sa_json).read()).get("client_email")
        else:
            tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            tmp.write(sa_json)
            tmp.close()
            key_path = tmp.name
            email = _json.loads(sa_json).get("client_email")
        creds = ee.ServiceAccountCredentials(email, key_path)
        ee.Initialize(creds, project=project)
        return

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def _west_coast_geometry(ee_module):
    w, s, e, n = CFG.west_coast_bbox
    return ee_module.Geometry.Rectangle([w, s, e, n])


def _era5_daily(ee, day_start: str, day_end: str, geom):
    """ERA5-Land HOURLY collapsed to daily means/maxes for the date range."""
    coll = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
              .filterDate(day_start, day_end)
              .filterBounds(geom)
              .select([
                  "temperature_2m",
                  "dewpoint_temperature_2m",
                  "u_component_of_wind_10m",
                  "v_component_of_wind_10m",
                  "volumetric_soil_water_layer_3",
              ]))
    return coll.mean()


def _srtm_topography(ee, geom):
    dem   = ee.Image("USGS/SRTMGL1_003").clip(geom).rename("elevation_m")
    terr  = ee.Terrain.products(dem.rename("elevation"))
    slope = terr.select("slope").rename("slope_deg")
    aspect = terr.select("aspect").rename("aspect_deg")
    return dem.addBands([slope, aspect])


def _safe_mean(ee, coll, src_band: str, dst_band: str, fallback: float = 0.0):
    """
    Mean of a collection that may be empty for some windows.

    The sentinel image MUST use src_band (the collection's native band name)
    so GEE sees a homogeneous collection. We rename to dst_band after mean().
    The sentinel is fully masked so it never contributes to the mean value.
    """
    sentinel = (ee.Image(float(fallback))
                  .float()
                  .rename(src_band)
                  .updateMask(ee.Image(0)))   # fully masked — no pixel contribution
    return (ee.ImageCollection([sentinel])
              .merge(coll)
              .mean()
              .unmask(float(fallback))
              .rename(dst_band)
              .float())


def _smap_surface(ee, day_start, day_end, geom):
    from datetime import datetime, timedelta
    ext_start = (datetime.strptime(day_start, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
    coll = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
              .filterDate(ext_start, day_end)
              .filterBounds(geom)
              .select("sm_surface"))
    return _safe_mean(ee, coll, "sm_surface", "ssm", fallback=0.2)


def _kbdi(ee, day_start, day_end, geom):
    from datetime import datetime, timedelta
    ext_start = (datetime.strptime(day_start, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    coll = (ee.ImageCollection("UTOKYO/WTLAB/KBDI/v1")
              .filterDate(ext_start, day_end)
              .filterBounds(geom)
              .select("KBDI"))
    return _safe_mean(ee, coll, "KBDI", "kbdi_index", fallback=300.0)


def _ndvi(ee, day_start, day_end, geom):
    """MODIS NDVI 16-day composites. 32-day lookback always captures one composite."""
    from datetime import datetime, timedelta
    ext_start = (datetime.strptime(day_start, "%Y-%m-%d") - timedelta(days=32)).strftime("%Y-%m-%d")
    coll = (ee.ImageCollection("MODIS/061/MOD13Q1")
              .filterDate(ext_start, day_end)
              .filterBounds(geom)
              .select("NDVI"))
    # MODIS NDVI is scaled ×10000; fallback 3000 = NDVI 0.3 (shrubland)
    return (_safe_mean(ee, coll, "NDVI", "ndvi", fallback=3000.0)
              .multiply(0.0001)
              .float())


def _firms_label(ee, day_start, day_end, geom):
    """Binary label: 1 if any FIRMS T21 detection in the window, else 0.
    Uses the same sentinel pattern so the band always exists."""
    coll = (ee.ImageCollection("FIRMS")
              .filterDate(day_start, day_end)
              .filterBounds(geom)
              .select("T21"))
    # Sentinel must use the same band name as FIRMS (T21) for homogeneous merge
    sentinel = (ee.Image(0.0).float()
                  .rename("T21")
                  .updateMask(ee.Image(0)))
    fire = (ee.ImageCollection([sentinel])
              .merge(coll)
              .count()                        # count of valid (unmasked) T21 observations
              .gt(0)                          # 1 if ≥1 thermal anomaly, else 0
              .unmask(0)
              .rename(CFG.target_column)
              .float())
    return fire


def _stack_for_window(ee, day_start: str, day_end: str, geom=None):
    if geom is None:
        geom = _west_coast_geometry(ee)
    stack = (
        _era5_daily(ee, day_start, day_end, geom)
            .addBands(_srtm_topography(ee, geom))
            .addBands(_smap_surface(ee, day_start, day_end, geom))
            .addBands(_kbdi(ee, day_start, day_end, geom))
            .addBands(_ndvi(ee, day_start, day_end, geom))
            .addBands(_firms_label(ee, day_start, day_end, geom))
    )
    return stack, geom


def _extract_patch_uncached(lon: float, lat: float, day_start: str, day_end: str,
                            size_px: int, pixel_km: float) -> "np.ndarray":
    import numpy as np
    ee = _require_ee()

    half_km   = (size_px * pixel_km) / 2.0
    half_deg  = half_km / 111.0   # ~111 km / degree latitude
    half_lon  = half_deg / max(0.2, np.cos(np.deg2rad(lat)))
    bbox = ee.Geometry.Rectangle(
        [lon - half_lon, lat - half_deg, lon + half_lon, lat + half_deg]
    )

    # Build the same feature stack used by training, but constrained to this bbox
    full_stack, _ = _stack_for_window(ee, day_start, day_end, geom=bbox)

    # Compute wind components separately so the spread schema lines up
    era5_mean = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                   .filterDate(day_start, day_end)
                   .filterBounds(bbox)
                   .select(["u_component_of_wind_10m", "v_component_of_wind_10m",
                            "temperature_2m", "dewpoint_temperature_2m"])
                   .mean())
    wind_u = era5_mean.select("u_component_of_wind_10m").rename("wind_u_ms")
    wind_v = era5_mean.select("v_component_of_wind_10m").rename("wind_v_ms")
    temp_c = era5_mean.select("temperature_2m").subtract(273.15).rename("temp_c")
    dewp_c = era5_mean.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_c")

    # Re-grab topography (separately so we can rename for the spread schema)
    dem    = ee.Image("USGS/SRTMGL1_003").clip(bbox).rename("elevation_m").toFloat()
    terr   = ee.Terrain.products(dem.rename("elevation"))
    slope  = terr.select("slope").rename("slope_deg").toFloat()
    aspect = terr.select("aspect").rename("aspect_deg").toFloat()

    ndvi   = _ndvi(ee, day_start, day_end, bbox).toFloat()
    kbdi   = _kbdi(ee, day_start, day_end, bbox).rename("kbdi_index").toFloat()
    smap   = _smap_surface(ee, day_start, day_end, bbox).rename("soil_moist").toFloat()
    fire   = _firms_label(ee, day_start, day_end, bbox).rename("prev_fire_mask").toFloat()

    patch_stack = (dem
                   .addBands(slope).addBands(aspect)
                   .addBands(ndvi).addBands(temp_c).addBands(dewp_c)
                   .addBands(wind_u).addBands(wind_v)
                   .addBands(kbdi).addBands(smap).addBands(fire))

    # Sample on a regular grid using sampleRectangle — returns nested lists
    # per band that we reshape to (H, W).
    scale_m = int(pixel_km * 1000)
    rect = patch_stack.reproject(crs="EPSG:4326", scale=scale_m).sampleRectangle(
        region=bbox, defaultValue=0
    )
    props = rect.getInfo()["properties"]

    channel_order = ["elevation_m", "slope_deg", "aspect_deg", "ndvi",
                     "temp_c", "dewpoint_c", "wind_u_ms", "wind_v_ms",
                     "kbdi_index", "soil_moist", "prev_fire_mask"]
    arrs = []
    for ch in channel_order:
        a = np.asarray(props.get(ch, [[0.0]]), dtype=np.float32)
        if a.ndim != 2:
            a = np.zeros((size_px, size_px), dtype=np.float32)
        # Resize/pad/crop to exactly size_px × size_px
        h, w = a.shape
        out = np.zeros((size_px, size_px), dtype=np.float32)
        h_use = min(h, size_px); w_use = min(w, size_px)
        out[:h_use, :w_use] = a[:h_use, :w_use]
        arrs.append(out)
    return np.stack(arrs, axis=0)   # (C, H, W)


def extract_patch(lon: float, lat: float, day_start: str, day_end: str,
                  size_px: int = 64, pixel_km: float = 1.0,
                  use_cache: bool = True) -> "np.ndarray":
    """
    Return a real NDWS-format (C, size_px, size_px) imagery patch from GEE
    centered on (lon, lat). Cached on disk by (lon, lat, dates, size, pixel_km)
    — repeated dashboard runs over the same hotspots hit zero GEE quota.
    """
    from .cache import get_or_compute_npy, key_patch

    key = key_patch(lon, lat, day_start, day_end, size_px, pixel_km)
    return get_or_compute_npy(
        key,
        lambda: _extract_patch_uncached(lon, lat, day_start, day_end, size_px, pixel_km),
        use_cache=use_cache,
    )


def _sample_once(ee, stack, geom, n_pixels: int, seed: int) -> pd.DataFrame:
    """Single sample call — stays under GEE's 5000-feature getInfo() limit."""
    sampled = stack.sample(
        region=geom,
        scale=CFG.cell_scale_m,
        numPixels=n_pixels,
        seed=seed,
        tileScale=CFG.gee_tile_scale,
        geometries=True,
        dropNulls=True,
    )
    feats = sampled.getInfo()["features"]
    if not feats:
        return pd.DataFrame()
    rows = []
    for f in feats:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        rows.append({**props, "longitude": coords[0], "latitude": coords[1]})
    return pd.DataFrame(rows)


def _extract_window_uncached(day_start: str, day_end: str,
                             n_pixels: int, n_pages: int) -> pd.DataFrame:
    ee = _require_ee()
    stack, geom = _stack_for_window(ee, day_start, day_end)

    chunks = []
    for page_idx in range(n_pages):
        try:
            chunk = _sample_once(ee, stack, geom, n_pixels, seed=CFG.gee_seed + page_idx)
            if not chunk.empty:
                chunks.append(chunk)
        except Exception as e:
            print(f"    page {page_idx} skipped: {e}", file=sys.stderr)
            continue

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    if "longitude" in df.columns and "latitude" in df.columns:
        df = df.drop_duplicates(subset=["longitude", "latitude"], keep="first").reset_index(drop=True)
    df["date"] = pd.Timestamp(day_start) + (pd.Timestamp(day_end) - pd.Timestamp(day_start)) / 2
    return df


def extract_window(day_start: str, day_end: str,
                   n_pixels: Optional[int] = None,
                   n_pages: Optional[int] = None,
                   use_cache: bool = True) -> pd.DataFrame:
    """
    Sample the GEE feature stack across the West Coast for one date window.
    Paged: runs n_pages independent samples with different seeds and concats.
    Cached on disk by (day_start, day_end, scale, pages, pixels_per_page) —
    repeated calls hit disk instead of GEE.
    """
    from .cache import get_or_compute_parquet, key_window

    per_call = n_pixels or CFG.gee_num_pixels
    pages    = n_pages  or CFG.gee_pages_per_window
    key = key_window(day_start, day_end, CFG.cell_scale_m, pages, per_call)

    return get_or_compute_parquet(
        key,
        lambda: _extract_window_uncached(day_start, day_end, per_call, pages),
        use_cache=use_cache,
    )


def extract_year(year: int, output_path: Optional[Path] = None,
                 step_days: int = 7,
                 months: Optional[tuple] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Walk the given months of a year in `step_days` windows and concatenate.
    Defaults to the Aug-Oct fire season; pass `months` for shoulder-season
    pulls (e.g. (5, 6, 7)). `end_date` (YYYY-MM-DD) clips the walk — needed
    when extracting the current year, where windows past ~today would be
    empty. Use `step_days` to control window size (smaller windows = more
    samples but higher GEE quota usage).
    """
    init_ee()

    months = tuple(months) if months else CFG.fire_season_months
    clip = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    pieces = []
    for month in months:
        start = datetime(year, month, 1)
        if month == 12:
            stop = datetime(year + 1, 1, 1)
        else:
            stop = datetime(year, month + 1, 1)
        if clip is not None:
            stop = min(stop, clip)

        cursor = start
        while cursor < stop:
            window_end = min(cursor + timedelta(days=step_days), stop)
            day_start = cursor.strftime("%Y-%m-%d")
            day_end   = window_end.strftime("%Y-%m-%d")
            print(f"  GEE sample {day_start} .. {day_end}", flush=True)
            try:
                chunk = extract_window(day_start, day_end)
            except Exception as e:
                print(f"  ! window failed ({e}); skipping", file=sys.stderr)
                chunk = pd.DataFrame()
            if not chunk.empty:
                # Cache-hit pieces round-trip through parquet as
                # datetime64[us] while fresh pieces are datetime64[ns];
                # normalize so a resumed (mixed) run can concat.
                if "date" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["date"]).astype("datetime64[ns]")
                pieces.append(chunk)
            cursor = window_end

    if not pieces:
        raise RuntimeError(f"No data extracted for year {year} — check GEE auth & quotas.")

    df = pd.concat(pieces, ignore_index=True)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"  wrote {len(df):,} rows -> {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract GEE features for part of a year.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", type=str, default=None,
                        help="Comma-separated month numbers (default: fire season 8,9,10).")
    parser.add_argument("--end", type=str, default=None,
                        help="Clip extraction at this date (YYYY-MM-DD) — for the current year.")
    parser.add_argument("--out",  type=Path, default=None,
                        help="Output parquet path; defaults to wildfire_data/raw_<year>.parquet "
                             "(suffixed with the month list when --months is given).")
    args = parser.parse_args()
    months = tuple(int(m) for m in args.months.split(",")) if args.months else None
    default_name = (f"raw_{args.year}.parquet" if months is None
                    else f"raw_{args.year}_m{'-'.join(str(m) for m in months)}.parquet")
    out = args.out or (CFG.data_dir / default_name)
    extract_year(args.year, output_path=out, months=months, end_date=args.end)
