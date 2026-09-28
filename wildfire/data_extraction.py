"""
Google Earth Engine extraction pipeline for the West Coast wildfire model.

Pulls the feature stack:
  * ERA5-Land (temperature, dewpoint, wind components, root-zone soil moisture)
  * SRTM elevation + derived slope/aspect
  * NASA SMAP L4 surface soil moisture (v008)
  * GRIDMET Energy Release Component (NFDRS fuel dryness)
  * MODIS NDVI (Sentinel-2 if you swap MODIS/061 for COPERNICUS/S2_SR)
  * NASA FIRMS MODIS T21 thermal anomalies (forecast labels)

Features describe a 7-day window [day_start, day_end). The label looks
FORWARD: 1 when FIRMS detects fire in the week after day_end in a cell that had
no detection in the two weeks before it (see CFG.label_horizon_days /
label_quiet_days). Schema v1 took features and label from the same week, which
made the model a same-week detector rather than a forecast.

Schema v2 also replaced two sources that had been failing silently behind the
sentinel fallback:
  * UTOKYO/WTLAB/KBDI/v1 covers 60E-180E only. It never had a pixel on the
    West Coast, so every real row carried the fallback KBDI of 300 and the
    feature varied only on synthetic augmented rows.
  * NASA/SMAP/SPL4SMGP/007 is deprecated and ends 2025-06-27.
Every gap-filled band now carries a `<band>_filled` flag, and `check_coverage`
raises when a band is mostly fill, so an outage is an error instead of a
constant feature.

Authentication (one-time):
    earthengine authenticate
    # then in Python:
    import ee; ee.Initialize(project='your-gcp-project-id')

Runtime is dominated by the FeatureCollection.getInfo() call — expect ~10 min
per year over the full West Coast at the configured sampling density.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CFG


#: Bumped whenever an extracted table changes meaning. It is part of every
#: cache key, so windows cached under an older schema are never reused.
#: v2: forecast label, GRIDMET ERC for KBDI, SMAP L4 v008, fill flags.
EXTRACTION_VERSION = 2

#: Bands that can come back empty and are gap-filled so the band always
#: exists. Each gets a `<band>_filled` flag; the value maps the band to its
#: source for error messages.
FILLABLE_BANDS = {
    "ssm":  "NASA/SMAP/SPL4SMGP/008",
    "erc":  "IDAHO_EPSCOR/GRIDMET",
    "ndvi": "MODIS/061/MOD13Q1",
}

#: FIRMS lands in Earth Engine about a day after acquisition. A label window
#: that ended less than this long ago is not fully observed yet.
FIRMS_LATENCY_DAYS = 2


class FeatureOutage(RuntimeError):
    """A source returned no data for most cells — an outage, not a reading."""


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


def _d(day: str) -> datetime:
    return datetime.strptime(day, "%Y-%m-%d")


def _s(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _utc_today() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0,
                                              second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Label windows (pure date math, unit-tested)
# ---------------------------------------------------------------------------

def label_windows(day_start: str, day_end: str) -> dict:
    """Date ranges the forecast label reads, relative to the feature window.

    ``next``   -- the week being forecast: [day_end, day_end + horizon)
    ``recent`` -- detections that mean the cell is already burning:
                  [day_end - quiet, day_end)
    """
    end = _d(day_end)
    return {
        "next": (day_end, _s(end + timedelta(days=CFG.label_horizon_days))),
        "recent": (_s(end - timedelta(days=CFG.label_quiet_days)), day_end),
    }


def label_complete(day_end: str, as_of: Optional[datetime] = None) -> bool:
    """True once FIRMS has had time to observe the whole forecast week."""
    as_of = as_of or _utc_today()
    label_end = _d(label_windows(day_end, day_end)["next"][1])
    return label_end + timedelta(days=FIRMS_LATENCY_DAYS) <= as_of


def iter_windows(year: int, months: tuple, step_days: int = 7,
                 end_date: Optional[str] = None,
                 as_of: Optional[datetime] = None) -> Iterator[Tuple[str, str]]:
    """Full-length feature windows covering `months` of `year`.

    Windows run back to back through each contiguous run of months and are
    always `step_days` long -- a truncated 3-day window at a month boundary
    would average weather differently from the 7-day window scored at serving
    time. A window is only yielded once its forecast week has been observed
    (and, with `end_date`, only if that week ends by `end_date`).
    """
    months = sorted(set(months))
    runs, run = [], [months[0]]
    for m in months[1:]:
        if m == run[-1] + 1:
            run.append(m)
        else:
            runs.append(run)
            run = [m]
    runs.append(run)

    clip = _d(end_date) if end_date else None
    for run in runs:
        start = datetime(year, run[0], 1)
        stop = datetime(year + 1, 1, 1) if run[-1] == 12 else datetime(year, run[-1] + 1, 1)
        cursor = start
        while cursor + timedelta(days=step_days) <= stop:
            window_end = cursor + timedelta(days=step_days)
            day_start, day_end = _s(cursor), _s(window_end)
            label_end = _d(label_windows(day_start, day_end)["next"][1])
            if clip is not None and label_end > clip:
                return
            if not label_complete(day_end, as_of=as_of):
                return
            yield day_start, day_end
            cursor = window_end


# ---------------------------------------------------------------------------
# Coverage: fill flags -> loud failure
# ---------------------------------------------------------------------------

def fill_fractions(df: pd.DataFrame) -> dict:
    """Share of rows where each gap-filled band holds the fallback, not data."""
    return {band: float(df[f"{band}_filled"].mean())
            for band in FILLABLE_BANDS if f"{band}_filled" in df.columns and len(df)}


def check_coverage(df: pd.DataFrame, where: str,
                   max_fill: Optional[float] = None) -> None:
    """Raise FeatureOutage if any gap-filled band is mostly fallback values."""
    max_fill = CFG.max_fill_fraction if max_fill is None else max_fill
    bad = {b: f for b, f in fill_fractions(df).items() if f > max_fill}
    if bad:
        detail = "; ".join(f"{b} is fallback in {f:.0%} of cells (source {FILLABLE_BANDS[b]})"
                           for b, f in bad.items())
        raise FeatureOutage(
            f"{where}: {detail}. The collection returned no data for this window "
            "-- check its date range and footprint before trusting these rows."
        )


def drop_filled_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove rows where any feature is a fallback value. The model never
    trains on invented values, so it should not be scored on them either."""
    flags = [f"{b}_filled" for b in FILLABLE_BANDS if f"{b}_filled" in df.columns]
    if not flags:
        return df, 0
    keep = (df[flags] == 0).all(axis=1)
    return df[keep].copy(), int((~keep).sum())


# ---------------------------------------------------------------------------
# Feature stack
# ---------------------------------------------------------------------------

def _west_coast_geometry(ee_module):
    w, s, e, n = CFG.west_coast_bbox
    return ee_module.Geometry.Rectangle([w, s, e, n])


def _era5_daily(ee, day_start: str, day_end: str, geom):
    """ERA5-Land HOURLY collapsed to a mean over the date range."""
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


def _safe_mean(ee, coll, src_band: str, dst_band: str, fallback: float = 0.0,
               scale: float = 1.0):
    """
    Mean of a collection that may be empty for some windows or pixels.

    Returns two bands: `dst_band` (the mean times `scale`, with `fallback`
    where there was no data) and `<dst_band>_filled` (1 where the fallback was
    used). The fallback only keeps the band present so downstream band maths
    doesn't crash on a 0-band image; the flag keeps it from being mistaken for
    a measurement.

    The sentinel MUST use src_band (the collection's native band name) so GEE
    sees a homogeneous collection. It is fully masked, so it never contributes
    to the mean.
    """
    sentinel = (ee.Image(0.0)
                  .float()
                  .rename(src_band)
                  .updateMask(ee.Image(0)))   # fully masked — no pixel contribution
    mean = ee.ImageCollection([sentinel]).merge(coll).mean()
    filled = mean.mask().unmask(0).eq(0).rename(f"{dst_band}_filled").float()
    value = (mean.multiply(scale)
                 .unmask(float(fallback))
                 .rename(dst_band)
                 .float())
    return value.addBands(filled)


def _smap_surface(ee, day_start, day_end, geom):
    ext_start = _s(_d(day_start) - timedelta(days=14))
    coll = (ee.ImageCollection(FILLABLE_BANDS["ssm"])
              .filterDate(ext_start, day_end)
              .filterBounds(geom)
              .select("sm_surface"))
    return _safe_mean(ee, coll, "sm_surface", "ssm", fallback=0.2)


def _erc(ee, day_start, day_end, geom):
    """GRIDMET Energy Release Component, mean over the feature window.

    ERC is the NFDRS fuel-dryness index fire agencies use for seasonal
    severity -- daily, ~4 km, CONUS. It replaces UTOKYO KBDI, which never
    covered this region. The bbox's slivers of Canada and Mexico fall outside
    GRIDMET and come back flagged as filled.
    """
    coll = (ee.ImageCollection(FILLABLE_BANDS["erc"])
              .filterDate(day_start, day_end)
              .filterBounds(geom)
              .select("erc"))
    return _safe_mean(ee, coll, "erc", "erc", fallback=0.0)


def _ndvi(ee, day_start, day_end, geom):
    """MODIS NDVI 16-day composites. 32-day lookback always captures one composite."""
    ext_start = _s(_d(day_start) - timedelta(days=32))
    coll = (ee.ImageCollection(FILLABLE_BANDS["ndvi"])
              .filterDate(ext_start, day_end)
              .filterBounds(geom)
              .select("NDVI"))
    # MODIS NDVI is scaled ×10000; fallback 0.3 = shrubland
    return _safe_mean(ee, coll, "NDVI", "ndvi", fallback=0.3, scale=0.0001)


def _firms_any(ee, start: str, end: str, geom, name: str):
    """1 where FIRMS has at least one T21 thermal anomaly in [start, end).
    Uses the same sentinel pattern so the band always exists."""
    coll = (ee.ImageCollection("FIRMS")
              .filterDate(start, end)
              .filterBounds(geom)
              .select("T21"))
    # Sentinel must use the same band name as FIRMS (T21) for homogeneous merge
    sentinel = (ee.Image(0.0).float()
                  .rename("T21")
                  .updateMask(ee.Image(0)))
    return (ee.ImageCollection([sentinel])
              .merge(coll)
              .count()                        # count of valid (unmasked) T21 observations
              .gt(0)                          # 1 if ≥1 thermal anomaly, else 0
              .unmask(0)
              .rename(name)
              .float())


def _fire_bands(ee, day_start: str, day_end: str, geom, labelled: bool):
    """`fire_recent` (already burning) and, if labelled, the forecast label."""
    windows = label_windows(day_start, day_end)
    recent = _firms_any(ee, *windows["recent"], geom, "fire_recent")
    if not labelled:
        return recent
    upcoming = _firms_any(ee, *windows["next"], geom, "fire_next")
    label = upcoming.And(recent.Not()).rename(CFG.target_column).float()
    return recent.addBands(label)


def _stack_for_window(ee, day_start: str, day_end: str, geom=None,
                      labelled: bool = True):
    """Features over [day_start, day_end), plus fire bands.

    `labelled=False` is for serving: the forecast week is still in the future,
    so a label band would read as "no fire" rather than "unknown".
    """
    if geom is None:
        geom = _west_coast_geometry(ee)
    stack = (
        _era5_daily(ee, day_start, day_end, geom)
            .addBands(_srtm_topography(ee, geom))
            .addBands(_smap_surface(ee, day_start, day_end, geom))
            .addBands(_erc(ee, day_start, day_end, geom))
            .addBands(_ndvi(ee, day_start, day_end, geom))
            .addBands(_fire_bands(ee, day_start, day_end, geom, labelled))
    )
    return stack, geom


# ---------------------------------------------------------------------------
# Spread-model patches, built from the sources NDWS was built from
# ---------------------------------------------------------------------------

#: The Next Day Wildfire Spread dataset (Huot et al. 2022) was exported from
#: these collections. The spread U-Net was trained on NDWS, so its serving
#: patches must come from the same place with the same units -- the previous
#: builder fed ERA5 / KBDI / SMAP / MODIS NDVI into channels trained on
#: GRIDMET / PDSI / specific humidity / VIIRS NDVI.
NDWS_SOURCES = {
    "weather": "IDAHO_EPSCOR/GRIDMET",       # tmmx, tmmn, vs, th, sph (daily)
    "drought": "GRIDMET/DROUGHT",            # pdsi (every 5 days)
    "ndvi":    "NASA/VIIRS/002/VNP13A1",     # NDVI, raw x1e4 units like NDWS
    "fire":    "MODIS/061/MOD14A1",          # FireMask (daily)
}
_NDWS_LOOKBACK_DAYS = {"drought": 30, "ndvi": 48}


def _last_image_day(ee, collection_id: str, start: str, end: str, geom) -> Optional[str]:
    ms = (ee.ImageCollection(collection_id)
            .filterDate(start, end)
            .filterBounds(geom)
            .aggregate_max("system:time_start")
            .getInfo())
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def ndws_input_day(target_date: str, lookback_days: int = 21) -> str:
    """Latest day <= target_date with both GRIDMET weather and a MOD14A1 fire
    mask, so every channel of a spread patch describes the same day.

    Also confirms the slower products (PDSI, VIIRS NDVI) have an image inside
    their lookback, so an outage is a FeatureOutage here rather than a
    cryptic null-image error on every patch.
    """
    ee = _require_ee()
    geom = _west_coast_geometry(ee)
    end = _s(_d(target_date) + timedelta(days=1))
    start = _s(_d(target_date) - timedelta(days=lookback_days))

    latest = {k: _last_image_day(ee, NDWS_SOURCES[k], start, end, geom)
              for k in ("weather", "fire")}
    missing = [NDWS_SOURCES[k] for k, v in latest.items() if v is None]
    if missing:
        raise FeatureOutage(f"No images in {missing} between {start} and {target_date}.")
    day = min(latest.values())

    for key, days in _NDWS_LOOKBACK_DAYS.items():
        since = _s(_d(day) - timedelta(days=days))
        if _last_image_day(ee, NDWS_SOURCES[key], since, _s(_d(day) + timedelta(days=1)), geom) is None:
            raise FeatureOutage(f"No {NDWS_SOURCES[key]} image in the {days} days before {day}.")
    return day


def _latest_image(ee, collection_id: str, day: str, lookback_days: int, bands: list):
    start = _s(_d(day) - timedelta(days=lookback_days))
    end = _s(_d(day) + timedelta(days=1))
    return ee.Image(ee.ImageCollection(collection_id)
                      .filterDate(start, end)
                      .select(bands)
                      .sort("system:time_start", False)
                      .first())


def _extract_patch_uncached(lon: float, lat: float, day: str,
                            size_px: int, pixel_km: float) -> np.ndarray:
    from .spread.config import SPREAD_CFG

    ee = _require_ee()

    half_km   = (size_px * pixel_km) / 2.0
    half_deg  = half_km / 111.0   # ~111 km / degree latitude
    half_lon  = half_deg / max(0.2, np.cos(np.deg2rad(lat)))
    bbox = ee.Geometry.Rectangle(
        [lon - half_lon, lat - half_deg, lon + half_lon, lat + half_deg]
    )

    met = _latest_image(ee, NDWS_SOURCES["weather"], day, 0,
                        ["tmmx", "tmmn", "vs", "th", "sph"])
    pdsi = _latest_image(ee, NDWS_SOURCES["drought"], day,
                         _NDWS_LOOKBACK_DAYS["drought"], ["pdsi"])
    ndvi = _latest_image(ee, NDWS_SOURCES["ndvi"], day,
                         _NDWS_LOOKBACK_DAYS["ndvi"], ["NDVI"])
    fire_mask = _latest_image(ee, NDWS_SOURCES["fire"], day, 0, ["FireMask"]).unmask(0)

    # MOD14A1 FireMask classes -> NDWS {-1, 0, 1}: 7-9 fire, 3/5 non-fire
    # water/land, everything else (cloud, unknown, not processed) uncertain.
    prev_fire = (ee.Image(-1.0).toFloat()
                   .where(fire_mask.eq(3).Or(fire_mask.eq(5)), 0)
                   .where(fire_mask.gte(7), 1))
    zero = ee.Image(0.0).toFloat()
    wind_from_rad = met.select("th").multiply(math.pi / 180.0)

    channels = {
        "elevation":    ee.Image("USGS/SRTMGL1_003"),
        # NDWS carries no slope/aspect, so the trained model has only ever
        # seen zeros in these two channels.
        "slope_deg":    zero,
        "aspect_deg":   zero,
        "NDVI":         ndvi.select("NDVI"),
        "tmmx_c":       met.select("tmmx").subtract(273.15),
        "tmmn_c":       met.select("tmmn").subtract(273.15),
        "vs":           met.select("vs"),
        "vs_cos_th":    met.select("vs").multiply(wind_from_rad.cos()),
        "pdsi":         pdsi.select("pdsi"),
        "sph":          met.select("sph"),
        "PrevFireMask": prev_fire,
    }
    names = list(SPREAD_CFG.feature_channels)
    valid = met.select("tmmx").mask().gt(0).rename("_valid")
    patch_stack = ee.Image.cat(
        [channels[n].toFloat().rename(n) for n in names] + [valid.toFloat()]
    )

    # Sample on a regular grid using sampleRectangle — returns nested lists
    # per band that we reshape to (H, W).
    scale_m = int(pixel_km * 1000)
    rect = patch_stack.reproject(crs="EPSG:4326", scale=scale_m).sampleRectangle(
        region=bbox, defaultValue=0
    )
    props = rect.getInfo()["properties"]

    arrs = []
    for ch in names + ["_valid"]:
        a = np.asarray(props.get(ch, [[0.0]]), dtype=np.float32)
        if a.ndim != 2:
            a = np.zeros((size_px, size_px), dtype=np.float32)
        # Resize/pad/crop to exactly size_px × size_px
        h, w = a.shape
        out = np.zeros((size_px, size_px), dtype=np.float32)
        h_use = min(h, size_px); w_use = min(w, size_px)
        out[:h_use, :w_use] = a[:h_use, :w_use]
        arrs.append(out)
    return np.stack(arrs, axis=0)   # (C + 1, H, W); last plane is validity


def extract_patch(lon: float, lat: float, day: str,
                  size_px: int = 64, pixel_km: float = 1.0,
                  use_cache: bool = True) -> Tuple[np.ndarray, float]:
    """
    Return an NDWS-format (C, size_px, size_px) patch for day `day` centred on
    (lon, lat), and the fraction of it with weather data (the rest is ocean
    or outside CONUS and holds zeros). Cached on disk by
    (lon, lat, day, size, pixel_km, schema) — repeated dashboard runs over the
    same fires hit zero GEE quota.
    """
    from .cache import get_or_compute_npy, key_patch
    from .spread.config import SPREAD_CFG

    key = key_patch(lon, lat, day, day, size_px, pixel_km, version=EXTRACTION_VERSION)
    stacked = get_or_compute_npy(
        key,
        lambda: _extract_patch_uncached(lon, lat, day, size_px, pixel_km),
        use_cache=use_cache,
    )
    patch, valid = stacked[:-1].copy(), stacked[-1]

    # NDWS stores VIIRS NDVI as raw integers (x1e4). Guard against the
    # collection being served with its scale factor already applied.
    ndvi_idx = list(SPREAD_CFG.feature_channels).index("NDVI")
    peak = float(np.abs(patch[ndvi_idx]).max())
    if 0.0 < peak <= 1.5:
        patch[ndvi_idx] *= 10000.0
    return patch, float((valid > 0).mean())


# ---------------------------------------------------------------------------
# Ignition sampling
# ---------------------------------------------------------------------------

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
                             n_pixels: int, n_pages: int,
                             labelled: bool) -> pd.DataFrame:
    ee = _require_ee()
    stack, geom = _stack_for_window(ee, day_start, day_end, labelled=labelled)

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
                   use_cache: bool = True,
                   labelled: bool = True) -> pd.DataFrame:
    """
    Sample the GEE feature stack across the West Coast for one date window.
    Paged: runs n_pages independent samples with different seeds and concats.
    Cached on disk by (day_start, day_end, scale, pages, pixels_per_page,
    schema, labelled) — repeated calls hit disk instead of GEE.

    `labelled=True` (training) refuses windows whose forecast week FIRMS has
    not finished observing; serving pulls pass `labelled=False`. Raises
    FeatureOutage when a source is mostly fallback for the window.
    """
    from .cache import get_or_compute_parquet, key_window

    if labelled and not label_complete(day_end):
        raise ValueError(
            f"The forecast week after {day_end} has not been fully observed by "
            "FIRMS yet; a label pulled now would read unobserved days as 'no "
            "fire'. Use labelled=False for serving."
        )

    per_call = n_pixels or CFG.gee_num_pixels
    pages    = n_pages  or CFG.gee_pages_per_window
    key = key_window(day_start, day_end, CFG.cell_scale_m, pages, per_call,
                     version=EXTRACTION_VERSION, labelled=labelled)

    df = get_or_compute_parquet(
        key,
        lambda: _extract_window_uncached(day_start, day_end, per_call, pages, labelled),
        use_cache=use_cache,
    )
    if not df.empty:
        check_coverage(df, f"window {day_start}..{day_end}")
    return df


def extract_year(year: int, output_path: Optional[Path] = None,
                 step_days: int = 7,
                 months: Optional[tuple] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Walk the given months of a year in full `step_days` windows (see
    iter_windows) and concatenate. Defaults to the Aug-Oct fire season; pass
    `months` for shoulder-season pulls (e.g. (5, 6, 7)). `end_date`
    (YYYY-MM-DD) additionally requires each window's forecast week to end by
    that date; windows whose forecast week FIRMS hasn't observed yet are
    always skipped.
    """
    init_ee()

    months = tuple(months) if months else CFG.fire_season_months

    pieces = []
    for day_start, day_end in iter_windows(year, months, step_days, end_date=end_date):
        print(f"  GEE sample {day_start} .. {day_end} "
              f"(label: fires {label_windows(day_start, day_end)['next'][0]} "
              f".. {label_windows(day_start, day_end)['next'][1]})", flush=True)
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

    if not pieces:
        raise RuntimeError(f"No data extracted for year {year} — check GEE auth, quotas, "
                           "and the '! window failed' lines above.")

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
                        help="Only keep windows whose forecast week ends by this date (YYYY-MM-DD).")
    parser.add_argument("--out",  type=Path, default=None,
                        help="Output parquet path; defaults to wildfire_data/raw_<year>_v<schema>.parquet "
                             "(with the month list when --months is given).")
    args = parser.parse_args()
    months = tuple(int(m) for m in args.months.split(",")) if args.months else None
    default_name = (f"raw_{args.year}_v{EXTRACTION_VERSION}.parquet" if months is None
                    else f"raw_{args.year}_m{'-'.join(str(m) for m in months)}"
                         f"_v{EXTRACTION_VERSION}.parquet")
    out = args.out or (CFG.data_dir / default_name)
    extract_year(args.year, output_path=out, months=months, end_date=args.end)
