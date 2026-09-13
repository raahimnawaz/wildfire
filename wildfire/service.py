"""Interactive per-click ignition risk, served from the shipped XGBoost model.

Why this exists
---------------
``wildfire.dashboard`` answers "where is the risk today?" for the whole West
Coast, once a day, as a static page. It cannot answer "what about *here*?" for an
arbitrary point and date, which is the one question a person looking at a map
actually asks. This serves that, and it does it by calling the same extraction
and scoring path the dashboard uses rather than reimplementing either.

What it deliberately does not do
--------------------------------
An earlier prototype of this service (``backend/main.py``, never committed) got
three things wrong, and the shapes of those mistakes are worth recording so they
do not come back:

1. **It reported a hardcoded ``f1_score = 0.89`` and ``false_negative_rate =
   0.08``**, commented as "placeholder reliability figures", and the UI rendered
   them as if measured. Every number of that kind already exists, measured, in
   ``wildfire_models/feature_meta.json`` — so this module reads them from the
   artifact and has no literal metric anywhere in it. If the artifact does not
   carry a metric, the response omits the field rather than inventing one.

2. **It min-max normalised each prediction to [0, 1]** before returning it. That
   makes every response look the same: the hottest cell in a uniformly safe
   patch renders identically to the hottest cell in a patch that is about to
   burn, and a "max risk" readout is then always exactly 100%. XGBoost already
   emits a calibrated probability, so this returns it untouched. The only number
   that makes a cell "an alert" is the validation-tuned threshold below.

3. **If the weights file was missing it created a dummy random-weights model,
   saved it, and served predictions from it**, printing "Successfully loaded
   model weights" either way. There is no fallback model here. A missing
   artifact is a 503 that names the command to fix it.

And one more, inherited from the same prototype: when Earth Engine was
unavailable it silently substituted synthetic data. Synthetic features are the
right thing for tests and the wrong thing for a risk map, so ``/predict``
returns 503 when GEE is unavailable rather than a plausible-looking lie. The
sampler is injectable so tests can supply their own without touching GEE.

No uncertainty or saliency field
--------------------------------
The prototype served MC-dropout variance and a gradient saliency map. Both came
from a ConvLSTM that has no evaluation recorded anywhere in this repo. Dropping
them is not a downgrade: a boosted tree has no dropout to sample, so any
"uncertainty" here would have been invented. What replaces saliency is the
model's real gain-based feature importance, which is measured and which the
README already quotes (KBDI and wind dominate).
"""

# NOTE: deliberately no `from __future__ import annotations` here. It turns
# annotations into strings, and FastAPI resolves a route's hints against the
# module globals -- where `PredictRequest`, defined inside create_app(), does not
# exist. It does not raise: it silently decides `req` is a query parameter, and
# every POST /predict comes back 422 "field required: query.req". Keeping
# annotations eager costs nothing on 3.9+ and keeps fastapi an optional import.

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import CFG
from .predict import resolve_threshold, score_dataframe

#: Sampling resolution. Matches ``CFG.cell_scale_m`` on purpose — the model was
#: trained on 2 km cells, and scoring it on a finer grid would feed it feature
#: statistics it never saw (a 200 m KBDI cell is not a 2 km KBDI cell).
DEFAULT_PIXEL_KM = CFG.cell_scale_m / 1000.0

#: Grid edge in cells. 32 x 32 at 2 km is a ~64 km box, which is a sensible
#: "region around this click" and stays far under sampleRectangle's per-band
#: pixel ceiling.
DEFAULT_GRID = 32

#: ERA5-Land lags realtime by ~5 days, so a window ending today can come back
#: empty. Walk it back in 2-day steps, exactly as ``dashboard._ignition_scores``
#: does, and report the window actually used.
WINDOW_SHIFTS = (0, 2, 4, 6, 8)

#: Raw band names ``_stack_for_window`` produces that ``features.derive_all``
#: needs. Asserted on the sampled frame so a band rename upstream fails here
#: with a readable message instead of inside the model call.
REQUIRED_BANDS = (
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "volumetric_soil_water_layer_3",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "ssm",
    "kbdi_index",
    "ndvi",
)


class ModelUnavailable(RuntimeError):
    """The trained artifact is missing. Actionable, not a 500."""


class DataUnavailable(RuntimeError):
    """Earth Engine returned nothing usable for any window near the date."""


@lru_cache(maxsize=1)
def _meta() -> dict:
    path = CFG.feature_meta_path()
    if not path.exists():
        raise ModelUnavailable(
            f"No feature metadata at {path}. Train a model first: "
            "python -m wildfire.train --synthetic"
        )
    return json.loads(path.read_text())


def model_card() -> dict:
    """Everything the UI should say about model quality, read from the artifact.

    Fields are omitted rather than defaulted when the artifact does not carry
    them — a missing metric must not be indistinguishable from a real one.
    """
    meta = _meta()
    metrics = meta.get("metrics") or {}
    val = metrics.get("val") or {}
    test = metrics.get("test") or {}
    data = meta.get("data") or {}

    card: dict = {
        "threshold": resolve_threshold(meta),
        "threshold_strategy": meta.get("threshold_strategy"),
        "threshold_tuned_on": meta.get("threshold_tuned_on"),
        "trained_on": meta.get("trained_on"),
        "evaluation": {
            "train_years": data.get("train_years"),
            "val_years": data.get("val_years"),
            "test_years": data.get("test_years"),
            "production_years": data.get("production_years"),
            "refit_all": data.get("refit_all"),
        },
        # Gain-based importance over the 15 features, as trained. This is what
        # replaces the prototype's saliency map.
        "feature_importance": meta.get("feature_importance") or {},
    }

    for label, block in (("val", val), ("test", test)):
        for key in ("roc_auc", "average_precision"):
            if block.get(key) is not None:
                card[f"{label}_{key}"] = block[key]

    # The test-year figures are the headline ones: tested exactly once, on a
    # year never used for fitting or threshold tuning.
    card["headline"] = (
        f"ROC-AUC {card['test_roc_auc']:.3f} on {data.get('test_years')}, "
        f"tested once"
        if card.get("test_roc_auc") is not None
        else "No test metric recorded in the model artifact"
    )
    return card


def _gee_sampler(lon: float, lat: float, day_start: str, day_end: str,
                 grid: int, pixel_km: float) -> dict[str, np.ndarray]:
    """Sample the training feature stack on a regular grid around (lon, lat).

    Returns one (grid, grid) array per raw band. This reuses
    ``data_extraction._stack_for_window`` rather than rebuilding the stack, so a
    change to how a feature is computed reaches this service automatically — the
    prototype's separate ``gee_data.py`` was a second copy of the pipeline that
    could drift from the one the model was trained on.
    """
    from .data_extraction import _require_ee, _stack_for_window, init_ee

    init_ee()
    ee = _require_ee()

    # Half-width in degrees. Latitude is ~111 km per degree; longitude shrinks
    # by cos(lat), so a naive square box would be too narrow in x up north.
    half_km = (grid * pixel_km) / 2.0
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(np.cos(np.radians(lat)), 0.1))
    bbox = ee.Geometry.Rectangle([lon - dlon, lat - dlat, lon + dlon, lat + dlat])

    stack, _ = _stack_for_window(ee, day_start, day_end, geom=bbox)
    scale_m = int(pixel_km * 1000)
    rect = stack.reproject(crs="EPSG:4326", scale=scale_m).sampleRectangle(
        region=bbox, defaultValue=0
    )
    props = rect.getInfo()["properties"]

    out: dict[str, np.ndarray] = {}
    for band, value in props.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2:
            continue
        # sampleRectangle honours the region loosely; pad or crop to the exact
        # grid so every band lines up cell-for-cell.
        fixed = np.zeros((grid, grid), dtype=np.float32)
        h, w = min(arr.shape[0], grid), min(arr.shape[1], grid)
        fixed[:h, :w] = arr[:h, :w]
        out[band] = fixed
    return out


def _frame_from_bands(bands: dict[str, np.ndarray], date: str, grid: int) -> pd.DataFrame:
    missing = [b for b in REQUIRED_BANDS if b not in bands]
    if missing:
        raise DataUnavailable(
            f"Earth Engine returned no data for these bands: {missing}. "
            "Either the window is empty or a band was renamed upstream."
        )
    frame = pd.DataFrame({band: bands[band].reshape(-1) for band in bands})
    # derive_all reads `date` for month / day_of_year.
    frame["date"] = date
    if len(frame) != grid * grid:
        raise DataUnavailable(
            f"Expected {grid * grid} cells, sampled {len(frame)}."
        )
    return frame


def risk_grid(lat: float, lon: float,
              target_date: Optional[str] = None,
              grid: int = DEFAULT_GRID,
              pixel_km: float = DEFAULT_PIXEL_KM,
              sampler: Optional[Callable[..., dict[str, np.ndarray]]] = None) -> dict:
    """Score a grid of cells around (lat, lon) with the shipped ignition model.

    ``sampler`` exists so tests can run the whole scoring path -- derive_all,
    the real model, the real threshold -- without touching Earth Engine.
    """
    if not CFG.model_path().exists():
        raise ModelUnavailable(
            f"No trained model at {CFG.model_path()}. "
            "Train one first: python -m wildfire.train --synthetic"
        )

    sampler = sampler or _gee_sampler
    target = datetime.strptime(
        target_date or datetime.now(timezone.utc).date().isoformat(), "%Y-%m-%d"
    ).date()

    last_error: Optional[Exception] = None
    for shift in WINDOW_SHIFTS:
        end = target - timedelta(days=shift)
        start = end - timedelta(days=7)
        day_start, day_end = start.isoformat(), end.isoformat()
        try:
            bands = sampler(lon, lat, day_start, day_end, grid, pixel_km)
            frame = _frame_from_bands(bands, day_end, grid)
        except DataUnavailable as exc:
            last_error = exc
            continue
        except Exception as exc:  # GEE raises on fully-empty ERA5 windows
            last_error = exc
            continue

        scored = score_dataframe(frame)
        threshold = resolve_threshold(_meta())
        proba = scored["fire_probability"].to_numpy(dtype=np.float32)

        half_km = (grid * pixel_km) / 2.0
        dlat = half_km / 111.0
        dlon = half_km / (111.0 * max(np.cos(np.radians(lat)), 0.1))

        return {
            "lat": lat,
            "lon": lon,
            "grid": grid,
            "pixel_km": pixel_km,
            "bounds": {
                "west": lon - dlon, "south": lat - dlat,
                "east": lon + dlon, "north": lat + dlat,
            },
            "window": {"start": day_start, "end": day_end,
                       "days_behind_target": shift},
            "data_source": "live_gee",
            # Calibrated probabilities, returned exactly as the model emits
            # them. Not rescaled -- see the module docstring.
            "fire_probability": proba.reshape(grid, grid).tolist(),
            "fire_alert": (proba >= threshold).astype(np.int8)
                          .reshape(grid, grid).tolist(),
            "max_probability": float(proba.max()),
            "mean_probability": float(proba.mean()),
            "alert_cells": int((proba >= threshold).sum()),
            "model": model_card(),
        }

    raise DataUnavailable(
        f"No usable Earth Engine data for any window within "
        f"{max(WINDOW_SHIFTS)} days of {target}. Last error: {last_error}"
    )


def create_app():
    """Build the FastAPI app. Imported lazily so the package works without it."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="West Coast wildfire ignition risk",
        description=__doc__,
    )

    # The dashboard is a static page served from elsewhere, so the browser
    # calling this is always cross-origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    class PredictRequest(BaseModel):
        lat: float = Field(..., ge=-90, le=90)
        lon: float = Field(..., ge=-180, le=180)
        target_date: Optional[str] = None
        grid: int = Field(DEFAULT_GRID, ge=8, le=96)

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "model_present": CFG.model_path().exists(),
            "meta_present": CFG.feature_meta_path().exists(),
        }

    @app.get("/model")
    def model() -> dict:
        try:
            return model_card()
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        try:
            return risk_grid(req.lat, req.lon, req.target_date, grid=req.grid)
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except DataUnavailable as exc:
            # 503 and not a synthetic grid: a risk map that quietly switches to
            # made-up features is worse than one that says it has no data.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve interactive ignition risk.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
