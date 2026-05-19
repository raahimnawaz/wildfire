"""
Synthetic-but-realistic dataset generator.

The point: let the full pipeline (feature engineering -> rebalancing -> training
-> evaluation -> prediction) run end-to-end without Google Earth Engine
authentication. The class-conditional distributions are deliberately calibrated
to match the qualitative pattern seen in real West Coast fire datasets:

  * High-fire days are hot, dry, windy, drought-stressed (high KBDI, low NDVI).
  * Non-fire days lean cooler, wetter, calmer.
  * Topography is sampled from a uniform West Coast prior.
  * Severe ~150:1 class imbalance, matching the paper.

This is NOT a substitute for real GEE-extracted data — but it lets you debug
plumbing, sanity-check metrics, and demonstrate the model behavior offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG
from .features import vpd_kpa, relative_humidity_pct


def _sample_class(n: int, fire: bool, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n rows for one class from a regime-conditional distribution."""
    if fire:
        temp_c     = rng.normal(34.0, 4.0, n)
        dewpoint_c = rng.normal(2.0,  3.5, n)
        u          = rng.normal(0.0, 5.0, n)
        v          = rng.normal(0.0, 5.0, n)
        wind_boost = rng.normal(7.0, 2.5, n)              # Santa Ana / Diablo boost
        wind_dir_bias = rng.uniform(0, 1, n)
        u = np.where(wind_dir_bias > 0.5, u + wind_boost, u - wind_boost)
        kbdi       = rng.normal(680.0, 70.0, n)
        ndvi       = rng.normal(0.18, 0.08, n)
        sm_surf    = rng.beta(1.2, 9.0, n) * 0.4
        sm_root    = rng.beta(1.5, 7.0, n) * 0.5
    else:
        temp_c     = rng.normal(22.0, 5.5, n)
        dewpoint_c = rng.normal(10.0, 4.5, n)
        u          = rng.normal(0.0, 2.5, n)
        v          = rng.normal(0.0, 2.5, n)
        kbdi       = rng.normal(380.0, 130.0, n)
        ndvi       = rng.normal(0.45, 0.15, n)
        sm_surf    = rng.beta(2.5, 5.0, n) * 0.6
        sm_root    = rng.beta(3.0, 4.0, n) * 0.7

    elevation = rng.uniform(0, 3500, n)
    slope     = np.abs(rng.normal(8.0, 7.0, n)).clip(0, 60)
    aspect    = rng.uniform(0, 360, n)
    month     = rng.choice(CFG.fire_season_months, size=n)

    # Per-row year so train/test years both get representation
    years     = rng.choice([CFG.train_year, CFG.test_year], size=n)
    day       = rng.integers(1, 29, size=n)
    dates     = pd.to_datetime(
        {"year": years, "month": month, "day": day}
    )

    df = pd.DataFrame({
        "date":              dates,
        "temp_c":            temp_c,
        "dewpoint_c":        np.minimum(dewpoint_c, temp_c - 0.5),  # dewpoint <= temp
        "u_component_of_wind_10m": u,
        "v_component_of_wind_10m": v,
        "ssm":               np.clip(sm_surf, 0.0, 1.0),
        "volumetric_soil_water_layer_3": np.clip(sm_root, 0.0, 1.0),
        "kbdi_index":        np.clip(kbdi, 0.0, 800.0),
        "ndvi":              np.clip(ndvi, -0.2, 1.0),
        "elevation_m":       elevation,
        "slope_deg":         slope,
        "aspect_deg":        aspect,
        "longitude":         rng.uniform(CFG.west_coast_bbox[0], CFG.west_coast_bbox[2], n),
        "latitude":          rng.uniform(CFG.west_coast_bbox[1], CFG.west_coast_bbox[3], n),
        CFG.target_column:   np.int8(1 if fire else 0),
    })
    return df


def make_synthetic_dataset(
    n_total: int = 200_000,
    positive_rate: float = 1.0 / 150.0,   # paper's typical ~150:1 imbalance
    seed: int = CFG.gee_seed,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pos = max(1, int(round(n_total * positive_rate)))
    n_neg = n_total - n_pos

    pos = _sample_class(n_pos, fire=True,  rng=rng)
    neg = _sample_class(n_neg, fire=False, rng=rng)
    df  = pd.concat([pos, neg], ignore_index=True)
    df  = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = make_synthetic_dataset(n_total=20_000)
    print(df.head())
    print(f"\nTotal rows: {len(df):,}")
    print(f"Positive rate: {df[CFG.target_column].mean():.5f}")
