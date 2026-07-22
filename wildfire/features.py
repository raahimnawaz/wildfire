"""
Feature engineering shared by the GEE pipeline, the synthetic data generator,
and the prediction script. Keep all derivations here so train and serve agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG


def kelvin_to_celsius(arr):
    return arr - 273.15


def relative_humidity_pct(temp_c, dewpoint_c):
    """August–Roche–Magnus approximation."""
    a, b = 17.625, 243.04
    es = np.exp((a * temp_c) / (b + temp_c))
    e  = np.exp((a * dewpoint_c) / (b + dewpoint_c))
    return np.clip(100.0 * e / es, 0.0, 100.0)


def vpd_kpa(temp_c, rh_pct):
    """Vapor pressure deficit in kPa — a classic fire-weather driver."""
    es_kpa = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    return np.clip(es_kpa * (1.0 - rh_pct / 100.0), 0.0, None)


def dewpoint_c_from_rh(temp_c, rh_pct):
    """
    Invert the August–Roche–Magnus approximation: recover the dewpoint that
    is thermodynamically consistent with a given (temperature, RH) pair.
    Uses the same constants as relative_humidity_pct so the round trip
    dewpoint -> RH -> dewpoint is exact.
    """
    a, b = 17.625, 243.04
    rh = np.clip(rh_pct, 1.0, 100.0)  # log(0) guard
    gamma = np.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return b * gamma / (a - gamma)


def wind_speed_ms(u, v):
    return np.sqrt(u * u + v * v)


def wind_direction_deg(u, v):
    """Meteorological convention: direction wind is coming FROM."""
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def derive_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a dataframe of raw GEE columns and return the feature matrix
    described in CFG.feature_columns. Idempotent — safe to run twice.
    """
    out = df.copy()

    # ERA5 temperature unit conversion if still in Kelvin
    if "temperature_2m" in out.columns and "temp_c" not in out.columns:
        out["temp_c"] = kelvin_to_celsius(out["temperature_2m"])
    if "dewpoint_temperature_2m" in out.columns and "dewpoint_c" not in out.columns:
        out["dewpoint_c"] = kelvin_to_celsius(out["dewpoint_temperature_2m"])

    # Wind components -> speed + direction
    if {"u_component_of_wind_10m", "v_component_of_wind_10m"}.issubset(out.columns):
        u = out["u_component_of_wind_10m"]
        v = out["v_component_of_wind_10m"]
        out["wind_speed_ms"] = wind_speed_ms(u, v)
        out["wind_dir_deg"]  = wind_direction_deg(u, v)

    # Humidity & VPD
    if {"temp_c", "dewpoint_c"}.issubset(out.columns):
        out["rh_pct"]  = relative_humidity_pct(out["temp_c"], out["dewpoint_c"])
        out["vpd_kpa"] = vpd_kpa(out["temp_c"], out["rh_pct"])

    # Soil moisture aliasing
    if "ssm" in out.columns and "soil_moist_surf" not in out.columns:
        out["soil_moist_surf"] = out["ssm"]
    if "volumetric_soil_water_layer_3" in out.columns and "soil_moist_root" not in out.columns:
        out["soil_moist_root"] = out["volumetric_soil_water_layer_3"]

    # Calendar features
    if "date" in out.columns:
        dates = pd.to_datetime(out["date"])
        out["month"]       = dates.dt.month.astype("int16")
        out["day_of_year"] = dates.dt.dayofyear.astype("int16")

    # Final column ordering — fail loud if anything is missing
    missing = [c for c in CFG.feature_columns if c not in out.columns]
    if missing:
        raise ValueError(
            f"Feature matrix is missing required columns: {missing}. "
            "Check your data extraction step."
        )
    return out


def feature_matrix(df: pd.DataFrame):
    """Return X (np.ndarray) and y (np.ndarray | None) ready for XGBoost."""
    df = derive_all(df)
    X = df[CFG.feature_columns].astype("float32").values
    y = df[CFG.target_column].astype("int8").values if CFG.target_column in df.columns else None
    return X, y
