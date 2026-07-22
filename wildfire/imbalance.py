"""
Class-imbalance handling for wildfire ignition data.

Why a custom routine instead of plain SMOTE:
  The project's research doc ("Comprehensive Analysis of Wildfire Prediction")
  recommends *domain-guided* synthetic minority injection. Pure SMOTE
  interpolates between fire samples without regard for fire physics, so the
  synthetic samples cluster near the existing minority manifold. Bumping the
  perturbed samples toward extreme-fire-weather regimes (hot, dry, windy,
  drought-stressed) teaches the model that those regimes carry high ignition
  probability — which is exactly the operational regime that matters most.

This routine MUST only run on training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG


def _perturb_minority(minority: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Push duplicated fire rows toward extreme fire-weather regimes."""
    out = minority.copy()
    noise = lambda scale, n: rng.normal(0.0, scale, size=n)
    n = len(out)

    if "temp_c" in out:
        out["temp_c"]    = out["temp_c"] + CFG.extreme_temp_bump_c + noise(1.5, n)
    if "wind_speed_ms" in out:
        out["wind_speed_ms"] = np.maximum(
            0.0, out["wind_speed_ms"] + CFG.extreme_wind_bump_ms + noise(1.0, n)
        )
    if "kbdi_index" in out:
        out["kbdi_index"] = np.clip(
            np.maximum(out["kbdi_index"], CFG.extreme_kbdi_floor) + noise(15.0, n),
            0.0, 800.0,
        )
    if "rh_pct" in out:
        # Floor at 1% (not 0): record-low fire-weather RH is a few percent,
        # and the dewpoint inversion below is undefined at RH=0.
        out["rh_pct"] = np.clip(
            np.minimum(out["rh_pct"], CFG.extreme_rh_ceiling_pct) + noise(2.0, n),
            1.0, 100.0,
        )
    if "soil_moist_surf" in out:
        out["soil_moist_surf"] = np.clip(out["soil_moist_surf"] * 0.5 + noise(0.01, n), 0.0, 1.0)
    if "soil_moist_root" in out:
        out["soil_moist_root"] = np.clip(out["soil_moist_root"] * 0.6 + noise(0.01, n), 0.0, 1.0)
    # Keep the thermodynamic triple (temp, dewpoint, RH) physically consistent:
    # temp and RH were just perturbed independently, so dewpoint and VPD are
    # re-derived from the perturbed pair instead of carrying stale values that
    # sit off the Magnus curve.
    if {"temp_c", "rh_pct"}.issubset(out.columns):
        from .features import dewpoint_c_from_rh, vpd_kpa
        if "dewpoint_c" in out.columns:
            out["dewpoint_c"] = dewpoint_c_from_rh(out["temp_c"], out["rh_pct"])
        out["vpd_kpa"] = vpd_kpa(out["temp_c"], out["rh_pct"])

    return out


def rebalance_training_set(
    df: pd.DataFrame,
    seed: int = CFG.gee_seed,
) -> pd.DataFrame:
    """
    Apply domain-guided synthetic minority injection (per the research doc)
    followed by majority downsampling so the final neg:pos ratio is at most
    `CFG.majority_downsample_ratio`. Returns a shuffled dataframe.

    Note: because the injected positives are deliberately pushed toward
    extreme fire weather, feature importances of the drought/wind features are
    partly shaped by this augmentation — compare against a no-injection run
    before reading physics into them.
    """
    rng = np.random.default_rng(seed)
    target = CFG.target_column

    if target not in df.columns:
        raise KeyError(f"Expected target column '{target}' in dataframe.")

    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    if len(pos) == 0:
        raise ValueError("No positive (fire) samples in training set — cannot rebalance.")

    # 1. Duplicate minority then perturb each duplicate independently
    duplicated = pd.concat(
        [_perturb_minority(pos, rng) for _ in range(CFG.minority_oversample_factor)],
        ignore_index=True,
    )
    augmented_pos = pd.concat([pos, duplicated], ignore_index=True)

    # 2. Cap majority class at the configured ratio
    target_neg = min(len(neg), len(augmented_pos) * CFG.majority_downsample_ratio)
    sampled_neg = neg.sample(n=target_neg, random_state=seed)

    out = pd.concat([augmented_pos, sampled_neg], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def scale_pos_weight(df: pd.DataFrame) -> float:
    """Compute XGBoost's scale_pos_weight = neg_count / pos_count on residual imbalance."""
    pos = int((df[CFG.target_column] == 1).sum())
    neg = int((df[CFG.target_column] == 0).sum())
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)
