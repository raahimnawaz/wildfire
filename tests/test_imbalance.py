"""Domain-guided minority injection contract."""

import numpy as np
import pytest

from wildfire.config import CFG
from wildfire.imbalance import rebalance_training_set, scale_pos_weight


def test_rebalance_increases_minority_and_caps_majority(small_synthetic_df):
    n_pos_in  = int((small_synthetic_df[CFG.target_column] == 1).sum())
    n_neg_in  = int((small_synthetic_df[CFG.target_column] == 0).sum())
    out = rebalance_training_set(small_synthetic_df)

    n_pos_out = int((out[CFG.target_column] == 1).sum())
    n_neg_out = int((out[CFG.target_column] == 0).sum())

    # Minority grew (originals + oversample_factor duplicates)
    assert n_pos_out == n_pos_in * (1 + CFG.minority_oversample_factor)
    # Majority capped at the configured ratio
    assert n_neg_out <= n_pos_out * CFG.majority_downsample_ratio + 1
    # Inputs not mutated
    assert int((small_synthetic_df[CFG.target_column] == 1).sum()) == n_pos_in


def test_perturbed_minority_pushed_to_extreme_regimes(small_synthetic_df):
    """Synthetic-augmented fire rows should be hotter / drier than originals."""
    from wildfire.features import derive_all
    # Rebalance operates on the derived feature matrix (wind_speed_ms etc.
    # are computed by derive_all from raw u/v components)
    derived = derive_all(small_synthetic_df)
    pos = derived[derived[CFG.target_column] == 1]
    out = rebalance_training_set(derived)
    out_pos = out[out[CFG.target_column] == 1]

    # On average the rebalanced positives are pushed toward extreme fire weather
    assert out_pos["temp_c"].mean() > pos["temp_c"].mean()
    assert out_pos["kbdi_index"].mean() > pos["kbdi_index"].mean()
    # RH gets pushed DOWN (drier) — sign-flipped check
    assert out_pos["rh_pct"].mean() < pos["rh_pct"].mean()


def test_rebalance_keeps_thermodynamics_consistent(small_synthetic_df):
    """Perturbed rows must stay on the Magnus curve: RH re-derived from the
    (temp, dewpoint) pair matches the stored rh_pct column."""
    from wildfire.features import derive_all, relative_humidity_pct, vpd_kpa
    derived = derive_all(small_synthetic_df)
    out = rebalance_training_set(derived)

    rh_check = relative_humidity_pct(out["temp_c"].values, out["dewpoint_c"].values)
    assert np.allclose(rh_check, out["rh_pct"].values, atol=0.5)

    vpd_check = vpd_kpa(out["temp_c"].values, out["rh_pct"].values)
    assert np.allclose(vpd_check, out["vpd_kpa"].values, atol=1e-3)


def test_scale_pos_weight_matches_ratio(small_synthetic_df):
    out = rebalance_training_set(small_synthetic_df)
    spw = scale_pos_weight(out)
    n_pos = int((out[CFG.target_column] == 1).sum())
    n_neg = int((out[CFG.target_column] == 0).sum())
    assert spw == pytest.approx(n_neg / n_pos)


def test_rebalance_requires_positive_samples(small_synthetic_df):
    """Rebalance should raise if there are no positives to oversample from."""
    from wildfire.features import derive_all
    df = derive_all(small_synthetic_df)
    # Force zero positives — make_synthetic_dataset has a min-1-positive floor
    df_neg_only = df[df[CFG.target_column] == 0].copy()
    with pytest.raises(ValueError):
        rebalance_training_set(df_neg_only)
