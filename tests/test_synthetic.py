"""Smoke test that the synthetic dataset matches the documented schema."""

import pytest

from wildfire.config import CFG
from wildfire.features import derive_all
from wildfire.synthetic_data import make_synthetic_dataset


def test_synthetic_has_target_column():
    df = make_synthetic_dataset(n_total=500, seed=0)
    assert CFG.target_column in df.columns
    assert df[CFG.target_column].isin([0, 1]).all()


def test_synthetic_imbalance_roughly_matches_positive_rate():
    df = make_synthetic_dataset(n_total=5_000, positive_rate=0.05, seed=0)
    actual_rate = df[CFG.target_column].mean()
    # within +/- 1.5% of the requested rate at this sample size
    assert abs(actual_rate - 0.05) < 0.015


def test_derived_features_present_after_derive_all():
    df = make_synthetic_dataset(n_total=200, seed=0)
    df = derive_all(df)
    for col in CFG.feature_columns:
        assert col in df.columns, f"Missing feature column: {col}"


def test_synthetic_year_range_covers_both_train_and_test_year():
    """Bug we hit in development: synthetic split was picking one year per call."""
    import pandas as pd
    df = make_synthetic_dataset(n_total=1_000, seed=0)
    years = pd.to_datetime(df["date"]).dt.year.unique()
    assert CFG.train_year in years
    assert CFG.test_year in years


def test_dewpoint_never_exceeds_temperature():
    # Physics check: dewpoint <= temp by construction
    df = make_synthetic_dataset(n_total=500, seed=0)
    assert (df["dewpoint_c"] <= df["temp_c"]).all()
