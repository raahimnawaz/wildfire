"""Forecast label windows, fill-flag coverage guards, and dead-feature checks.

The Earth Engine side can't run here; these pin down the pure logic that
decides WHICH dates a label reads and WHEN a table is refused.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from wildfire.config import CFG
from wildfire.data_extraction import (
    FILLABLE_BANDS,
    FeatureOutage,
    check_coverage,
    drop_filled_rows,
    extract_window,
    fill_fractions,
    iter_windows,
    label_complete,
    label_windows,
)

# --------------------------------------------------------------------------
# Forecast label: strictly after the features, never same-week
# --------------------------------------------------------------------------

def test_label_week_starts_where_the_feature_window_ends():
    w = label_windows("2024-08-01", "2024-08-08")
    assert w["next"] == ("2024-08-08", "2024-08-15")
    # "Already burning" looks back over the feature week and the one before.
    assert w["recent"] == ("2024-07-25", "2024-08-08")


def test_label_is_incomplete_until_firms_has_seen_the_whole_week():
    # Forecast week 08-08..08-15, plus FIRMS ingest latency.
    assert not label_complete("2024-08-08", as_of=datetime(2024, 8, 15))
    assert label_complete("2024-08-08", as_of=datetime(2024, 8, 17))


def test_training_extraction_refuses_an_unobserved_label_week():
    """Pulling a label before its week is over would record 'no fire' for
    days that haven't happened -- and then cache it."""
    with pytest.raises(ValueError, match="not been fully observed"):
        extract_window("2999-08-01", "2999-08-08")


def test_windows_are_full_length_and_back_to_back():
    windows = list(iter_windows(2023, (8, 9, 10), as_of=datetime(2024, 1, 1)))
    starts = [datetime.strptime(a, "%Y-%m-%d") for a, _ in windows]
    ends = [datetime.strptime(b, "%Y-%m-%d") for _, b in windows]

    assert all((e - s).days == 7 for s, e in zip(starts, ends))
    assert all(ends[i] == starts[i + 1] for i in range(len(windows) - 1))
    assert windows[0][0] == "2023-08-01"
    assert ends[-1] <= datetime(2023, 11, 1)
    assert len(windows) == 13          # 92 days -> 13 full weeks


def test_windows_stop_before_the_unobserved_future():
    windows = list(iter_windows(2026, (8, 9, 10), as_of=datetime(2026, 9, 28)))
    last_label_end = datetime.strptime(label_windows(*windows[-1])["next"][1], "%Y-%m-%d")
    assert last_label_end <= datetime(2026, 9, 26)


def test_end_date_clips_on_the_label_week():
    windows = list(iter_windows(2023, (8,), end_date="2023-08-22",
                                as_of=datetime(2024, 1, 1)))
    assert windows == [("2023-08-01", "2023-08-08"), ("2023-08-08", "2023-08-15")]


def test_non_contiguous_months_restart_at_each_run():
    windows = list(iter_windows(2023, (5, 8), as_of=datetime(2024, 1, 1)))
    assert windows[0][0] == "2023-05-01"
    assert any(start == "2023-08-01" for start, _ in windows)


# --------------------------------------------------------------------------
# Fill flags: a fallback is never data
# --------------------------------------------------------------------------

def _frame(n=100, **filled):
    df = pd.DataFrame({"x": np.arange(n)})
    for band in FILLABLE_BANDS:
        df[f"{band}_filled"] = 0.0
    for band, count in filled.items():
        df.loc[: count - 1, f"{band}_filled"] = 1.0
    return df


def test_fill_fractions_reads_the_flags():
    assert fill_fractions(_frame(erc=25)) == {"ssm": 0.0, "erc": 0.25, "ndvi": 0.0}


def test_a_mostly_filled_band_is_an_outage():
    """Regression: UTOKYO KBDI never covered the West Coast, so 100% of cells
    held the fallback 300 for every season and nothing complained."""
    with pytest.raises(FeatureOutage, match="erc is fallback in 100%"):
        check_coverage(_frame(erc=100), "window X")
    check_coverage(_frame(erc=int(CFG.max_fill_fraction * 100)), "window X")  # at the limit: ok


def test_filled_rows_are_dropped_not_scored():
    df, n = drop_filled_rows(_frame(ssm=3, ndvi=2))
    assert n == 3                       # rows 0-2 (ndvi rows overlap ssm rows)
    assert len(df) == 97
    assert (df[[f"{b}_filled" for b in FILLABLE_BANDS]] == 0).all().all()


# --------------------------------------------------------------------------
# Training refuses dead features and drops already-burning cells
# --------------------------------------------------------------------------

def test_training_refuses_a_constant_feature(small_synthetic_df):
    from wildfire.features import derive_all
    from wildfire.train import _assert_features_alive

    df = derive_all(small_synthetic_df)
    _assert_features_alive(df, "train")         # healthy synthetic data passes

    df["erc"] = 0.0                              # what an outage looks like
    with pytest.raises(ValueError, match="dead feature.*erc"):
        _assert_features_alive(df, "train")


def test_calendar_features_may_be_constant_in_a_short_holdout(small_synthetic_df):
    from wildfire.features import derive_all
    from wildfire.train import _assert_features_alive

    df = derive_all(small_synthetic_df)
    df["date"] = pd.Timestamp("2022-10-15")
    df = derive_all(df)                          # month / day_of_year now constant
    _assert_features_alive(df, "val")


def test_prep_drops_burning_cells_and_fallback_rows(small_synthetic_df):
    from wildfire.train import _prep

    raw = small_synthetic_df.copy()
    for band in FILLABLE_BANDS:
        raw[f"{band}_filled"] = 0.0
    raw["fire_recent"] = 0.0
    raw.loc[:9, "fire_recent"] = 1.0          # 10 cells already burning
    raw.loc[10:14, "erc_filled"] = 1.0        # 5 cells outside GRIDMET

    stats = {}
    out = _prep(raw, stats)
    assert len(out) == len(raw) - 15
    assert stats["dropped_already_burning"] == 10
    assert stats["dropped_fallback_rows"] == 5
    assert "fire_recent" not in out.columns   # never a model input
