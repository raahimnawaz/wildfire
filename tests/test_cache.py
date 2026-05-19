"""Cache hit/miss + key determinism."""

import numpy as np
import pandas as pd
import pytest

from wildfire.cache import (
    cache_summary,
    get_or_compute_npy,
    get_or_compute_parquet,
    key_patch,
    key_window,
)


def test_key_window_deterministic():
    a = key_window("2025-08-01", "2025-08-08", 4000, 10, 4000)
    b = key_window("2025-08-01", "2025-08-08", 4000, 10, 4000)
    assert a == b


def test_key_window_changes_with_args():
    a = key_window("2025-08-01", "2025-08-08", 4000, 10, 4000)
    b = key_window("2025-08-01", "2025-08-08", 2000, 10, 4000)  # different scale
    c = key_window("2025-08-08", "2025-08-15", 4000, 10, 4000)  # different dates
    assert a != b and a != c and b != c


def test_key_patch_includes_coords():
    a = key_patch(-118.5, 34.0, "2025-09-08", "2025-09-15", 64, 1.0)
    b = key_patch(-118.5, 34.1, "2025-09-08", "2025-09-15", 64, 1.0)
    assert a != b


def test_parquet_cache_fill_then_hit(tmp_cache):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    key = "test_window_xyz"
    df1 = get_or_compute_parquet(key, compute, verbose=False)
    df2 = get_or_compute_parquet(key, compute, verbose=False)

    # compute_fn called once, identical output both times
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(df1, df2)
    assert (tmp_cache / f"{key}.parquet").exists()


def test_npy_cache_fill_then_hit(tmp_cache):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.arange(12, dtype=np.float32).reshape(3, 4)

    key = "test_patch_abc"
    arr1 = get_or_compute_npy(key, compute, verbose=False)
    arr2 = get_or_compute_npy(key, compute, verbose=False)

    assert calls["n"] == 1
    np.testing.assert_array_equal(arr1, arr2)
    assert (tmp_cache / f"{key}.npy").exists()


def test_use_cache_false_bypasses_disk(tmp_cache):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2, 3]})

    key = "test_bypass"
    get_or_compute_parquet(key, compute, verbose=False)
    get_or_compute_parquet(key, compute, verbose=False, use_cache=False)
    assert calls["n"] == 2  # bypass forced a recompute


def test_empty_dataframe_not_written(tmp_cache):
    """Empty results from upstream errors shouldn't poison the cache."""
    key = "test_empty"
    df = get_or_compute_parquet(key, lambda: pd.DataFrame(), verbose=False)
    assert df.empty
    assert not (tmp_cache / f"{key}.parquet").exists()


def test_cache_summary_keys(tmp_cache):
    s = cache_summary()
    assert {"cache_dir", "n_files", "n_windows", "n_patches", "total_mb"}.issubset(s)
