"""
Shared pytest fixtures for the wildfire test suite.

All tests are CPU-only and require no external services (no GEE, no Kaggle).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_synthetic_df(rng):
    """Tiny regime-conditional dataset for fast tests."""
    from wildfire.synthetic_data import make_synthetic_dataset
    return make_synthetic_dataset(n_total=2_000, seed=42)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Redirect the cache to a tmp dir so tests can't pollute the real cache."""
    from wildfire import cache as cache_mod
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    return tmp_path
