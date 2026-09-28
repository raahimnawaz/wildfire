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


def _set_cfg(**overrides) -> dict:
    """Override fields on the frozen CFG; returns the previous values."""
    from wildfire.config import CFG
    previous = {k: getattr(CFG, k) for k in overrides}
    for k, v in overrides.items():
        object.__setattr__(CFG, k, v)
    return previous


@pytest.fixture(scope="session", autouse=True)
def isolated_model_dir(tmp_path_factory):
    """Point model artifacts at a temp dir for the whole session, so no test
    can read -- or overwrite -- the artifact committed in wildfire_models/."""
    model_dir = tmp_path_factory.mktemp("wildfire_models")
    previous = _set_cfg(model_dir=model_dir)
    yield model_dir
    _set_cfg(**previous)


@pytest.fixture(scope="session")
def trained_model(isolated_model_dir):
    """A small ignition model trained end-to-end by wildfire.train on
    synthetic data: real rebalancing, calibration, threshold tuning and
    metadata, just fewer rows and trees."""
    pytest.importorskip("xgboost")
    from wildfire import train as train_mod

    previous = _set_cfg(xgb_n_estimators=40)
    try:
        meta = train_mod.train(synthetic=True, synthetic_rows=20_000)
    finally:
        _set_cfg(**previous)
    return meta


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
