"""Data-driven threshold tuning from the PR curve."""

import numpy as np
import pytest

from wildfire.train import _tune_threshold


def test_tune_threshold_picks_f1_optimum():
    # Construct a known PR scenario: 100 samples, half positive, model perfectly
    # separates classes at threshold 0.5 -> F1 at 0.5 should be 1.0
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    proba = np.array([rng.uniform(0.0, 0.4) for _ in range(50)] +
                     [rng.uniform(0.6, 1.0) for _ in range(50)])
    out = _tune_threshold(y, proba, strategy="f1")

    # F1 at the chosen threshold should be near-perfect, threshold near 0.5
    assert 0.4 < out["f1"]["threshold"] < 0.65
    assert out["f1"]["f1"] > 0.95
    assert out["chosen_strategy"] == "f1"
    assert out["chosen_threshold"] == out["f1"]["threshold"]


def test_tune_threshold_reports_all_three_strategies():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.1).astype(int)
    proba = rng.random(200)
    out = _tune_threshold(y, proba, strategy="f1")

    for k in ("f1", "youden", "precision@0.10"):
        assert k in out
        assert "threshold" in out[k]


def test_tune_threshold_handles_perfect_predictor():
    # Both classes perfectly ranked
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    out = _tune_threshold(y, proba, strategy="f1")
    # F1 = 1 achievable somewhere in [0.3, 0.7] (float-tolerant compare)
    assert out["f1"]["f1"] == pytest.approx(1.0)
