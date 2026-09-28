"""Platt calibration: recovers a known distortion, keeps ranking, reports honestly."""

import numpy as np
import pytest

from wildfire.calibration import (
    apply_calibration,
    calibration_report,
    fit_platt,
)


def _inflated_scores(n=200_000, seed=0):
    """True probabilities around a 0.5% base rate, reported the way a model
    trained on a rebalanced prior reports them: ~10x too high."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-5.5, 1.0, n)
    p_true = 1.0 / (1.0 + np.exp(-true_logit))
    y = (rng.random(n) < p_true).astype(int)
    raw = 1.0 / (1.0 + np.exp(-(true_logit + np.log(10.0))))
    return raw, y


def test_platt_removes_a_prior_shift():
    raw, y = _inflated_scores()
    cal = fit_platt(raw, y)
    p = apply_calibration(raw, cal)

    assert raw.mean() / y.mean() > 5            # the problem being fixed
    assert p.mean() == pytest.approx(y.mean(), rel=0.02)
    assert cal["a"] == pytest.approx(1.0, abs=0.1)
    assert cal["b"] == pytest.approx(-np.log(10.0), abs=0.2)


def test_calibration_is_monotonic_so_ranking_metrics_do_not_move():
    from sklearn.metrics import average_precision_score, roc_auc_score
    raw, y = _inflated_scores(n=50_000, seed=1)
    p = apply_calibration(raw, fit_platt(raw, y))

    assert np.all(np.diff(p[np.argsort(raw)]) >= -1e-15)
    assert roc_auc_score(y, p) == pytest.approx(roc_auc_score(y, raw))
    assert average_precision_score(y, p) == pytest.approx(average_precision_score(y, raw))


def test_no_calibration_means_unchanged_scores():
    raw = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(apply_calibration(raw, None), raw)


def test_backwards_model_is_refused():
    y = np.array([0] * 50 + [1] * 50)
    raw = np.r_[np.linspace(0.6, 0.9, 50), np.linspace(0.1, 0.4, 50)]
    with pytest.raises(ValueError, match="backwards"):
        fit_platt(raw, y)


def test_report_uses_equal_count_bins_and_flags_overprediction():
    raw, y = _inflated_scores(n=20_000, seed=2)
    rep = calibration_report(y, raw, n_bins=10)

    assert len(rep["reliability"]) == 10
    assert sum(b["n"] for b in rep["reliability"]) == len(y)
    assert rep["predicted_over_observed"] > 5
    assert 0 <= rep["ece"] <= 1


def test_trained_artifact_is_calibrated_out_of_sample(trained_model):
    """End to end through wildfire.train: the calibrator is fit on validation
    only, and the TEST year's mean prediction lands near its observed rate."""
    test_cal = trained_model["metrics"]["test"]["calibration"]
    assert trained_model["calibration"]["method"] == "platt"
    assert trained_model["threshold_scale"] == "calibrated"
    assert test_cal["predicted_over_observed"] == pytest.approx(1.0, abs=0.5)


def test_score_dataframe_applies_the_artifact_calibration(trained_model):
    from wildfire.predict import score_dataframe
    from wildfire.synthetic_data import make_synthetic_dataset

    df = make_synthetic_dataset(n_total=5_000, seed=7)
    scored = score_dataframe(df)
    rate = df["fire_occurred"].mean()
    # Raw booster scores sit ~10x above the rate; calibrated ones near it.
    assert scored["fire_probability"].mean() == pytest.approx(rate, rel=0.6)
