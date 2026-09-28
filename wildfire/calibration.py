"""
Probability calibration for the ignition model.

Training rebalances hard -- 30 perturbed copies of every fire row, negatives
capped at 10:1, then scale_pos_weight on top -- so the booster learns on a
roughly balanced prior. Its raw output is a ranking score, not a probability:
on the committed schema-v1 snapshots the mean raw score was 9-17x the observed
weekly fire rate.

Platt scaling fixes that: fit p = sigmoid(a * logit(raw) + b) by maximum
likelihood on the validation year. It is monotonic, so ROC-AUC and AP are
unchanged and the threshold keeps the same operating point. Fitting it on the
validation year keeps the test year untouched.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(t: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -t))


def fit_platt(proba, y, max_iter: int = 100) -> dict:
    """Maximum-likelihood Platt scaling on logit(raw), by damped Newton steps.

    Two parameters and a convex loss, so this converges in a handful of
    iterations even on millions of rows, with no solver-version drift.
    """
    z = _logit(proba)
    y = np.asarray(y, dtype=np.float64)
    if y.min() == y.max():
        raise ValueError("Calibration needs both classes in the validation set.")
    X = np.column_stack([z, np.ones_like(z)])

    def nll(w):
        t = X @ w
        return float(np.sum(np.logaddexp(0.0, t) - y * t))

    w = np.array([1.0, 0.0])
    loss = nll(w)
    for _ in range(max_iter):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y)
        hess = (X * (p * (1.0 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        scale = 1.0
        while scale > 1e-6:
            candidate = w - scale * step
            new_loss = nll(candidate)
            if new_loss <= loss:
                break
            scale /= 2.0
        else:
            break
        converged = abs(loss - new_loss) < 1e-10 * max(1.0, abs(loss))
        w, loss = candidate, new_loss
        if converged:
            break

    a, b = float(w[0]), float(w[1])
    if a <= 0:
        raise ValueError(
            f"Calibration slope {a:.3f} <= 0: the model ranks the validation "
            "year backwards, so there is nothing to calibrate."
        )
    return {"method": "platt", "a": a, "b": b}


def apply_calibration(proba, calibration: Optional[dict]) -> np.ndarray:
    """Map raw scores to calibrated probabilities. No calibration -> unchanged."""
    proba = np.asarray(proba, dtype=np.float64)
    if not calibration:
        return proba
    if calibration.get("method") != "platt":
        raise ValueError(f"Unknown calibration method {calibration.get('method')!r}")
    return _sigmoid(calibration["a"] * _logit(proba) + calibration["b"])


def calibration_report(y, proba, n_bins: int = 10) -> dict:
    """How far predicted probabilities are from observed rates.

    Bins are equal-count (quantiles of the prediction), because at a ~0.1%
    base rate nearly everything sits in the lowest tenth of [0, 1] and
    equal-width bins would put all rows in one bin.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    bins = []
    ece = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        mean_p, rate = float(p[idx].mean()), float(y[idx].mean())
        ece += len(idx) / len(p) * abs(mean_p - rate)
        bins.append({"n": int(len(idx)), "mean_predicted": mean_p, "observed_rate": rate})
    mean_p, rate = float(p.mean()), float(y.mean())
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "mean_predicted": mean_p,
        "observed_rate": rate,
        "predicted_over_observed": (mean_p / rate) if rate > 0 else None,
        "ece": float(ece),
        "reliability": bins,
    }
