"""Interactive risk service: real probabilities, real metrics, no fallbacks.

Every test here runs the full scoring path -- derive_all, the shipped XGBoost
artifact, the validation-tuned threshold -- against an injected sampler, so the
suite stays CPU-only and never touches Earth Engine.

Three of these are regression tests for specific defects in the uncommitted
prototype this module replaced: invented metrics, min-max rescaled
probabilities, and a silent synthetic fallback.
"""

import json

import numpy as np
import pytest

from wildfire.config import CFG
from wildfire.service import (
    DEFAULT_GRID,
    DataUnavailable,
    model_card,
    risk_grid,
)

pytest.importorskip("xgboost", reason="scoring needs the trained booster")

MODEL_PRESENT = CFG.model_path().exists() and CFG.feature_meta_path().exists()
needs_model = pytest.mark.skipif(
    not MODEL_PRESENT, reason="trained artifact not present in this checkout"
)


def _bands(grid, *, kbdi=300.0, wind=2.0, temp_k=300.0, dewpoint_k=285.0):
    """Plausible raw GEE bands, uniform across the grid unless varied."""
    def const(value):
        return np.full((grid, grid), value, dtype=np.float32)

    return {
        "temperature_2m": const(temp_k),
        "dewpoint_temperature_2m": const(dewpoint_k),
        "u_component_of_wind_10m": const(wind),
        "v_component_of_wind_10m": const(0.0),
        "volumetric_soil_water_layer_3": const(0.2),
        "elevation_m": const(500.0),
        "slope_deg": const(5.0),
        "aspect_deg": const(180.0),
        "ssm": const(0.2),
        "kbdi_index": const(kbdi),
        "ndvi": const(0.3),
    }


def _sampler(**kwargs):
    def sample(lon, lat, day_start, day_end, grid, pixel_km):
        return _bands(grid, **kwargs)
    return sample


# --------------------------------------------------------------------------
# Metrics come from the artifact, never from a literal
# --------------------------------------------------------------------------

@needs_model
def test_model_card_metrics_match_the_artifact_exactly():
    meta = json.loads(CFG.feature_meta_path().read_text())
    card = model_card()

    assert card["test_roc_auc"] == meta["metrics"]["test"]["roc_auc"]
    assert card["val_roc_auc"] == meta["metrics"]["val"]["roc_auc"]
    assert card["threshold"] == pytest.approx(meta["operational_threshold"])
    # Provenance has to travel with the number, or "ROC-AUC 0.785" is just a
    # claim: tuned on the validation year, tested once on a later one.
    assert card["threshold_tuned_on"] == meta["threshold_tuned_on"]
    assert card["evaluation"]["test_years"] == meta["data"]["test_years"]


@needs_model
def test_model_card_carries_real_feature_importance():
    card = model_card()
    importance = card["feature_importance"]

    assert importance, "importance is what replaced the prototype's saliency map"
    # Gain-based importance over the trained features; the README's claim that
    # drought and wind dominate should be visible here and not asserted blind.
    top = max(importance, key=importance.get)
    assert top in {"kbdi_index", "wind_speed_ms"}
    assert set(importance) <= set(CFG.feature_columns)


@needs_model
def test_model_card_omits_metrics_it_does_not_have(monkeypatch):
    """A missing metric must be absent, not defaulted to something plausible."""
    from wildfire import service

    stripped = json.loads(CFG.feature_meta_path().read_text())
    stripped["metrics"] = {"val": {}, "test": {}}
    monkeypatch.setattr(service, "_meta", lambda: stripped)

    card = service.model_card()
    assert "test_roc_auc" not in card
    assert "val_average_precision" not in card
    assert "No test metric recorded" in card["headline"]


# --------------------------------------------------------------------------
# Probabilities are the model's own, not rescaled
# --------------------------------------------------------------------------

@needs_model
def test_probabilities_are_not_rescaled_per_request():
    """Regression: the prototype min-max normalised every response to [0, 1].

    Under that bug a uniformly mild patch and a tinder-dry one both came back
    with max exactly 1.0, so the UI's "max risk" readout was always 100%.
    """
    mild = risk_grid(40.0, -121.0, "2025-08-01", grid=16,
                     sampler=_sampler(kbdi=120.0, wind=1.0))
    dry = risk_grid(40.0, -121.0, "2025-08-01", grid=16,
                    sampler=_sampler(kbdi=780.0, wind=12.0))

    assert mild["max_probability"] != 1.0
    assert dry["max_probability"] != 1.0
    # The two patches must be distinguishable, which is the whole point.
    assert dry["max_probability"] > mild["max_probability"]


@needs_model
def test_uniform_input_gives_a_uniform_grid():
    """A constant field must score constant -- rescaling would manufacture spread."""
    out = risk_grid(40.0, -121.0, "2025-08-01", grid=16, sampler=_sampler())
    grid = np.asarray(out["fire_probability"])

    assert grid.shape == (16, 16)
    assert np.ptp(grid) == pytest.approx(0.0, abs=1e-6)
    assert out["max_probability"] == pytest.approx(out["mean_probability"])


@needs_model
def test_alert_mask_uses_the_tuned_threshold():
    out = risk_grid(40.0, -121.0, "2025-08-01", grid=16,
                    sampler=_sampler(kbdi=780.0, wind=12.0))
    proba = np.asarray(out["fire_probability"])
    alert = np.asarray(out["fire_alert"])
    threshold = out["model"]["threshold"]

    assert np.array_equal(alert, (proba >= threshold).astype(np.int8))
    assert out["alert_cells"] == int(alert.sum())


@needs_model
def test_probabilities_stay_in_range_and_grid_defaults_to_training_scale():
    out = risk_grid(40.0, -121.0, "2025-08-01", sampler=_sampler())
    proba = np.asarray(out["fire_probability"])

    assert out["grid"] == DEFAULT_GRID
    # Sampling must match the 2 km cells the model was trained on.
    assert out["pixel_km"] == pytest.approx(CFG.cell_scale_m / 1000.0)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


# --------------------------------------------------------------------------
# No data is an error, not an invention
# --------------------------------------------------------------------------

@needs_model
def test_window_walks_back_when_the_freshest_window_is_empty():
    """ERA5-Land lags ~5 days; report the window actually used."""
    calls = []

    def flaky(lon, lat, day_start, day_end, grid, pixel_km):
        calls.append(day_end)
        if len(calls) == 1:
            raise DataUnavailable("ERA5 window empty")
        return _bands(grid)

    out = risk_grid(40.0, -121.0, "2025-08-10", grid=16, sampler=flaky)

    assert out["window"]["days_behind_target"] == 2
    assert out["window"]["end"] == "2025-08-08"
    assert len(calls) == 2


@needs_model
def test_exhausted_windows_raise_instead_of_returning_synthetic_data():
    """Regression: the prototype silently served synthetic features here."""
    def always_empty(lon, lat, day_start, day_end, grid, pixel_km):
        raise DataUnavailable("nothing for this window")

    with pytest.raises(DataUnavailable, match="No usable Earth Engine data"):
        risk_grid(40.0, -121.0, "2025-08-10", grid=16, sampler=always_empty)


@needs_model
def test_a_renamed_or_missing_band_is_named_not_guessed():
    def drops_kbdi(lon, lat, day_start, day_end, grid, pixel_km):
        bands = _bands(grid)
        del bands["kbdi_index"]
        return bands

    with pytest.raises(DataUnavailable, match="kbdi_index"):
        risk_grid(40.0, -121.0, "2025-08-10", grid=16, sampler=drops_kbdi)


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------

@needs_model
def test_health_and_model_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx", reason="fastapi.testclient needs httpx")
    from fastapi.testclient import TestClient

    from wildfire.service import create_app

    client = TestClient(create_app())

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_present"] is True

    model = client.get("/model")
    assert model.status_code == 200
    body = model.json()
    assert body["test_roc_auc"] == pytest.approx(0.7854, abs=5e-4)
    assert "f1_score" not in body
    assert "false_negative_rate" not in body


def test_predict_returns_503_when_earth_engine_has_nothing(monkeypatch):
    """A risk map with no data must say so, not render something plausible."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx", reason="fastapi.testclient needs httpx")
    from fastapi.testclient import TestClient

    from wildfire import service

    monkeypatch.setattr(
        service, "risk_grid",
        lambda *a, **k: (_ for _ in ()).throw(DataUnavailable("no GEE data")),
    )
    client = TestClient(service.create_app())

    response = client.post("/predict", json={"lat": 40.0, "lon": -121.0})
    assert response.status_code == 503
    assert "no GEE data" in response.json()["detail"]
