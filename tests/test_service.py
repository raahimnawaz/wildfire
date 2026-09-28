"""Interactive risk service: real probabilities, real metrics, no fallbacks.

Every test here runs the full scoring path -- derive_all, a real XGBoost
artifact trained by wildfire.train (on synthetic data, into a temp dir), the
validation-fitted calibration and threshold -- against an injected sampler,
so the suite stays CPU-only and never touches Earth Engine.

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
    ModelUnavailable,
    model_card,
    risk_grid,
)

pytest.importorskip("xgboost", reason="scoring needs the trained booster")


@pytest.fixture(autouse=True)
def fresh_meta_cache():
    from wildfire import service
    service._meta.cache_clear()
    yield
    service._meta.cache_clear()


def _bands(grid, *, erc=45.0, wind=2.0, temp_k=300.0, dewpoint_k=285.0):
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
        "ssm_filled": const(0.0),
        "erc": const(erc),
        "erc_filled": const(0.0),
        "ndvi": const(0.3),
        "ndvi_filled": const(0.0),
        "fire_recent": const(0.0),
    }


def _sampler(**kwargs):
    def sample(lon, lat, day_start, day_end, grid, pixel_km):
        return _bands(grid, **kwargs)
    return sample


# --------------------------------------------------------------------------
# Metrics come from the artifact, never from a literal
# --------------------------------------------------------------------------

def test_model_card_metrics_match_the_artifact_exactly(trained_model):
    meta = json.loads(CFG.feature_meta_path().read_text())
    card = model_card()

    assert card["test_roc_auc"] == meta["metrics"]["test"]["roc_auc"]
    assert card["val_roc_auc"] == meta["metrics"]["val"]["roc_auc"]
    assert card["threshold"] == pytest.approx(meta["operational_threshold"])
    # Provenance has to travel with the number: tuned on the validation year,
    # tested once on a later one, and what the label actually is.
    assert card["threshold_tuned_on"] == meta["threshold_tuned_on"]
    assert card["evaluation"]["test_years"] == meta["data"]["test_years"]
    assert card["label"] == meta["label"]
    assert card["probability_calibrated"] is True


def test_model_card_carries_real_feature_importance(trained_model):
    card = model_card()
    importance = card["feature_importance"]

    assert importance, "importance is what replaced the prototype's saliency map"
    assert set(importance) <= set(CFG.feature_columns)
    assert importance == trained_model["feature_importance"]


def test_model_card_omits_metrics_it_does_not_have(trained_model, monkeypatch):
    """A missing metric must be absent, not defaulted to something plausible."""
    from wildfire import service

    stripped = json.loads(CFG.feature_meta_path().read_text())
    stripped["metrics"] = {"val": {}, "test": {}}
    monkeypatch.setattr(service, "_meta", lambda: stripped)

    card = service.model_card()
    assert "test_roc_auc" not in card
    assert "val_average_precision" not in card
    assert "No test metric recorded" in card["headline"]


def test_legacy_artifact_is_refused_not_misread(trained_model, monkeypatch):
    """A schema-v1 artifact (kbdi_index in the feature list) must be a 503
    that says to retrain -- scoring it would feed ERC into the KBDI column."""
    from wildfire import service

    legacy = json.loads(CFG.feature_meta_path().read_text())
    legacy["feature_columns"] = [("kbdi_index" if c == "erc" else c)
                                 for c in CFG.feature_columns]
    path = CFG.feature_meta_path()
    original = path.read_text()
    try:
        path.write_text(json.dumps(legacy))
        service._meta.cache_clear()
        with pytest.raises(ModelUnavailable, match="Retrain"):
            model_card()
        with pytest.raises(ModelUnavailable, match="Retrain"):
            risk_grid(40.0, -121.0, "2025-08-01", grid=8, sampler=_sampler())
    finally:
        path.write_text(original)


# --------------------------------------------------------------------------
# Probabilities are the model's own, calibrated, not rescaled
# --------------------------------------------------------------------------

def test_probabilities_are_not_rescaled_per_request(trained_model):
    """Regression: the prototype min-max normalised every response to [0, 1].

    Under that bug a uniformly mild patch and a tinder-dry one both came back
    with max exactly 1.0, so the UI's "max risk" readout was always 100%.
    """
    mild = risk_grid(40.0, -121.0, "2025-08-01", grid=16,
                     sampler=_sampler(erc=20.0, wind=0.5, temp_k=290.0, dewpoint_k=285.0))
    dry = risk_grid(40.0, -121.0, "2025-08-01", grid=16,
                    sampler=_sampler(erc=90.0, wind=12.0, temp_k=310.0, dewpoint_k=270.0))

    assert mild["max_probability"] != 1.0
    assert dry["max_probability"] != 1.0
    # The two patches must be distinguishable, which is the whole point.
    assert dry["max_probability"] > mild["max_probability"]


def test_uniform_input_gives_a_uniform_grid(trained_model):
    """A constant field must score constant -- rescaling would manufacture spread."""
    out = risk_grid(40.0, -121.0, "2025-08-01", grid=16, sampler=_sampler())
    grid = np.asarray(out["fire_probability"], dtype=float)

    assert grid.shape == (16, 16)
    assert np.ptp(grid) == pytest.approx(0.0, abs=1e-9)
    assert out["max_probability"] == pytest.approx(out["mean_probability"])
    assert out["probability_calibrated"] is True


def test_alert_mask_uses_the_tuned_threshold(trained_model):
    out = risk_grid(40.0, -121.0, "2025-08-01", grid=16,
                    sampler=_sampler(erc=90.0, wind=12.0, temp_k=310.0, dewpoint_k=270.0))
    proba = np.asarray(out["fire_probability"], dtype=float)
    alert = np.asarray(out["fire_alert"])
    threshold = out["model"]["threshold"]

    assert np.array_equal(alert, (proba >= threshold).astype(int))
    assert out["alert_cells"] == int(alert.sum())


def test_probabilities_stay_in_range_and_grid_defaults_to_training_scale(trained_model):
    out = risk_grid(40.0, -121.0, "2025-08-01", sampler=_sampler())
    proba = np.asarray(out["fire_probability"], dtype=float)

    assert out["grid"] == DEFAULT_GRID
    # Sampling must match the 2 km cells the model was trained on.
    assert out["pixel_km"] == pytest.approx(CFG.cell_scale_m / 1000.0)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_ocean_padding_and_filled_cells_are_null_not_scored(trained_model):
    """sampleRectangle writes 0 into masked pixels and pads the grid with 0.
    Those cells used to be scored as if 0 K were a temperature."""
    def coastal(lon, lat, day_start, day_end, grid, pixel_km):
        bands = _bands(grid)
        for band in bands:
            bands[band][:, :4] = 0.0          # ocean / padding: everything 0
        bands["erc_filled"][0, 10] = 1.0      # one land cell outside GRIDMET
        return bands

    out = risk_grid(40.0, -124.0, "2025-08-01", grid=16, sampler=coastal)
    proba = out["fire_probability"]

    assert all(proba[r][c] is None for r in range(16) for c in range(4))
    assert proba[0][10] is None
    assert out["fire_alert"][0][10] is None
    assert out["valid_cells"] == 16 * 12 - 1
    assert proba[5][10] is not None


def test_frame_is_stamped_with_the_window_midpoint(trained_model, monkeypatch):
    """Training rows carry their window's midpoint date; serving must match."""
    from wildfire import service
    seen = {}
    real = service._frame_from_bands

    def spy(bands, date, grid):
        seen["date"] = date
        return real(bands, date, grid)

    monkeypatch.setattr(service, "_frame_from_bands", spy)
    risk_grid(40.0, -121.0, "2025-08-10", grid=8, sampler=_sampler())
    assert seen["date"] == "2025-08-06T12:00:00" or seen["date"].startswith("2025-08-06")


# --------------------------------------------------------------------------
# No data is an error, not an invention
# --------------------------------------------------------------------------

def test_window_walks_back_when_the_freshest_window_is_empty(trained_model):
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


def test_a_dead_source_is_an_outage_not_a_constant(trained_model):
    """Regression: KBDI returned its fallback in 100% of cells for years and
    nothing noticed. A mostly-filled band must fail the window."""
    def erc_outage(lon, lat, day_start, day_end, grid, pixel_km):
        bands = _bands(grid)
        bands["erc_filled"][:] = 1.0
        return bands

    with pytest.raises(DataUnavailable, match="erc is fallback"):
        risk_grid(40.0, -121.0, "2025-08-10", grid=16, sampler=erc_outage)


def test_exhausted_windows_raise_instead_of_returning_synthetic_data(trained_model):
    """Regression: the prototype silently served synthetic features here."""
    def always_empty(lon, lat, day_start, day_end, grid, pixel_km):
        raise DataUnavailable("nothing for this window")

    with pytest.raises(DataUnavailable, match="No usable Earth Engine data"):
        risk_grid(40.0, -121.0, "2025-08-10", grid=16, sampler=always_empty)


def test_a_renamed_or_missing_band_is_named_not_guessed(trained_model):
    def drops_erc(lon, lat, day_start, day_end, grid, pixel_km):
        bands = _bands(grid)
        del bands["erc"]
        return bands

    with pytest.raises(DataUnavailable, match="erc"):
        risk_grid(40.0, -121.0, "2025-08-10", grid=16, sampler=drops_erc)


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------

def test_health_and_model_endpoints(trained_model):
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
    assert body["test_roc_auc"] == pytest.approx(
        trained_model["metrics"]["test"]["roc_auc"])
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
