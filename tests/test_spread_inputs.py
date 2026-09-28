"""Spread-model channels mean the same thing in training and serving.

Until schema v2 the channel called kbdi_index held PDSI in training and KBDI at
serving, soil_moist held specific humidity vs SMAP, dewpoint_c held the daily
MIN temperature, and so on for 9 of 11 channels. These tests pin the NDWS
mapping and check that the serving patch builder fills every channel by name.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from wildfire.spread.config import SPREAD_CFG
from wildfire.spread.ndws import NDWS_FEATURE_MAP, ndws_channels

S = 4


def _parsed():
    """One fake NDWS example with distinct, recognisable values per feature."""
    def plane(v):
        return np.full(S * S, v, dtype=np.float32)
    return {
        "elevation": plane(1200.0),
        "NDVI": plane(5100.0),
        "tmmx": plane(305.15),       # K
        "tmmn": plane(285.15),       # K
        "vs": plane(4.0),
        "th": plane(180.0),          # wind from the south
        "pdsi": plane(-3.5),
        "sph": plane(0.006),
        "PrevFireMask": plane(1.0),
        "erc": plane(60.0),          # present in NDWS but unused
    }


def test_channel_order_is_unchanged_so_old_checkpoints_still_load():
    """Checkpoints store no channel names before v2, only the count; the
    rename kept every channel in its original position."""
    legacy_order = ["elevation_m", "slope_deg", "aspect_deg", "ndvi", "temp_c",
                    "dewpoint_c", "wind_u_ms", "wind_v_ms", "kbdi_index",
                    "soil_moist", "prev_fire_mask"]
    legacy_source = {"elevation_m": "elevation", "ndvi": "NDVI", "temp_c": "tmmx",
                     "dewpoint_c": "tmmn", "wind_u_ms": "vs", "wind_v_ms": "th",
                     "kbdi_index": "pdsi", "soil_moist": "sph",
                     "prev_fire_mask": "PrevFireMask"}
    reverse = {v: k for k, v in NDWS_FEATURE_MAP.items()}
    for old, new in zip(legacy_order, SPREAD_CFG.feature_channels):
        assert legacy_source.get(old) == reverse.get(new), (old, new)


def test_ndws_channels_hold_what_their_names_say():
    x = ndws_channels(_parsed(), S)
    ch = {name: x[i, 0, 0] for i, name in enumerate(SPREAD_CFG.feature_channels)}

    assert x.shape == (len(SPREAD_CFG.feature_channels), S, S)
    assert ch["elevation"] == pytest.approx(1200.0)
    assert ch["NDVI"] == pytest.approx(5100.0)          # raw x1e4, as NDWS stores it
    assert ch["tmmx_c"] == pytest.approx(32.0)
    assert ch["tmmn_c"] == pytest.approx(12.0)
    assert ch["vs"] == pytest.approx(4.0)
    assert ch["vs_cos_th"] == pytest.approx(-4.0)       # 4 * cos(180°)
    assert ch["pdsi"] == pytest.approx(-3.5)
    assert ch["sph"] == pytest.approx(0.006)
    assert ch["PrevFireMask"] == pytest.approx(1.0)
    assert ch["slope_deg"] == 0.0 and ch["aspect_deg"] == 0.0   # not in NDWS


def test_serving_patch_builder_fills_every_channel_by_name():
    """extract_patch can't run without Earth Engine, but its channel table
    must cover exactly SPREAD_CFG.feature_channels."""
    from wildfire import data_extraction
    src = inspect.getsource(data_extraction._extract_patch_uncached)
    for name in SPREAD_CFG.feature_channels:
        assert f'"{name}":' in src, f"serving patch never fills {name}"
    for dead in ("kbdi", "SMAP", "ERA5", "MOD13Q1"):
        assert dead not in src


def test_spread_patches_are_centred_on_active_fires_one_per_cluster():
    from wildfire.dashboard import _active_fire_centers

    rows = [(-120.00, 40.00, 1), (-120.01, 40.01, 1), (-120.02, 40.00, 1),   # cluster A (3)
            (-118.00, 36.00, 1),                                             # lone fire B
            (-121.00, 41.00, 0), (-117.0, 35.0, 0)]                          # not burning
    scored = pd.DataFrame(rows, columns=["longitude", "latitude", "fire_recent"])
    centers = _active_fire_centers(scored, top_k=10, patch_km=64.0)

    assert len(centers) == 2
    assert centers.iloc[0]["burning_cells"] == 3        # densest cluster first
    assert centers.iloc[1]["longitude"] == pytest.approx(-118.0)


def test_no_active_fires_means_no_spread_patches():
    from wildfire.dashboard import _active_fire_centers
    scored = pd.DataFrame({"longitude": [-120.0], "latitude": [40.0], "fire_recent": [0]})
    assert _active_fire_centers(scored, top_k=5, patch_km=64.0).empty
