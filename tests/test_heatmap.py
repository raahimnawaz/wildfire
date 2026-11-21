"""The map page builds from a scored frame and only shows California."""

import numpy as np

from wildfire import heatmap
from wildfire.synthetic_data import make_synthetic_dataset


def _scored(n=3_000):
    df = make_synthetic_dataset(n_total=n, seed=1)
    rng = np.random.default_rng(1)
    df["fire_probability"] = rng.beta(0.6, 12, len(df))
    df["fire_alert"] = (df["fire_probability"] >= 0.16).astype(int)
    df["wind_speed_ms"] = 3.0
    return df


def test_rasterize_grid_is_california_only():
    grid = heatmap.rasterize(_scored())
    painted = ~np.isnan(grid["p"])
    assert painted.any()
    # every painted cell is a sampled cell or a land cell filled from one
    assert ((grid["county"] >= 0) | (grid["flag"] == 1))[painted].all()
    assert set(np.unique(grid["flag"][painted])) <= {1, 2}


def test_render_writes_a_self_contained_page(tmp_path):
    out = tmp_path / "map.html"
    heatmap.render(_scored(), meta={}, target_date="2025-09-15",
                   summary={"operational_threshold": 0.16}, out_html=out)
    page = out.read_text()
    assert "California Wildfire Risk — 2025-09-15" in page
    assert "trained on West Coast (CA, OR, WA) samples" in page  # legacy meta, no region recorded
    assert out.stat().st_size < 2_000_000


def test_ramp_is_anchored_to_threshold():
    assert heatmap.ramp_color(0.16, 0.16) == "#ffc43d"
    assert heatmap.ramp_color(0.0, 0.16) == "#0b1e36"
    assert heatmap.ramp_color(0.9, 0.16) == "#ff2d4a"
