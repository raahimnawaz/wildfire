"""The California boundary: what the pipeline keeps and what it drops."""

import numpy as np
import pandas as pd

from wildfire import region
from wildfire.config import CFG
from wildfire.synthetic_data import make_synthetic_dataset

INSIDE = {
    "San Francisco": (-122.42, 37.77),
    "Truckee": (-120.18, 39.33),
    "Needles (east edge)": (-114.61, 34.85),
    "Catalina Island": (-118.42, 33.39),
    "Crescent City (NW corner)": (-124.20, 41.76),
}
OUTSIDE = {
    "Reno, NV": (-119.81, 39.53),
    "Las Vegas, NV": (-115.14, 36.17),
    "Yuma, AZ": (-114.62, 32.69),
    "Medford, OR": (-122.87, 42.33),
    "Pacific off Big Sur": (-123.50, 36.00),
    "Tijuana, MX": (-117.03, 32.51),
}


def test_known_places_inside():
    lon, lat = np.array(list(INSIDE.values())).T
    assert region.contains(lon, lat).all()


def test_known_places_outside():
    """Every one of these sits inside the old lat/lon box."""
    lon, lat = np.array(list(OUTSIDE.values())).T
    assert not region.contains(lon, lat).any()


def test_restrict_keeps_only_california_rows():
    df = pd.DataFrame(
        [{"longitude": x, "latitude": y, "name": k} for k, (x, y) in {**INSIDE, **OUTSIDE}.items()]
    )
    assert set(region.restrict(df)["name"]) == set(INSIDE)


def test_synthetic_points_land_in_california():
    df = make_synthetic_dataset(n_total=2_000, seed=0)
    assert region.contains(df["longitude"], df["latitude"]).all()
    w, s, e, n = CFG.region_bbox
    assert df["longitude"].between(w, e).all() and df["latitude"].between(s, n).all()


def test_land_cells_matches_california_area():
    """~424k km2 of land at 4 km2 per cell."""
    assert 95_000 < region.land_cells(2000) < 110_000


def test_pages_follow_the_coverage_target():
    from wildfire.data_extraction import default_pages, pages_for_coverage

    m = 102_744
    assert pages_for_coverage(0.55, 4000, m) == 21
    assert pages_for_coverage(0.54, 4000, m) == 20
    assert pages_for_coverage(0.8, 4000, m) > pages_for_coverage(0.5, 4000, m)
    assert 18 <= default_pages() <= 24
