"""
California study region.

The model scores California only. Two boundaries are used:

  * Earth Engine side: the US Census TIGER 2018 state polygon, filtered to
    California. Every GEE sample is drawn inside it, so fresh pulls never
    contain Nevada, Oregon or ocean cells.
  * Local side: the bundled county polygons (``ca_counties.geojson``, ~100 m
    simplified, from codeforgermany/click_that_hood). Used to cut older
    West Coast parquets down to California, to place synthetic points, to
    reject out-of-state service requests, and as the county lines on the map.

Point-in-polygon is a vectorized even-odd ray cast per county, with a bbox
pre-filter so each county only tests the points that could fall inside it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

COUNTIES_GEOJSON = Path(__file__).resolve().parent / "ca_counties.geojson"


@lru_cache(maxsize=1)
def counties() -> dict:
    return json.loads(COUNTIES_GEOJSON.read_text())


@lru_cache(maxsize=1)
def _polygons() -> tuple:
    """Each county polygon as (county index, bbox, [ring arrays]); rings[0] is
    the exterior."""
    out = []
    for i, feat in enumerate(counties()["features"]):
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            rings = [np.asarray(r, dtype=np.float64) for r in poly]
            ext = rings[0]
            bbox = (ext[:, 0].min(), ext[:, 1].min(), ext[:, 0].max(), ext[:, 1].max())
            out.append((i, bbox, rings))
    return tuple(out)


def _ring_crossings(x: np.ndarray, y: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Parity of ray crossings from each point toward +x against one ring."""
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    px, py = x[:, None], y[:, None]
    straddles = (y1 > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
    return (straddles & (px < x_at)).sum(axis=1) % 2 == 1


def county_names() -> list[str]:
    return [f["properties"]["name"] for f in counties()["features"]]


def locate(lon, lat) -> np.ndarray:
    """County index of each (lon, lat) point, or -1 outside California."""
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    county = np.full(lon.shape, -1, dtype=np.int16)
    for ci, (w, s, e, n), rings in _polygons():
        cand = np.flatnonzero((county < 0) & (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n))
        # Chunk so the (points x edges) work array stays small.
        for start in range(0, cand.size, 4096):
            idx = cand[start:start + 4096]
            hit = np.zeros(idx.size, dtype=bool)
            for ring in rings:  # holes flip parity back out
                hit ^= _ring_crossings(lon[idx], lat[idx], ring)
            county[idx[hit]] = ci
    return county


def contains(lon, lat) -> np.ndarray:
    """Boolean mask: which (lon, lat) points fall inside California."""
    return locate(lon, lat) >= 0


def restrict(df, lon_col: str = "longitude", lat_col: str = "latitude"):
    """Rows of ``df`` that fall inside California."""
    return df[contains(df[lon_col].to_numpy(), df[lat_col].to_numpy())]


def ee_geometry(ee):
    """California as an Earth Engine geometry (TIGER 2018 state boundary)."""
    return (ee.FeatureCollection("TIGER/2018/States")
              .filter(ee.Filter.eq("NAME", "California"))
              .geometry())
