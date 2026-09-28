"""
Next Day Wildfire Spread (NDWS) feature -> model channel mapping.

This is the one place that decides what each spread-model channel holds. The
TFRecord dataset, the numpy converters, and (by the same channel names) the
Earth Engine serving patch in ``wildfire.data_extraction.extract_patch`` all
follow it. Kept free of torch so the mapping can be tested and used by the
converters on their own.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import SPREAD_CFG

# NDWS feature schema in the original Google Research TFRecords.
# Source: github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread
NDWS_TFRECORD_FEATURES: Tuple[str, ...] = (
    "elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr",
    "pdsi", "NDVI", "population", "erc", "PrevFireMask", "FireMask",
)

# Map TFRecord feature name -> SPREAD_CFG.feature_channels name.
# erc, pr, population: dropped. slope_deg / aspect_deg: not in NDWS -> zeros.
NDWS_FEATURE_MAP = {
    "elevation":    "elevation",
    "NDVI":         "NDVI",
    "tmmx":         "tmmx_c",       # K -> °C in ndws_channels
    "tmmn":         "tmmn_c",       # K -> °C in ndws_channels
    "vs":           "vs",
    "th":           "vs_cos_th",    # combined with vs in ndws_channels
    "pdsi":         "pdsi",
    "sph":          "sph",
    "PrevFireMask": "PrevFireMask",
}


def ndws_channels(parsed: dict, s: int) -> np.ndarray:
    """(C, s, s) float32 stack in SPREAD_CFG.feature_channels order from one
    parsed NDWS example. Channels NDWS doesn't supply are zeros."""
    by_channel = {ours: tf for tf, ours in NDWS_FEATURE_MAP.items()}
    stack = []
    for ch in SPREAD_CFG.feature_channels:
        arr = np.zeros((s, s), dtype=np.float32)
        tf_name = by_channel.get(ch)
        if tf_name is not None and tf_name in parsed:
            try:
                arr = parsed[tf_name].reshape(s, s).astype(np.float32)
            except ValueError:
                arr = np.zeros((s, s), dtype=np.float32)
            else:
                if tf_name in ("tmmx", "tmmn"):
                    arr = arr - 273.15
                elif tf_name == "th":
                    speed = parsed.get("vs", np.zeros(s * s, dtype=np.float32)).reshape(s, s)
                    arr = speed * np.cos(np.deg2rad(arr))
        stack.append(arr.astype(np.float32))
    return np.stack(stack, axis=0)
