"""
Disk cache for GEE-pulled results.

The Earth Engine API has rate limits and quotas, and most dashboard re-runs
hit the same date windows + the same cell coordinates. This module keeps a
tiny content-addressed cache under wildfire_data/cache/ keyed by the call
arguments. First call fills the cache; subsequent calls read from disk.

Two storage formats:
  * Parquet (pandas DataFrames) — for sampled-feature tables
  * NumPy   (np.ndarray)        — for sampled imagery patches

The cache is intentionally simple — no expiry, no LRU eviction. Cache files
are small (~2 MB per window, ~50 KB per patch); if you ever want a clean
pull just `Remove-Item wildfire_data/cache -Recurse -Force`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import CFG


CACHE_DIR = CFG.data_dir / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash(parts: dict) -> str:
    """Stable 12-char hex digest of the call arguments."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def key_window(day_start: str, day_end: str, scale_m: int, pages: int,
               pixels_per_page: int) -> str:
    h = _hash({
        "kind": "window",
        "day_start": day_start, "day_end": day_end,
        "scale_m": scale_m, "pages": pages, "pixels_per_page": pixels_per_page,
    })
    return f"window_{day_start}_{day_end}_{scale_m}m_{pages}p_{h}"


def key_patch(lon: float, lat: float, day_start: str, day_end: str,
              size_px: int, pixel_km: float) -> str:
    h = _hash({
        "kind": "patch",
        "lon": round(lon, 4), "lat": round(lat, 4),
        "day_start": day_start, "day_end": day_end,
        "size_px": size_px, "pixel_km": pixel_km,
    })
    return f"patch_{lon:.4f}_{lat:.4f}_{day_start}_{int(pixel_km*1000)}m_{h}"


def get_or_compute_parquet(
    key: str,
    compute_fn: Callable[[], pd.DataFrame],
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return cached parquet if present, else run compute_fn and write."""
    path = CACHE_DIR / f"{key}.parquet"
    if use_cache and path.exists():
        if verbose:
            print(f"  [cache hit ] {path.name}")
        return pd.read_parquet(path)
    if verbose:
        print(f"  [cache fill] {path.name}")
    df = compute_fn()
    if df is not None and not df.empty:
        df.to_parquet(path, index=False)
    return df


def get_or_compute_npy(
    key: str,
    compute_fn: Callable[[], np.ndarray],
    use_cache: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Return cached numpy array if present, else run compute_fn and write."""
    path = CACHE_DIR / f"{key}.npy"
    if use_cache and path.exists():
        if verbose:
            print(f"  [cache hit ] {path.name}")
        return np.load(path)
    if verbose:
        print(f"  [cache fill] {path.name}")
    arr = compute_fn()
    if arr is not None:
        np.save(path, arr)
    return arr


def clear_cache(prefix: Optional[str] = None) -> int:
    """Delete cache files. Optionally filter to a prefix like 'window_' or 'patch_'."""
    removed = 0
    for p in CACHE_DIR.iterdir():
        if not p.is_file():
            continue
        if prefix and not p.name.startswith(prefix):
            continue
        p.unlink()
        removed += 1
    return removed


def cache_summary() -> dict:
    """Quick stats — used by the CLI to show what's been cached."""
    files = list(CACHE_DIR.iterdir())
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())
    n_window = sum(1 for p in files if p.name.startswith("window_"))
    n_patch  = sum(1 for p in files if p.name.startswith("patch_"))
    return {
        "cache_dir": str(CACHE_DIR),
        "n_files": len(files),
        "n_windows": n_window,
        "n_patches": n_patch,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }
