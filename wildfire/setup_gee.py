"""
Earth Engine connectivity check + tiny end-to-end smoke pull.

Run this AFTER you have:
  1. pip install earthengine-api
  2. earthengine authenticate          (browser-based, one-time)
  3. set GEE_PROJECT=your-gcp-project-id   (PowerShell: $env:GEE_PROJECT="...")

It will:
  - confirm earthengine-api is importable,
  - try to ee.Initialize with your project,
  - sample 100 pixels of ERA5-Land + FIRMS over a tiny LA-area bbox for
    one recent week, and print the result.

If this works, you're ready to run:
    python -m wildfire.data_extraction --year 2022
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta


def main() -> int:
    print("[gee] checking earthengine-api ...")
    try:
        import ee
    except ImportError:
        print("  X earthengine-api is not installed.\n"
              "    Run:  pip install earthengine-api", file=sys.stderr)
        return 2

    project = os.environ.get("GEE_PROJECT")
    if not project:
        print("  ! GEE_PROJECT env var not set.\n"
              "    Set it to your Google Cloud project id:\n"
              "      PowerShell:  $env:GEE_PROJECT = 'your-project-id'\n"
              "      bash:        export GEE_PROJECT=your-project-id",
              file=sys.stderr)
        return 2

    print(f"[gee] initializing with project '{project}' ...")
    try:
        ee.Initialize(project=project)
    except Exception as e:
        print(f"  X ee.Initialize failed: {e}\n"
              "    Did you run 'earthengine authenticate' in this user account?",
              file=sys.stderr)
        return 3

    # Tiny bbox over Los Angeles area, last week
    geom = ee.Geometry.Rectangle([-118.5, 33.9, -118.0, 34.3])
    end  = datetime.utcnow().date()
    start = end - timedelta(days=7)

    print(f"[gee] pulling 100 sample pixels for {start} .. {end} over LA bbox ...")
    try:
        era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                  .filterDate(str(start), str(end))
                  .filterBounds(geom)
                  .select(["temperature_2m", "u_component_of_wind_10m"])
                  .mean())
        firms = (ee.ImageCollection("FIRMS")
                   .filterDate(str(start), str(end))
                   .filterBounds(geom)
                   .select("T21")
                   .count()
                   .gt(0)
                   .rename("fire_occurred")
                   .unmask(0))
        stack = era5.addBands(firms)
        sample = stack.sample(region=geom, scale=4000, numPixels=100,
                              seed=1, tileScale=4, geometries=False, dropNulls=True)
        feats = sample.getInfo()["features"]
    except Exception as e:
        print(f"  X GEE query failed: {e}", file=sys.stderr)
        return 4

    print(f"[gee] OK — got {len(feats)} sampled pixels.")
    if feats:
        print("[gee] sample row:", feats[0]["properties"])
    print("\n[gee] You're ready. Next step:")
    print("    python -m wildfire.data_extraction --year 2022 \\")
    print("        --out wildfire_data/raw_2022.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
