"""
End-to-end real-data runner.

After you have set up Kaggle and/or GEE credentials, run:

    python -m wildfire.run_real

This script auto-detects which credentials are present and runs:
  * Kaggle (NDWS spread data)  -> download + train U-Net for next-day spread
  * GEE (West Coast ignition)  -> extract 2 fire seasons + retrain XGBoost

Anything not authenticated is skipped with a clear message of what you need
to do. The script is idempotent — re-running picks up where it left off.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from .config import CFG
from .spread.config import SPREAD_CFG


KAGGLE_DATASET = "fantineh/next-day-wildfire-spread"
NDWS_RAW_DIR   = SPREAD_CFG.data_dir / "ndws_raw"
NDWS_NPY_DIR   = SPREAD_CFG.data_dir / "ndws_npy"


# ---------------------------------------------------------------------------
# Credential checks
# ---------------------------------------------------------------------------

def _kaggle_ok() -> bool:
    home = Path.home()
    candidates = [
        home / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"
            if os.environ.get("KAGGLE_CONFIG_DIR") else None,
    ]
    for p in candidates:
        if p and p.exists():
            return True
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _gee_ok() -> bool:
    if not os.environ.get("GEE_PROJECT"):
        return False
    home = Path.home()
    creds = [
        home / ".config" / "earthengine" / "credentials",
        home / ".config" / "earthengine" / "credentials.json",
    ]
    return any(p.exists() for p in creds)


# ---------------------------------------------------------------------------
# Kaggle / NDWS pipeline
# ---------------------------------------------------------------------------

def _kaggle_download_ndws() -> None:
    NDWS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = NDWS_RAW_DIR / "ndws.zip"

    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        print(f"[kaggle] zip already at {zip_path} ({zip_path.stat().st_size/1e9:.2f} GB) — skip download")
    else:
        print(f"[kaggle] downloading {KAGGLE_DATASET} -> {NDWS_RAW_DIR}")
        # Defer import; kaggle triggers auth on import
        cmd = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET, "-p", str(NDWS_RAW_DIR), "--force",
        ]
        subprocess.run(cmd, check=True)
        # Find the resulting zip (kaggle CLI names it after the dataset slug)
        zips = list(NDWS_RAW_DIR.glob("*.zip"))
        if not zips:
            raise RuntimeError("Kaggle download produced no zip file.")
        zips[0].rename(zip_path)

    extract_marker = NDWS_RAW_DIR / ".extracted"
    if not extract_marker.exists():
        print(f"[kaggle] unpacking {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(NDWS_RAW_DIR)
        extract_marker.write_text("ok")
    else:
        print(f"[kaggle] already unpacked")


def _convert_ndws_to_numpy() -> None:
    """
    Walk NDWS_RAW_DIR for TFRecord shards and convert into X.npy + y.npy
    using our dependency-free parser. The Kaggle redistribution we target
    (fantineh/next-day-wildfire-spread) ships TFRecord shards.
    """
    if (NDWS_NPY_DIR / "X.npy").exists() and (NDWS_NPY_DIR / "y.npy").exists():
        print(f"[ndws] {NDWS_NPY_DIR}/X.npy already exists — skip conversion")
        return

    NDWS_NPY_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(NDWS_RAW_DIR.rglob("*.tfrecord*"))
    if not shards:
        # Some Kaggle copies ship .npy directly — accept that too
        npys = list(NDWS_RAW_DIR.rglob("*.npy"))
        if any(p.name == "X.npy" for p in npys) and any(p.name == "y.npy" for p in npys):
            print("[ndws] found pre-converted X.npy / y.npy in raw dir; copying")
            for p in npys:
                if p.name in ("X.npy", "y.npy"):
                    shutil.copy(p, NDWS_NPY_DIR / p.name)
            return
        raise RuntimeError(
            f"No TFRecord shards or X.npy/y.npy in {NDWS_RAW_DIR}. "
            "The Kaggle dataset layout may have changed — inspect manually."
        )

    from .spread.data import _read_tfrecord_messages, _parse_example_floats, NDWS_FEATURE_MAP

    s = SPREAD_CFG.patch_size
    channels = SPREAD_CFG.feature_channels
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    print(f"[ndws] converting {len(shards)} shards to numpy ...")
    for shard_idx, shard in enumerate(shards):
        for record in _read_tfrecord_messages(shard):
            parsed = _parse_example_floats(record)
            if "FireMask" not in parsed:
                continue
            try:
                fire = parsed["FireMask"].reshape(s, s)
            except ValueError:
                continue
            channel_stack = []
            for ch in channels:
                arr = np.zeros((s, s), dtype=np.float32)
                for tf_name, our_name in NDWS_FEATURE_MAP.items():
                    if our_name == ch and tf_name in parsed:
                        try:
                            arr = parsed[tf_name].reshape(s, s).astype(np.float32)
                        except ValueError:
                            continue
                        if tf_name in ("tmmx", "tmmn"):
                            arr = arr - 273.15
                        if tf_name == "th":
                            speed = parsed.get("vs", np.zeros_like(arr)).reshape(s, s)
                            arr = speed * np.cos(np.deg2rad(arr))
                        break
                channel_stack.append(arr)
            x = np.stack(channel_stack, axis=0).astype(np.float32)   # (C, H, W)
            y = (fire > 0.5).astype(np.float32)[None]                # (1, H, W)
            xs.append(x)
            ys.append(y)
        if (shard_idx + 1) % 5 == 0:
            print(f"  shard {shard_idx + 1}/{len(shards)}  cumulative samples: {len(xs):,}")

    if not xs:
        raise RuntimeError("No samples extracted from TFRecords.")
    X = np.stack(xs)
    y = np.stack(ys)
    np.save(NDWS_NPY_DIR / "X.npy", X)
    np.save(NDWS_NPY_DIR / "y.npy", y)
    print(f"[ndws] wrote X.npy {X.shape} and y.npy {y.shape} -> {NDWS_NPY_DIR}")


def _train_spread() -> None:
    print("\n[spread] training U-Net on real NDWS data ...")
    cmd = [
        sys.executable, "-m", "wildfire.spread.train",
        "--source", "ndws-numpy",
        "--data-dir", str(NDWS_NPY_DIR),
        "--arch", "unet",
        "--epochs", "30",
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# GEE / ignition pipeline
# ---------------------------------------------------------------------------

def _gee_extract_year(year: int, months: tuple | None = None, suffix: str = "") -> Path:
    out = CFG.data_dir / f"raw_{year}{suffix}.parquet"
    if out.exists() and out.stat().st_size > 1000:
        print(f"[gee] {out} already exists — skip extraction")
        return out
    label = "shoulder-season" if months else "fire-season"
    print(f"[gee] extracting {label} {year} (this takes ~10 min) ...")
    cmd = [sys.executable, "-m", "wildfire.data_extraction", "--year", str(year), "--out", str(out)]
    if months:
        cmd += ["--months", ",".join(str(m) for m in months)]
    subprocess.run(cmd, check=True)
    return out


def _train_ignition(train_paths: list[Path], val_paths: list[Path], test_paths: list[Path]) -> None:
    print("\n[ignition] training XGBoost on real GEE data "
          "(threshold tuned on validation year, production refit on all years) ...")
    cmd = [
        sys.executable, "-m", "wildfire.train",
        "--train", *[str(p) for p in train_paths],
        "--val",   *[str(p) for p in val_paths],
        "--test",  *[str(p) for p in test_paths],
        "--refit-all",
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    did_anything = False

    print("=== Wildfire real-data runner ===\n")

    if _kaggle_ok():
        print("[ok] Kaggle credentials detected -> running spread pipeline")
        try:
            _kaggle_download_ndws()
            _convert_ndws_to_numpy()
            _train_spread()
            did_anything = True
        except subprocess.CalledProcessError as e:
            print(f"[!] Kaggle pipeline step failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[!] Kaggle pipeline error: {e}", file=sys.stderr)
    else:
        print("[skip] Kaggle credentials not found.")
        print("       To enable the spread pipeline:")
        print("         1. Sign up at https://www.kaggle.com")
        print("         2. kaggle.com/settings -> Create New API Token (downloads kaggle.json)")
        print("         3. Move the file to %USERPROFILE%\\.kaggle\\kaggle.json")
        print(f"         4. Visit https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print("            and click 'I Understand and Accept' on the dataset terms.")
        print("         5. Re-run:  python -m wildfire.run_real\n")

    if _gee_ok():
        print("\n[ok] Earth Engine credentials detected -> running ignition pipeline")
        try:
            # Extract every configured year: Aug-Oct fire season plus the
            # May-Jul shoulder season so the model covers May-October. The
            # last year is the test holdout, second-to-last the validation
            # year for threshold tuning; production refits on everything.
            season   = [_gee_extract_year(y) for y in CFG.data_years]
            shoulder = [_gee_extract_year(y, months=(5, 6, 7), suffix="_preseason")
                        for y in CFG.data_years]
            _train_ignition(train_paths=season[:-2] + shoulder[:-2],
                            val_paths=[season[-2], shoulder[-2]],
                            test_paths=[season[-1], shoulder[-1]])
            did_anything = True
        except subprocess.CalledProcessError as e:
            print(f"[!] GEE pipeline step failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[!] GEE pipeline error: {e}", file=sys.stderr)
    else:
        print("\n[skip] Earth Engine credentials not found.")
        print("       To enable the ignition pipeline:")
        print("         1. Create a Google Cloud project: https://console.cloud.google.com/projectcreate")
        print("         2. Register for Earth Engine: https://signup.earthengine.google.com/")
        print("         3. Run:  earthengine authenticate    (opens browser)")
        print("         4. Set the project id for this shell:")
        print("              PowerShell:  $env:GEE_PROJECT = 'your-project-id'")
        print("              bash:        export GEE_PROJECT=your-project-id")
        print("         5. Verify:  python -m wildfire.setup_gee")
        print("         6. Re-run:  python -m wildfire.run_real")

    if not did_anything:
        print("\nNothing ran — set up at least one set of credentials above.")
        return 1
    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
