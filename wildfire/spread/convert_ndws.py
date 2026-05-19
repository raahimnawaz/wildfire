"""
Convert NDWS TFRecord shards to numpy .npy files, respecting the official
train / eval / test split that ships with the dataset.

Output layout:
    <out_dir>/train/X.npy, y.npy
    <out_dir>/val/X.npy,   y.npy   (eval shards)
    <out_dir>/test/X.npy,  y.npy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np

from .config import SPREAD_CFG
from .data import _read_tfrecord_messages, _parse_example_floats, NDWS_FEATURE_MAP


def _shard_split(name: str) -> str:
    m = re.search(r"_(train|eval|test)_", name)
    return m.group(1) if m else "train"


def _convert_shards(shards: List[Path]) -> tuple[np.ndarray, np.ndarray]:
    s = SPREAD_CFG.patch_size
    channels = SPREAD_CFG.feature_channels
    xs, ys = [], []
    for shard in shards:
        for record in _read_tfrecord_messages(shard):
            parsed = _parse_example_floats(record)
            if "FireMask" not in parsed:
                continue
            try:
                fire = parsed["FireMask"].reshape(s, s)
            except ValueError:
                continue
            stack = []
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
                stack.append(arr)
            xs.append(np.stack(stack, axis=0).astype(np.float32))   # (C, H, W)
            # FireMask: paper uses {-1, 0, 1}; treat anything > 0 as fire
            ys.append((fire > 0.5).astype(np.float32)[None])         # (1, H, W)
    return np.stack(xs), np.stack(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir",  type=Path, default=SPREAD_CFG.data_dir / "ndws_raw")
    parser.add_argument("--out-dir", type=Path, default=SPREAD_CFG.data_dir / "ndws_npy")
    args = parser.parse_args()

    all_shards = sorted(args.in_dir.glob("*.tfrecord*"))
    if not all_shards:
        raise SystemExit(f"No TFRecord shards in {args.in_dir}")

    bucket = {"train": [], "eval": [], "test": []}
    for s in all_shards:
        bucket[_shard_split(s.name)].append(s)

    for split_name, shards in bucket.items():
        if not shards:
            continue
        out_split = args.out_dir / ("val" if split_name == "eval" else split_name)
        out_split.mkdir(parents=True, exist_ok=True)
        print(f"[ndws] {split_name}: {len(shards)} shards -> {out_split}")
        X, y = _convert_shards(shards)
        np.save(out_split / "X.npy", X)
        np.save(out_split / "y.npy", y)
        print(f"        X: {X.shape}  y: {y.shape}  "
              f"fire_pixel_rate: {(y > 0.5).mean():.4f}")

    print("\n[ndws] all splits written.")


if __name__ == "__main__":
    main()
