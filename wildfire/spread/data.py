"""
Datasets for the spread forecaster.

Three sources, in order of fidelity:

1. SyntheticSpreadDataset — physics-lite generator that simulates a fire
   front advancing under wind influence. Useful for plumbing smoke tests
   and CPU-only sanity runs. Channels match the real NDWS schema so models
   trained here transfer straight to real data.

2. NDWSNumpyDataset — loads the Next Day Wildfire Spread dataset from
   the common Kaggle redistribution as .npy/.npz tensors. This is the
   recommended path: real CONUS fires from 2012-2020, ~70k 64x64 patches,
   12 channels. Source paper: Huot et al., ICDM 2022.

3. NDWSTFRecordDataset — parses the original TFRecord shards from
   Google Research without taking a TensorFlow or `tfrecord` package
   dependency (the latter has a known protobuf conflict). Pure Python
   parser, ~80 lines, handles the 12-feature NDWS schema.

Channel order is fixed by SPREAD_CFG.feature_channels for ALL datasets.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .config import SPREAD_CFG
from .ndws import NDWS_FEATURE_MAP, NDWS_TFRECORD_FEATURES, ndws_channels  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------------
# Synthetic — physics-lite fire-front advection
# ---------------------------------------------------------------------------

class SyntheticSpreadDataset(Dataset):
    """
    Generates (T, C, H, W) input + (1, H, W) next-day fire mask pairs.

    Physics: a fire-front mask drifts each day in the direction of the wind
    vector (u, v), proportional to wind speed and inversely to soil moisture
    + NDVI, then dilates with a probabilistic kernel. The "next-day" target
    is the fire mask one step beyond the input window.

    Not a substitute for real data, but the spatial dynamics are closer to
    real fires than uniform noise and let the model learn meaningful spatial
    filters during smoke testing.
    """

    def __init__(self, n_samples: int = 512, seed: int = 0):
        self.n_samples = n_samples
        self.seed = seed
        self.size = SPREAD_CFG.patch_size
        self.T = SPREAD_CFG.lookback_days
        self.channels = SPREAD_CFG.feature_channels

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 100_003 + idx)
        H = W = self.size
        T = self.T

        # Wind drift (constant per patch — synoptic-scale stable for a few days)
        wind_u = rng.normal(0.0, 6.0)
        wind_v = rng.normal(0.0, 6.0)

        # Static channels
        elevation = rng.uniform(0, 3000)  + 50.0 * rng.standard_normal((H, W))
        slope     = np.abs(rng.normal(8, 6, (H, W))).clip(0, 60)
        aspect    = rng.uniform(0, 360, (H, W))
        ndvi      = np.clip(rng.normal(0.4, 0.15, (H, W)), -0.1, 0.95)
        kbdi      = np.full((H, W), rng.uniform(300, 780))
        soil_m    = np.clip(rng.beta(2, 5, (H, W)), 0.0, 1.0)

        # Fire mask seed — start with a small kernel
        mask = np.zeros((H, W), dtype=np.float32)
        cy, cx = rng.integers(H // 4, 3 * H // 4), rng.integers(W // 4, 3 * W // 4)
        mask[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 1.0

        prev_masks: List[np.ndarray] = []
        # roll forward T+1 days; first T are inputs, the (T+1)-th is target
        for _ in range(T + 1):
            prev_masks.append(mask.copy())
            mask = self._advance_fire(mask, wind_u, wind_v, ndvi, soil_m, kbdi, rng)

        # Build (T, C, H, W)
        temp = float(rng.uniform(15.0, 40.0))
        dewp = temp - float(rng.uniform(5.0, 25.0))

        # Expressed in the NDWS channel semantics (see ndws.py) so synthetic
        # and real batches share one schema: wind as speed and speed*cos(from-
        # direction) (= -v), drought as a PDSI-like index, NDVI in x1e4 units.
        speed = float(np.hypot(wind_u, wind_v))
        pdsi = -(kbdi - 400.0) / 60.0
        per_step = []
        for t in range(T):
            channels_dict = {
                "elevation":       elevation,
                "slope_deg":       slope,
                "aspect_deg":      aspect,
                "NDVI":            ndvi * 10000.0,
                "tmmx_c":          np.full((H, W), temp + 0.5 * t),
                "tmmn_c":          np.full((H, W), dewp),
                "vs":              np.full((H, W), speed),
                "vs_cos_th":       np.full((H, W), -wind_v),
                "pdsi":            pdsi,
                "sph":             np.full((H, W), 0.002 + 0.01 * (1.0 - soil_m.mean())),
                "PrevFireMask":    prev_masks[t],
            }
            stack = np.stack([channels_dict[name] for name in self.channels], axis=0)
            per_step.append(stack)
        x = np.stack(per_step, axis=0).astype(np.float32)  # (T, C, H, W)

        # Target = fire mask T+1
        target = (prev_masks[T] > 0.5).astype(np.float32)[None]  # (1, H, W)
        return torch.from_numpy(x), torch.from_numpy(target)

    @staticmethod
    def _advance_fire(mask: np.ndarray,
                      wind_u: float, wind_v: float,
                      ndvi: np.ndarray, soil_m: np.ndarray, kbdi: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
        H, W = mask.shape

        # Shift mask by integer wind step (downstream propagation)
        dx = int(np.clip(round(wind_u / 4.0), -2, 2))
        dy = int(np.clip(round(wind_v / 4.0), -2, 2))
        shifted = np.zeros_like(mask)
        ys = slice(max(0, dy), H + min(0, dy))
        xs = slice(max(0, dx), W + min(0, dx))
        ys_src = slice(max(0, -dy), H - max(0, dy))
        xs_src = slice(max(0, -dx), W - max(0, dx))
        shifted[ys, xs] = mask[ys_src, xs_src]

        # Probabilistic dilation in 8-neighborhood, gated by fuel & dryness
        from scipy.ndimage import binary_dilation
        candidate = binary_dilation(shifted > 0.5, iterations=1).astype(np.float32)
        ignite_prob = (
            0.35
            + 0.5 * np.clip(ndvi, 0.0, 1.0)
            - 0.4 * soil_m
            + 0.0008 * (kbdi - 400.0)
        )
        ignite_prob = np.clip(ignite_prob, 0.05, 0.95)
        rolls = rng.uniform(0, 1, mask.shape)
        new_mask = np.maximum(shifted, candidate * (rolls < ignite_prob).astype(np.float32))
        return new_mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Real NDWS (numpy redistribution) — drop-in replacement
# ---------------------------------------------------------------------------

class NDWSNumpyDataset(Dataset):
    """
    Loads the Next Day Wildfire Spread dataset from a numpy-formatted
    redistribution. The expected on-disk layout is one of:

      <data_dir>/X.npy (shape (N, C, H, W) or (N, T, C, H, W))
      <data_dir>/y.npy (shape (N, 1, H, W))

    or sharded:

      <data_dir>/shard_*.npz   each with arrays "X" and "y" of matching shape.

    If data is provided as (N, C, H, W) (single-day input, original NDWS
    schema) it's automatically replicated across the lookback window so the
    ConvLSTM gets a (T, C, H, W) tensor.
    """

    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"NDWS data dir not found: {data_dir}\n"
                "Download from Kaggle (search 'next-day-wildfire-spread') and "
                "either save X.npy/y.npy here, or use the TFRecord -> npy "
                "converter described in wildfire/spread/README.md."
            )

        self.X_path = data_dir / "X.npy"
        self.y_path = data_dir / "y.npy"

        if self.X_path.exists() and self.y_path.exists():
            self.X = np.load(self.X_path, mmap_mode="r")
            self.y = np.load(self.y_path, mmap_mode="r")
            self._sharded = False
        else:
            shards = sorted(data_dir.glob("shard_*.npz"))
            if not shards:
                raise FileNotFoundError(
                    f"No X.npy/y.npy or shard_*.npz files in {data_dir}."
                )
            self.shards = shards
            self._sharded = True
            # Build a flat index over all shards
            self._index: List[Tuple[int, int]] = []
            for shard_id, sp in enumerate(shards):
                with np.load(sp) as data:
                    n = data["y"].shape[0]
                    self._index.extend((shard_id, i) for i in range(n))

        self.lookback = SPREAD_CFG.lookback_days

    def __len__(self) -> int:
        if self._sharded:
            return len(self._index)
        return int(self.X.shape[0])

    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._sharded:
            shard_id, i = self._index[idx]
            with np.load(self.shards[shard_id]) as data:
                return np.asarray(data["X"][i]), np.asarray(data["y"][i])
        return np.asarray(self.X[idx]), np.asarray(self.y[idx])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._load_pair(idx)

        # Normalize x to (T, C, H, W)
        if x.ndim == 3:                  # (C, H, W) -> replicate over T
            x = np.broadcast_to(x[None], (self.lookback, *x.shape)).copy()
        elif x.ndim == 4 and x.shape[0] != self.lookback:
            # Take last `lookback` timesteps if too long, pad-replicate if too short
            if x.shape[0] > self.lookback:
                x = x[-self.lookback:]
            else:
                pad = np.broadcast_to(x[:1], (self.lookback - x.shape[0], *x.shape[1:]))
                x = np.concatenate([pad, x], axis=0)

        # Normalize y to (1, H, W)
        if y.ndim == 2:
            y = y[None]
        y = (y > 0.5).astype(np.float32)

        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Original Google TFRecord loader — minimal dependency-free parser
# ---------------------------------------------------------------------------

def _read_tfrecord_messages(path: Path) -> Iterator[bytes]:
    """Yield raw protobuf-encoded record bytes from a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (length,) = struct.unpack("<Q", head)
            f.read(4)                          # length CRC32C — skip
            payload = f.read(length)
            if len(payload) < length:
                raise IOError(f"Truncated record in {path}")
            f.read(4)                          # data CRC32C — skip
            yield payload


def _parse_example_floats(record: bytes) -> dict:
    """
    Minimal protobuf parser for tf.train.Example wrapping float_list tensors.

    Wire format we need:
      Example: features (1) -> Features: feature (1, map<string, Feature>) ->
        Feature: float_list (2) -> FloatList: value (1, repeated fixed32)

    We walk the wire format manually and pull out per-feature float arrays.
    """
    out: dict = {}
    buf = io.BytesIO(record)

    def read_varint(b):
        shift = 0
        val = 0
        while True:
            byte = b.read(1)
            if not byte:
                return None
            byte = byte[0]
            val |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                return val
            shift += 7

    def read_tag(b):
        v = read_varint(b)
        if v is None:
            return None, None
        return v >> 3, v & 0x07

    def parse_features(bb: bytes):
        bs = io.BytesIO(bb)
        while True:
            tag, wire = read_tag(bs)
            if tag is None:
                return
            if tag == 1 and wire == 2:    # feature map entry
                ln = read_varint(bs)
                entry = bs.read(ln)
                parse_feature_map_entry(entry)
            else:
                _skip(bs, wire)

    def parse_feature_map_entry(bb: bytes):
        bs = io.BytesIO(bb)
        key = None
        feature = None
        while True:
            tag, wire = read_tag(bs)
            if tag is None:
                break
            if tag == 1 and wire == 2:
                ln = read_varint(bs)
                key = bs.read(ln).decode("utf-8")
            elif tag == 2 and wire == 2:
                ln = read_varint(bs)
                feature = bs.read(ln)
            else:
                _skip(bs, wire)
        if key is None or feature is None:
            return
        # Feature has float_list (tag 2)
        fbs = io.BytesIO(feature)
        floats: List[float] = []
        while True:
            tag, wire = read_tag(fbs)
            if tag is None:
                break
            if tag == 2 and wire == 2:    # float_list
                ln = read_varint(fbs)
                fl = fbs.read(ln)
                fbs2 = io.BytesIO(fl)
                while True:
                    t2, w2 = read_tag(fbs2)
                    if t2 is None:
                        break
                    if t2 == 1 and w2 == 2:
                        ln2 = read_varint(fbs2)
                        chunk = fbs2.read(ln2)
                        floats.extend(struct.unpack(f"<{ln2 // 4}f", chunk))
                    else:
                        _skip(fbs2, w2)
            else:
                _skip(fbs, wire)
        out[key] = np.asarray(floats, dtype=np.float32)

    def _skip(bs, wire):
        if wire == 0:
            read_varint(bs)
        elif wire == 1:
            bs.read(8)
        elif wire == 2:
            ln = read_varint(bs)
            bs.read(ln)
        elif wire == 5:
            bs.read(4)
        else:
            raise ValueError(f"Unsupported wire type {wire}")

    while True:
        tag, wire = read_tag(buf)
        if tag is None:
            break
        if tag == 1 and wire == 2:    # features
            ln = read_varint(buf)
            features_bytes = buf.read(ln)
            parse_features(features_bytes)
        else:
            _skip(buf, wire)
    return out


class NDWSTFRecordDataset(IterableDataset):
    """Iterable dataset over the original Google NDWS TFRecord shards."""

    def __init__(self, data_dir: Path, patch_size: int = 64):
        self.shards = sorted(Path(data_dir).glob("*.tfrecord*"))
        if not self.shards:
            raise FileNotFoundError(
                f"No *.tfrecord files in {data_dir}. Download from "
                "Kaggle 'next-day-wildfire-spread' and unpack the shards here."
            )
        self.patch_size = patch_size
        self.lookback = SPREAD_CFG.lookback_days
        self.channels = SPREAD_CFG.feature_channels

    def _to_xy(self, parsed: dict) -> Tuple[np.ndarray, np.ndarray]:
        s = self.patch_size
        x_2d = ndws_channels(parsed, s)                  # (C, H, W)
        x = np.broadcast_to(x_2d[None], (self.lookback, *x_2d.shape)).copy()  # (T, C, H, W)

        fire = parsed.get("FireMask")
        if fire is None:
            raise KeyError("TFRecord example missing FireMask")
        y = (fire.reshape(s, s) > 0.5).astype(np.float32)[None]
        return x.astype(np.float32), y

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for shard in self.shards:
            for record in _read_tfrecord_messages(shard):
                parsed = _parse_example_floats(record)
                if "FireMask" not in parsed:
                    continue
                x, y = self._to_xy(parsed)
                yield torch.from_numpy(x), torch.from_numpy(y)


def build_dataset(source: str, data_dir: Optional[Path] = None,
                  n_samples: int = 512, seed: int = 0) -> Dataset:
    """Factory dispatch — keeps train.py simple."""
    src = source.lower()
    if src == "synthetic":
        return SyntheticSpreadDataset(n_samples=n_samples, seed=seed)
    if src == "ndws-numpy":
        if data_dir is None:
            raise ValueError("ndws-numpy requires --data-dir")
        return NDWSNumpyDataset(data_dir=data_dir)
    if src == "ndws-tfrecord":
        if data_dir is None:
            raise ValueError("ndws-tfrecord requires --data-dir")
        return NDWSTFRecordDataset(data_dir=data_dir)
    raise ValueError(f"Unknown source '{source}'. Use 'synthetic', 'ndws-numpy', 'ndws-tfrecord'.")


def build_split_datasets(data_dir: Path) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load NDWS data that's been pre-split into train/val/test subdirs.
    Each subdir must contain X.npy and y.npy. Honors NDWS's official split.
    """
    data_dir = Path(data_dir)
    splits = {}
    for name in ("train", "val", "test"):
        sub = data_dir / name
        if not (sub / "X.npy").exists():
            raise FileNotFoundError(
                f"Missing {sub}/X.npy. Run:\n"
                f"  python -m wildfire.spread.convert_ndws --in-dir <raw tfrecords> --out-dir {data_dir}"
            )
        splits[name] = NDWSNumpyDataset(data_dir=sub)
    return splits["train"], splits["val"], splits["test"]
