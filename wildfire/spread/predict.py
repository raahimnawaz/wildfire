"""
Score new patches with a trained spread model.

Reads .npy/.npz of input tensors (N, T, C, H, W) — or (N, C, H, W) which is
broadcast over T — and writes a .npz with predicted probabilities and binary
masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import SPREAD_CFG
from .model import build_model


def _load_model(device: torch.device):
    ckpt_path = SPREAD_CFG.model_path()
    if not ckpt_path.exists():
        raise SystemExit(
            f"No trained spread model at {ckpt_path}.\n"
            "Train one with:  python -m wildfire.spread.train --source synthetic"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["in_channels"], ckpt["lookback_days"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _load_inputs(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as d:
            if "X" in d:
                return d["X"]
            return d[d.files[0]]
    return np.load(path)


def _normalize_shape(x: np.ndarray, lookback: int) -> np.ndarray:
    if x.ndim == 4:                                  # (N, C, H, W)
        x = np.broadcast_to(x[:, None], (x.shape[0], lookback, *x.shape[1:])).copy()
    elif x.ndim == 5 and x.shape[1] != lookback:
        if x.shape[1] > lookback:
            x = x[:, -lookback:]
        else:
            pad = np.broadcast_to(x[:, :1], (x.shape[0], lookback - x.shape[1], *x.shape[2:]))
            x = np.concatenate([pad, x], axis=1)
    return x.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Score next-day fire spread.")
    parser.add_argument("--input",  type=Path, required=True,
                        help="Numpy file with input tensor (N,T,C,H,W) or (N,C,H,W).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to save the .npz with predictions.")
    parser.add_argument("--threshold", type=float, default=SPREAD_CFG.decision_threshold)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = _load_model(device)
    print(f"[predict] loaded {ckpt['arch']} model on {device}")

    raw = _load_inputs(args.input)
    x_full = _normalize_shape(raw, ckpt["lookback_days"])
    print(f"[predict] input shape: {x_full.shape}")

    probs = np.zeros((x_full.shape[0], 1, x_full.shape[-2], x_full.shape[-1]), dtype=np.float32)
    bs = SPREAD_CFG.batch_size
    with torch.no_grad():
        for i in range(0, x_full.shape[0], bs):
            chunk = torch.from_numpy(x_full[i:i + bs]).to(device)
            logits = model(chunk)
            probs[i:i + bs] = torch.sigmoid(logits).cpu().numpy()

    masks = (probs >= args.threshold).astype(np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, fire_probability=probs, fire_alert=masks)

    n_alert = int(masks.sum())
    print(f"[predict] wrote {args.output}  patches={probs.shape[0]}  fire_pixels={n_alert:,}")


if __name__ == "__main__":
    main()
