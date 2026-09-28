"""
Train the next-day fire-spread forecaster.

Usage:
    # Smoke test on synthetic patches (no data download required)
    python -m wildfire.spread.train --source synthetic --arch convlstm --epochs 5

    # Train on real Next Day Wildfire Spread data (Kaggle, numpy redistribution)
    python -m wildfire.spread.train --source ndws-numpy \\
        --data-dir wildfire_data/spread/ndws_npy \\
        --arch convlstm --epochs 20

    # U-Net baseline matching the original NDWS paper (Huot et al., ICDM 2022)
    python -m wildfire.spread.train --source ndws-numpy \\
        --data-dir wildfire_data/spread/ndws_npy \\
        --arch unet --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, IterableDataset, Dataset

from .config import SPREAD_CFG
from .data import build_dataset, build_split_datasets
from .losses import FireSpreadLoss
from .model import build_model


def _split_dataset(ds: Dataset, seed: int = 0) -> Tuple[Dataset, Dataset, Dataset]:
    if isinstance(ds, IterableDataset):
        # Iterable streams (TFRecord) can't be index-split; silently reusing
        # the training stream as val/test would report metrics computed on
        # training data. Refuse instead.
        raise SystemExit(
            "--source ndws-tfrecord streams an IterableDataset that cannot be "
            "split into train/val/test. Convert the shards to numpy first "
            "(python -m wildfire.spread.convert_ndws) and rerun with "
            "--source ndws-split, which uses the official NDWS split."
        )

    n = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(SPREAD_CFG.train_fraction * n)
    n_val   = int(SPREAD_CFG.val_fraction   * n)
    return (
        Subset(ds, idx[:n_train].tolist()),
        Subset(ds, idx[n_train:n_train + n_val].tolist()),
        Subset(ds, idx[n_train + n_val:].tolist()),
    )


@torch.no_grad()
def _evaluate(model, loader, device, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score

    model.eval()
    all_p, all_y = [], []
    tp = fp = fn = tn = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(torch.int8)

        ys = y.to(torch.int8)
        tp += int(((preds == 1) & (ys == 1)).sum().item())
        fp += int(((preds == 1) & (ys == 0)).sum().item())
        fn += int(((preds == 0) & (ys == 1)).sum().item())
        tn += int(((preds == 0) & (ys == 0)).sum().item())

        all_p.append(probs.flatten().cpu().numpy())
        all_y.append(ys.flatten().cpu().numpy())

    p = np.concatenate(all_p)
    y = np.concatenate(all_y)

    pos_rate = float(y.mean())
    auc = float(roc_auc_score(y, p)) if 0 < pos_rate < 1 else float("nan")
    ap  = float(average_precision_score(y, p)) if 0 < pos_rate < 1 else float("nan")

    iou      = tp / max(1, tp + fp + fn)
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "roc_auc": auc,
        "auc_pr": ap,                              # paper's headline metric for NDWS
        "iou_fire": iou,
        "precision_fire": precision,
        "recall_fire": recall,
        "f1_fire": f1,
        "fire_pixel_rate": pos_rate,
        "decision_threshold": threshold,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def train(args) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[spread] device: {device}")

    if args.source == "ndws-split":
        if args.data_dir is None:
            raise SystemExit("--source ndws-split requires --data-dir pointing at "
                             "the parent of train/val/test subdirs.")
        train_ds, val_ds, test_ds = build_split_datasets(args.data_dir)
    else:
        full = build_dataset(args.source, data_dir=args.data_dir,
                             n_samples=args.synthetic_size, seed=args.seed)
        train_ds, val_ds, test_ds = _split_dataset(full, seed=args.seed)
    print(f"[spread] dataset: {args.source}  "
          f"train={getattr(train_ds, '__len__', lambda: '?')() }  "
          f"val={getattr(val_ds, '__len__', lambda: '?')()}  "
          f"test={getattr(test_ds, '__len__', lambda: '?')()}")

    in_channels = len(SPREAD_CFG.feature_channels)
    train_loader = DataLoader(train_ds, batch_size=SPREAD_CFG.batch_size,
                              shuffle=not isinstance(train_ds, IterableDataset),
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=SPREAD_CFG.batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=SPREAD_CFG.batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=device.type == "cuda")

    model = build_model(args.arch, in_channels, SPREAD_CFG.lookback_days).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[spread] model: {args.arch}  params={n_params:,}")

    criterion = FireSpreadLoss(
        fire_pos_weight=SPREAD_CFG.fire_pos_weight,
        dice_weight=SPREAD_CFG.dice_loss_weight,
    )
    optim = torch.optim.AdamW(model.parameters(),
                              lr=SPREAD_CFG.learning_rate,
                              weight_decay=SPREAD_CFG.weight_decay)

    epochs = args.epochs or SPREAD_CFG.epochs
    best_val_ap = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running, count = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(loss.item()) * x.size(0)
            count   += x.size(0)
        train_loss = running / max(1, count)

        val_metrics = _evaluate(model, val_loader, device, SPREAD_CFG.decision_threshold)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"  epoch {epoch:02d}  loss={train_loss:.4f}  "
              f"val_auc={val_metrics['roc_auc']:.4f}  "
              f"val_ap={val_metrics['auc_pr']:.4f}  "
              f"val_iou={val_metrics['iou_fire']:.4f}")

        if val_metrics["auc_pr"] > best_val_ap:
            best_val_ap = val_metrics["auc_pr"]
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "in_channels": in_channels,
                "lookback_days": SPREAD_CFG.lookback_days,
                # Serving checks both: a synthetic-trained model is never run
                # on real imagery, and channels must match by name, not count.
                "feature_channels": list(SPREAD_CFG.feature_channels),
                "source": args.source,
            }, SPREAD_CFG.model_path())

    # Load best, evaluate on test
    ckpt = torch.load(SPREAD_CFG.model_path(), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = _evaluate(model, test_loader, device, SPREAD_CFG.decision_threshold)

    print("\n=== Test metrics (best-by-val-AP checkpoint) ===")
    for k, v in test_metrics.items():
        print(f"  {k:18s} {v}")

    meta = {
        "arch": args.arch,
        "source": args.source,
        "in_channels": in_channels,
        "lookback_days": SPREAD_CFG.lookback_days,
        "feature_channels": SPREAD_CFG.feature_channels,
        "epochs_run": epochs,
        "best_val_ap": best_val_ap,
        "test_metrics": test_metrics,
        "history": history,
    }
    SPREAD_CFG.meta_path().write_text(json.dumps(meta, indent=2))
    print(f"\n[spread] saved model -> {SPREAD_CFG.model_path()}")
    print(f"[spread] saved meta  -> {SPREAD_CFG.meta_path()}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train the wildfire spread forecaster.")
    parser.add_argument("--source", choices=["synthetic", "ndws-numpy", "ndws-tfrecord", "ndws-split"],
                        default="synthetic")
    parser.add_argument("--arch",   choices=["convlstm", "unet"], default="convlstm")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--synthetic-size", type=int, default=512,
                        help="Number of synthetic patches (only when --source synthetic).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device, e.g. 'cpu' or 'cuda'.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
