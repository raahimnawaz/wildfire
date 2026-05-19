# Wildfire Spread Forecaster (Phase 2)

Per-pixel **next-day fire mask** predictor on 64×64 km patches, the second
modeling tier from the project's research doc.

| Component | Choice | Source |
|---|---|---|
| ConvLSTM | Custom PyTorch implementation of Shi et al. 2015 | [`model.py`](model.py) |
| U-Net | `segmentation_models.pytorch` (resnet18 encoder, no pretrain) | [smp](https://github.com/qubvel/segmentation_models.pytorch) |
| Loss | Weighted BCE + soft Dice | [`losses.py`](losses.py) |
| Benchmark dataset | Next Day Wildfire Spread (Huot et al., ICDM 2022) | [Kaggle](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) |
| Headline metric | AUC-PR (paper standard for fire-pixel imbalance) | sklearn |

## Quick smoke test (no data download)

```powershell
python -m wildfire.spread.train --source synthetic --arch convlstm --epochs 5
python -m wildfire.spread.train --source synthetic --arch unet     --epochs 5
```

You should see loss going down each epoch. The synthetic patches simulate a
fire front advancing under wind drift; U-Net typically hits ~0.7-0.8 ROC-AUC
in a few epochs, ConvLSTM more like 0.55-0.65 (it needs longer sequences to
shine).

## Real data: Next Day Wildfire Spread (NDWS)

The canonical wildfire-spread benchmark — ~70k 64×64 patches of real CONUS
fires (2012-2020), 12 channels, published with the ICDM 2022 paper. Two ways
to get it on disk:

### Option A — Kaggle numpy redistribution (recommended, easiest)

1. Make a Kaggle account + accept the dataset's terms.
2. Search Kaggle for "next-day-wildfire-spread" and download the version that
   ships pre-converted `.npy` files (look for `X.npy` + `y.npy`, or sharded
   `shard_*.npz`).
3. Place files under `wildfire_data/spread/ndws_npy/`:
   ```
   wildfire_data/spread/ndws_npy/
       X.npy          # shape (N, C, 64, 64)  or  (N, T, C, 64, 64)
       y.npy          # shape (N, 1, 64, 64)
   ```

Train:
```powershell
python -m wildfire.spread.train --source ndws-numpy `
    --data-dir wildfire_data/spread/ndws_npy `
    --arch unet --epochs 30
```

Expected on real NDWS data (paper baselines):
- AUC-PR around **0.30** for U-Net
- AUC-PR around **0.28** for ConvLSTM
- Random baseline AUC-PR equals fire-pixel rate (~0.012)

### Option B — Original Google TFRecord shards

If you grab the original [Google Research NDWS
TFRecords](https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread):

```powershell
python -m wildfire.spread.train --source ndws-tfrecord `
    --data-dir path/to/tfrecord/shards `
    --arch unet --epochs 30
```

Our loader includes a **dependency-free TFRecord parser** (no TensorFlow,
no `tfrecord` pip package — the latter has a known protobuf conflict).

## Scoring new patches

```powershell
python -m wildfire.spread.predict `
    --input  my_patches.npy `
    --output predictions.npz
```

The output `.npz` contains `fire_probability` (per-pixel float 0-1) and
`fire_alert` (binary mask at the configured threshold).

## Architecture choice

- **U-Net** — better starting point. Tested encoder zoo, well-understood,
  matches the original NDWS paper's setup. Use this if your input is one
  day of features (NDWS default).
- **ConvLSTM** — preferred by the project's research doc when you have a
  multi-day stack. Slower to train, more parameters per useful capacity, but
  captures fire-front drift across days.

You can swap encoders inside the U-Net (`UNetSpreadModel.__init__`) — the
smp library supports resnet18/34/50, efficientnet, mit_b0 (vision
transformer), and many more. For the wildfire setting, resnet34 with
`encoder_weights="imagenet"` is a strong default once you're past
plumbing.
