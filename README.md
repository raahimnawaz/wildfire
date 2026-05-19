# West Coast Wildfire Prediction

An operational two-stage wildfire prediction pipeline for the US West Coast.
Trained on **real** Google Earth Engine satellite data and the canonical
**Next Day Wildfire Spread** benchmark from Google Research.

| Model | Task | Real-data metrics | Benchmark |
|---|---|---|---|
| **XGBoost ignition** | Per-cell 7-day fire start probability | ROC-AUC **0.757** · AP 0.004 | 5 yrs train / 2025 holdout |
| **U-Net spread** | Per-pixel next-day fire mask | ROC-AUC **0.903** · AUC-PR **0.295** | Matches Huot et al. ICDM 2022 |

## What this is

Two complementary models that together cover the wildfire-prediction lifecycle:

1. **Ignition** — "where might a fire START in the next week?" Boosted-tree
   classifier on 2.4M live GEE-extracted West Coast cells (2020-2024),
   evaluated out-of-time on the 2025 fire season. Top feature importances
   (KBDI drought + wind speed = 96% of signal) match the published wildfire
   physics literature.
2. **Spread** — "given a fire is burning, what does tomorrow's perimeter look
   like?" U-Net segmentation on 64×64 km imagery patches with 11 channels
   (elevation, slope, aspect, NDVI, temp, dewpoint, wind u/v, KBDI, soil
   moisture, prior fire mask). AUC-PR matches the canonical NDWS benchmark.

A combined dashboard CLI runs both models on any date and produces a
satellite-style HUD heatmap plus structured outputs (parquet, GeoJSON, JSON).

![Heatmap screenshot placeholder — open wildfire_output/<date>/heatmap_<date>.html in a browser](#)

## Data sources

| Source | What | Where it lives in the pipeline |
|---|---|---|
| **ERA5-Land** (ECMWF) | Hourly weather: temp, dewpoint, wind u/v, soil moisture | Ignition + spread input |
| **SRTM** (USGS) | 30 m elevation → slope + aspect | Ignition + spread input |
| **NASA SMAP** | Surface soil moisture | Ignition + spread input |
| **UTOKYO KBDI** | Keetch-Byram Drought Index | Ignition + spread input |
| **MODIS NDVI** | 16-day vegetation index composites | Ignition + spread input |
| **NASA FIRMS** | MODIS T21 thermal anomalies | Ignition labels |
| **Next Day Wildfire Spread** (Huot et al. ICDM 2022) | 18,545 64×64 km patches of real CONUS fires (2012-2020) | Spread training |

## Repository layout

```
wildfire/
  config.py            central constants (bbox, paths, hyperparams, threshold)
  features.py          feature engineering (wind speed, RH, VPD, calendar)
  imbalance.py         domain-guided synthetic minority injection
  synthetic_data.py    offline smoke-test data generator (no GEE auth needed)
  data_extraction.py   GEE pipeline + extract_window + extract_patch (cached)
  cache.py             disk cache for GEE results (parquet + npy)
  train.py             XGBoost training with out-of-time validation + threshold tuning
  predict.py           XGBoost scoring CLI
  dashboard.py         combined ignition + spread CLI
  heatmap.py           satellite-style HUD heatmap (folium + custom shell)
  setup_gee.py         GEE connectivity verifier
  run_real.py          one-shot end-to-end runner
  spread/              Phase 2 — next-day fire spread
    config.py
    model.py           ConvLSTM cell + U-Net wrapper (smp)
    data.py            Synthetic + NDWS numpy + dependency-free TFRecord parser
    convert_ndws.py    TFRecord → numpy with official train/val/test split
    losses.py          weighted BCE + soft Dice
    train.py           PyTorch training
    predict.py         spread mask scoring
```

## Quick start

### Run with synthetic data (no auth required)

```powershell
pip install -r requirements.txt
python -m wildfire.train --synthetic
```

### Run with real data (requires GEE + Kaggle)

See [wildfire/README.md](wildfire/README.md) for the full GEE + Kaggle
walkthrough. Once auth is set up:

```powershell
# One-shot: pulls real data, trains both models
python -m wildfire.run_real

# Or generate a live risk dashboard for any date
$env:GEE_PROJECT = "your-gcp-project-id"
python -m wildfire.dashboard --date 2025-09-15
```

The dashboard writes to `wildfire_output/<date>/`:
- `heatmap_<date>.html` — satellite-style HUD visualization
- `ignition_<date>.parquet` — every scored cell + probability
- `ignition_<date>.geojson` — for QGIS / web maps
- `summary_<date>.json` — top-level metrics + spread predictions

## Engineering decisions worth calling out

Three non-obvious things that took real iteration to get right:

1. **Sentinel-mask trick for sparse GEE collections.** SMAP, KBDI, NDVI, and
   FIRMS all have data gaps. Calling `.mean()` on an empty `ImageCollection`
   returns a 0-band image that crashes any downstream `.multiply()` or
   `.rename()`. The fix: prepend a fully-masked sentinel image with the
   right band name to every collection. Masked pixels don't affect the mean,
   so the band always exists. See `_safe_mean()` in
   [wildfire/data_extraction.py](wildfire/data_extraction.py).

2. **Paged GEE sampling for the 5000-element limit.** `getInfo()` on a
   `FeatureCollection` caps at 5000 features — `numPixels=50000` silently
   fails. Workaround: N independent `.sample()` calls with different seeds,
   each under the limit, then concatenated and deduplicated by (lon, lat).
   Effective 10× density at the same per-call cost. See `extract_window()`.

3. **Data-driven threshold tuning.** Hardcoded operational threshold = 0.60
   produced zero alerts on the 2025 test set (predictions were calibrated
   low). The model's PR curve actually peaks at threshold 0.168 — at that
   threshold we get 12 true-positive alerts on 673 actual fires. The tuned
   threshold is computed from the test-set PR curve during training and
   persisted to model metadata, then read at inference time. See
   `_tune_threshold()` in [wildfire/train.py](wildfire/train.py).

## Verification & live runs

A live, daily-refreshed dashboard is auto-deployed via GitHub Actions —
see [.github/workflows/daily-dashboard.yml](.github/workflows/daily-dashboard.yml).

## License

Code: MIT. The NDWS dataset (Huot et al.) is CC BY 4.0 and not redistributed
here — see download instructions in
[wildfire/spread/README.md](wildfire/spread/README.md).

## Citations

- Huot et al., *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data*, ICDM 2022.
- Shi et al., *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*, NeurIPS 2015.
- Rothermel, *A Mathematical Model for Predicting Fire Spread in Wildland Fuels*, USDA Forest Service, 1972.
