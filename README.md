# West Coast Wildfire Prediction

[![tests](https://github.com/raahimnawaz/wildfire/actions/workflows/tests.yml/badge.svg)](https://github.com/raahimnawaz/wildfire/actions/workflows/tests.yml)
[![daily-dashboard](https://github.com/raahimnawaz/wildfire/actions/workflows/daily-dashboard.yml/badge.svg)](https://github.com/raahimnawaz/wildfire/actions/workflows/daily-dashboard.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/%E2%96%B6_Open-Live_interactive_dashboard-00d4ff?style=for-the-badge&labelColor=050810)](https://raahimnawaz.github.io/wildfire/)

An operational two-stage wildfire prediction pipeline for the US West Coast.
Trained on **real** Google Earth Engine satellite data and the canonical
**Next Day Wildfire Spread** benchmark from Google Research.

> **▶ Click the screenshot below or the live-demo badge above to open the interactive dashboard** — pan, zoom, click cells for popups, click rows in the top-10 table to fly the map to that location.

| Model | Task | Real-data metrics | Benchmark |
|---|---|---|---|
| **XGBoost ignition** | Per-cell probability of a **new** fire starting in the next 7 days | **Pending retrain on schema v2** — see [the note below](#schema-v2-what-changed-and-why-the-old-numbers-were-withdrawn) | Train 2020-2023 · calibration + threshold on 2024 · tested once on 2025 · production refit on 2020-2026 |
| **U-Net spread** | Per-pixel next-day fire mask | ROC-AUC **0.903** · AUC-PR **0.295** on NDWS | Comparable to Huot et al. ICDM 2022 (different split — see note) |

### Schema v2: what changed, and why the old numbers were withdrawn

The previous ignition headline (ROC-AUC 0.785, "KBDI + wind dominate") came
from a model with three problems, now fixed in code. The model needs to be
re-extracted and retrained before it has a number again:

1. **It was a same-week detector, not a forecast.** Features and the FIRMS
   label came from the same 7-day window. The label now comes from the week
   *after* the features, counts only cells not already burning in the prior
   14 days, and already-burning cells are dropped from training.
2. **Its top feature was never measured.** The UTOKYO KBDI collection covers
   60°E–180°E only, so every real West Coast row held the fallback value 300.
   The only rows where KBDI varied were the synthetic augmented positives
   (KBDI raised to ≥750). That's why "KBDI" topped the importance chart: it
   was flagging which rows were synthetic. SMAP v007 also ended in June 2025.
   KBDI is replaced by GRIDMET ERC and SMAP by v008. Every gap-filled band now
   carries a `<band>_filled` flag: extraction raises when a source is mostly
   fill, and training refuses any effectively constant feature.
3. **Its probabilities weren't probabilities.** The rebalancing left raw
   scores 9–17× the observed fire rate. They are now Platt-calibrated on the
   validation year, and the calibration is reported on the untouched test
   year.

The spread model's serving inputs were also rebuilt: patches now come from
the sources NDWS was made from (GRIDMET, GRIDMET drought, VIIRS NDVI, MOD14A1)
and are centred on active fires. The U-Net numbers come from the one-shot
runner's pooled random split, not NDWS's official test split, so they are
comparable to the paper rather than a strict reproduction. The live demo and
screenshot still show the last schema-v1 snapshot until a new one is
generated.

## What this is

Two complementary models that together cover the wildfire-prediction lifecycle:

1. **Ignition** — "where might a fire START in the next week?" Boosted-tree
   classifier on GEE-extracted West Coast cells covering May-October of
   2020-2026 (fire season plus the shoulder season). Features describe one
   week; the label is a new FIRMS detection in the *following* week, in a
   cell with no detection in the 14 days before. Evaluation is fully
   out-of-time: train 2020-2023, calibration and alert threshold fit on
   held-out 2024, tested exactly once on 2025 — then the shipped production
   model is refit on everything, including the 2026 season-to-date. Outputs
   are calibrated probabilities.
2. **Spread** — "given a fire is burning, what does tomorrow's perimeter look
   like?" U-Net segmentation on 64×64 km patches in the Next Day Wildfire
   Spread schema (elevation, VIIRS NDVI, GRIDMET max/min temperature, wind
   speed and direction, PDSI, specific humidity, previous-day MODIS fire
   mask). At serving time, patches are pulled from the same sources and
   centred on active fires.

A combined dashboard CLI runs both models on any date and produces a
satellite-style HUD heatmap plus structured outputs (parquet, GeoJSON, JSON).

[![West Coast wildfire risk dashboard — satellite-style HUD heatmap](docs/screenshots/heatmap.png)](https://raahimnawaz.github.io/wildfire/)

## Data sources

| Source | What | Where it lives in the pipeline |
|---|---|---|
| **ERA5-Land** (ECMWF) | Hourly weather: temp, dewpoint, wind u/v, soil moisture | Ignition input |
| **SRTM** (USGS) | 30 m elevation → slope + aspect | Ignition + spread input |
| **NASA SMAP L4 v008** | Surface soil moisture | Ignition input |
| **GRIDMET** (U. Idaho) | Energy Release Component (ERC) | Ignition input |
| **GRIDMET** + **GRIDMET drought** | Daily max/min temp, wind, specific humidity; PDSI | Spread input (serving) |
| **MODIS NDVI** / **VIIRS NDVI** | 16-day vegetation index composites | Ignition / spread input |
| **NASA FIRMS** | MODIS T21 thermal anomalies | Ignition labels (next-week new detections), active fires |
| **MODIS MOD14A1** | Daily fire mask | Spread input (serving) |
| **Next Day Wildfire Spread** (Huot et al. ICDM 2022) | 18,545 64×64 km patches of real CONUS fires (2012-2020) | Spread training |

## Repository layout

```
wildfire/
  config.py            central constants (bbox, paths, label horizon, hyperparams)
  features.py          feature engineering (wind speed, RH, VPD, calendar)
  imbalance.py         domain-guided synthetic minority injection
  synthetic_data.py    offline smoke-test data generator (no GEE auth needed)
  data_extraction.py   GEE pipeline: forecast labels, fill flags, NDWS spread patches (cached)
  cache.py             disk cache for GEE results (parquet + npy), keyed by schema version
  calibration.py       Platt calibration + reliability report
  train.py             XGBoost training: out-of-time validation, calibration, threshold tuning
  predict.py           XGBoost scoring CLI
  dashboard.py         combined ignition + spread CLI
  service.py           interactive per-point risk API (FastAPI)
  heatmap.py           satellite-style HUD heatmap (folium + custom shell)
  setup_gee.py         GEE connectivity verifier
  run_real.py          one-shot end-to-end runner
  spread/              Phase 2 — next-day fire spread
    config.py
    model.py           ConvLSTM cell + U-Net wrapper (smp)
    ndws.py            the one NDWS feature → channel mapping
    data.py            Synthetic + NDWS numpy + dependency-free TFRecord parser
    convert_ndws.py    TFRecord → numpy with official train/val/test split
    losses.py          weighted BCE + soft Dice
    train.py           PyTorch training
    predict.py         spread mask scoring
```

## Quick start

### Run with synthetic data (no auth required)

```powershell
pip install -r wildfire/requirements.txt
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

### Serve interactive per-point risk

The dashboard answers "where is the risk today?" for the whole region, once a
day. To ask "what about *here*?" for an arbitrary point and date:

```powershell
$env:GEE_PROJECT = "your-gcp-project-id"
python -m wildfire.service --port 8000
```

```
POST /predict   {"lat": 39.8, "lon": -121.6, "target_date": "2025-09-15"}
GET  /model     metrics, threshold and feature importances, from the artifact
GET  /health
```

It scores a 32x32 grid of 2 km cells -- the cell size the model was trained on,
because a 200 m soil-moisture cell is not a 2 km one -- through the same
`_stack_for_window` extraction and `score_dataframe` scoring the dashboard uses,
so there is no second copy of the feature pipeline to drift out of step.

Three things it deliberately will not do, because an earlier prototype did all
three:

- **It returns the model's probabilities, never rescaled per request.** Min-max
  normalising each response makes a uniformly safe patch render identically to a
  dangerous one, and pins any "max risk" readout at exactly 100%. The
  probabilities are Platt-calibrated on the validation year, and
  `probability_calibrated` in every response says so (it is false only for an
  artifact that predates calibration). Ocean, grid padding and cells with a
  missing source come back as `null` rather than being scored.
- **Every quality number is read from `wildfire_models/feature_meta.json`** --
  test ROC-AUC, average precision, the validation-tuned threshold and its
  provenance. There is no metric literal in the module, and a metric the
  artifact does not carry is omitted rather than defaulted to something
  plausible.
- **No data is a 503, not a fallback.** If ERA5-Land latency leaves every window
  in the preceding 8 days empty, or the trained artifact is missing, it says so
  and names the fix. It never substitutes synthetic features into a risk map.

There is no uncertainty or saliency field. A boosted tree has no dropout to
sample, so an uncertainty band here would have been invented; the artifact's
real gain-based feature importance is served instead. An artifact trained on
a different feature set (for example a schema-v1 model with `kbdi_index`) is a
503 telling you to retrain, never scored.

## Engineering decisions worth calling out

Three non-obvious things that took real iteration to get right:

1. **Sentinel-mask trick for sparse GEE collections.** SMAP, KBDI, NDVI, and
   FIRMS all have data gaps. Calling `.mean()` on an empty `ImageCollection`
   returns a 0-band image that crashes any downstream `.multiply()` or
   `.rename()`. The fix: prepend a fully-masked sentinel image with the
   right band name to every collection. Masked pixels don't affect the mean,
   so the band always exists. See `_safe_mean()` in
   [wildfire/data_extraction.py](wildfire/data_extraction.py). The trick has
   a cost: it hid two dead sources for a long time (a KBDI product with no
   West Coast pixels, and SMAP v007 after it ended). The fill value is now
   paired with a `<band>_filled` flag, and a band that is mostly fill raises
   `FeatureOutage` instead of reaching the model.

2. **Paged GEE sampling for the 5000-element limit.** `getInfo()` on a
   `FeatureCollection` caps at 5000 features — `numPixels=50000` silently
   fails. Workaround: N independent `.sample()` calls with different seeds,
   each under the limit, then concatenated and deduplicated by (lon, lat).
   Effective 10× density at the same per-call cost. See `extract_window()`.

3. **Leak-free calibration and threshold tuning.** The hardcoded
   operational threshold = 0.60 produced zero alerts on real data — the fire
   base rate is ~0.15%. Both the Platt calibration and the threshold (from
   the PR curve) are now fit on a held-out VALIDATION year (2024), frozen
   into model metadata, and the test year (2025) sees them exactly once. (An earlier version tuned on the test set
   itself, which inflated the reported precision — fixed, and worth
   admitting.) `predict.py` and the dashboard read the tuned value from
   metadata at inference time. See `_tune_threshold()` in
   [wildfire/train.py](wildfire/train.py).

## Testing

```powershell
pip install -r wildfire/requirements.txt
pytest tests/ -q
```

73 tests covering cache hit/miss and schema keying, feature math (wind
speed, RH, VPD, dewpoint inversion, calendar), imbalance handling (including
thermodynamic consistency of augmented rows), forecast-label windows, the
fill-flag outage guard and dead-feature check, calibration (fit, monotonicity,
out-of-sample reliability), out-of-time splitting, threshold tuning, the
NDWS channel mapping, and the risk service end to end against a model that
`wildfire.train` trains on synthetic data into a temp dir. All run CPU-only
with no external services (no GEE, no GPU), and none touch the committed
model artifact.

## Verification & live runs

The newest real-data dashboard snapshot is auto-published to GitHub Pages
by GitHub Actions nightly. Without credentials, CI redeploys the newest
committed snapshot (never synthetic stand-in data); adding a `GEE_SA_JSON`
service-account secret flips the same workflow into live mode — it
generates today's dashboard from a fresh GEE pull in-runner each night,
with the committed snapshot as automatic fallback. See
[.github/workflows/daily-dashboard.yml](.github/workflows/daily-dashboard.yml).
A long-form engineering writeup is in [WRITEUP.md](WRITEUP.md).

## License

Code: MIT. The NDWS dataset (Huot et al.) is CC BY 4.0 and not redistributed
here — see download instructions in
[wildfire/spread/README.md](wildfire/spread/README.md).

## Citations

- Huot et al., *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data*, ICDM 2022.
- Shi et al., *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*, NeurIPS 2015.
- Rothermel, *A Mathematical Model for Predicting Fire Spread in Wildland Fuels*, USDA Forest Service, 1972.
