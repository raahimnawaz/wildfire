# West Coast Wildfire Ignition Predictor

A working ML pipeline that predicts daily wildfire ignition probability for
each ~4 km cell on the US West Coast. Built directly from the recommendations
in *Comprehensive Analysis of Wildfire Prediction* (the project's research doc).

## What this is

| Choice | Reason (per paper) |
|---|---|
| **XGBoost binary classifier** | Paper's reference architecture for tabular ignition data. SOTA on this problem class. SHAP-interpretable. |
| **4 km cell, daily resolution** | Matches paper's reference scale; tractable over the whole West Coast. |
| **ERA5-Land + SRTM + SMAP L4 + GRIDMET ERC + NDVI + FIRMS** | Paper's recommended stack, with ERC standing in for KBDI (the only KBDI in Earth Engine covers 60°E–180°E, not the West Coast). |
| **Domain-guided minority injection** | Paper rejects naive SMOTE. We duplicate fire rows then bump them toward extreme regimes (`+10°C`, `+8 m/s wind`, ERC ≥ the training set's 95th percentile, `RH<=15%`). |
| **Out-of-time train/test split** | Paper flags random K-fold as fatal leakage from spatial+temporal autocorrelation. |
| **Forecast label** | New FIRMS detection in the 7 days *after* the feature window, in a cell not already burning. |
| **Calibrated probabilities, validation-tuned threshold** | Platt scaling and the alert threshold are both fit on the validation year; the paper's fixed 0.60 produced zero alerts at a ~0.1% base rate. |

The "zenith" architectures from the paper (ConvLSTM for spread, MA-NET
attention for burned-area regression, PhysFire-WM hybrid PINN) are intentionally
**not** in this v1 — they need imagery tensors and orders of magnitude more
compute. Architecture stubs are documented at the bottom of this README.

## Project layout

```
wildfire/
  config.py            central constants (bbox, dates, hyperparams, thresholds)
  features.py          feature engineering (wind speed, RH, VPD, calendar)
  imbalance.py         domain-guided synthetic minority injection
  synthetic_data.py    offline regime-conditional dataset (no GEE required)
  data_extraction.py   Google Earth Engine pipeline (real data)
  train.py             out-of-time training + evaluation
  predict.py           CLI for scoring new observations
  requirements.txt
```

## Live risk dashboard (the combined endpoint)

If both the ignition model AND the spread model are trained, you can produce
a one-shot wildfire risk report for any date:

```powershell
$env:GEE_PROJECT = "your-project-id"
python -m wildfire.dashboard --date 2025-09-15
```

Outputs in `wildfire_output/2025-09-15/`:
- `dashboard_2025-09-15.html` — single-page report, opens in any browser
- `ignition_2025-09-15.parquet` — every scored cell + ignition probability
- `ignition_2025-09-15.geojson` — for QGIS / web maps
- `summary_2025-09-15.json` — top-level metrics + per-patch spread predictions

The dashboard:
1. Pulls a fresh GEE snapshot of West Coast features for a 7-day window
2. Scores every ~4 km cell with the XGBoost ignition model
3. Runs the U-Net spread model on up to 50 active fires, from NDWS-source patches
4. Writes the four output files above

Skip the spread step with `--ignition-only` (much faster, ignition-only HTML).

## Quick start (no GEE auth needed)

```bash
pip install -r wildfire/requirements.txt

# 1. Train on synthetic regime-conditional data
python -m wildfire.train --synthetic

# 2. Score the synthetic dataset back to see predictions
python -c "from wildfire.synthetic_data import make_synthetic_dataset; \
           make_synthetic_dataset(n_total=2000).to_parquet('demo.parquet')"
python -m wildfire.predict --input demo.parquet --output demo_scored.parquet
```

You'll get an `XGBoost` model + metadata in `wildfire_models/` and a parquet
with `fire_probability` and `fire_alert` columns.

## Real data (Google Earth Engine) — full walkthrough

GEE is free for non-commercial use, but the first-time setup has a few moving
parts. Do them in order:

### 1. Create a Google Cloud project (one-time, ~3 min)

1. Go to <https://console.cloud.google.com/projectcreate>
2. Project name: anything, e.g. `wildfire-predictor`
3. Note the **Project ID** that Google generates (usually
   `wildfire-predictor-123456`). You'll need it as `GEE_PROJECT`.
4. Visit <https://signup.earthengine.google.com/> and register that project
   for Earth Engine access (the form takes a minute; approval is usually
   automatic for non-commercial use).

### 2. Install + authenticate (one-time)

```powershell
pip install earthengine-api
earthengine authenticate          # opens browser, sign in with the same Google account
```

### 3. Set the project id for this shell session

```powershell
# Windows PowerShell (this terminal only)
$env:GEE_PROJECT = "wildfire-predictor-123456"

# bash
export GEE_PROJECT=wildfire-predictor-123456
```

### 4. Verify connectivity (10 sec)

```powershell
python -m wildfire.setup_gee
```

You should see "OK — got N sampled pixels" and a sample row of ERA5 +
FIRMS values. If anything fails, the script tells you exactly what's wrong.

### 5. Pull real fire seasons (~10 min each)

Each 7-day feature window is labelled by the FIRMS detections in the week
after it, so a window is only extracted once that week has been observed.
Parquets from before schema v2 can't be reused: they carry a same-week label
and a KBDI column that was a constant fallback. `derive_all` refuses them with
a message saying so.

```powershell
# One command per season; the full protocol uses 2020-2025
python -m wildfire.data_extraction --year 2024 --out wildfire_data/raw_2024_v2.parquet
python -m wildfire.data_extraction --year 2025 --out wildfire_data/raw_2025_v2.parquet
```

### 6. Retrain on real data, out-of-time

Train on the older years, tune the alert threshold on the second-newest,
test once on the newest, then refit the production model on everything:

```powershell
# Full protocol (recommended — every season contributes to the shipped model)
python -m wildfire.train `
    --train wildfire_data/raw_2020_v2.parquet wildfire_data/raw_2021_v2.parquet wildfire_data/raw_2022_v2.parquet wildfire_data/raw_2023_v2.parquet `
    --val   wildfire_data/raw_2024_v2.parquet `
    --test  wildfire_data/raw_2025_v2.parquet `
    --refit-all

# Smallest setup — two years; validation is split off the last ~20% of
# training dates automatically when --val is omitted
python -m wildfire.train --train wildfire_data/raw_2022_v2.parquet `
                         --test  wildfire_data/raw_2023_v2.parquet
```

Honest expected metrics on real GEE data (depend on cell density + years available):
- ROC-AUC 0.65-0.80 with paged sampling at 2 km cells
- AUC-PR 0.01-0.10 (fire base rate is ~0.1%, so AP is the honest number)
- Feature importances are partly shaped by the domain-guided augmentation,
  which pushes synthetic positives toward hot, windy, high-ERC regimes. Compare
  against a run with `minority_oversample_factor=0` before reading physics
  into them. (Schema v1's "KBDI dominates" was the augmentation alone: KBDI
  only ever varied on synthetic rows.)
- Test-year calibration (mean predicted vs observed rate, ECE, reliability
  bins) is in `feature_meta.json` under `metrics.test.calibration`

Expect ~10 minutes per year on the default sampling density (50k samples per
window). Bump `gee_num_pixels` in `config.py` for denser sampling at the cost
of GEE quota.

## Outputs

- `wildfire_models/west_coast_wildfire_model.json` — trained XGBoost
- `wildfire_models/feature_meta.json` — feature list + metrics + importances
- `wildfire_data/raw_<year>.parquet` — extracted GEE features per season
- `<your_input>_scored.parquet` — input rows + `fire_probability`, `fire_alert`

## Metrics reported

- ROC-AUC (primary)
- Average Precision (because the test set is imbalanced)
- Confusion matrix and per-class precision/recall/F1 at the operational threshold
- Top-10 feature importances

## Phase 2 — spread forecaster (BUILT, see [spread/](spread/))

The next-day fire mask predictor is now implemented in [wildfire/spread/](spread/):

- **ConvLSTM** (Shi et al. 2015 generalized to fire spread) — recommended by
  the research doc when you have multi-day input.
- **U-Net** via `segmentation_models.pytorch` — matches the architecture of
  the canonical Next Day Wildfire Spread paper (Huot et al., ICDM 2022).
- Real-data loader for the NDWS benchmark (~70k real CONUS fire patches,
  2012-2020), with both numpy and original-TFRecord paths.

```powershell
# Smoke test (no download needed)
python -m wildfire.spread.train --source synthetic --arch unet --epochs 5

# Real data
python -m wildfire.spread.train --source ndws-numpy `
    --data-dir wildfire_data/spread/ndws_npy --arch unet --epochs 30
```

See [spread/README.md](spread/README.md) for the full guide.

## Phase 3 (not implemented)

1. **MA-NET burned-area regressor** — Vision Transformer with multi-level
   fusion + position attention, trained on Sentinel-2 patches around historical
   fire perimeters.
2. **PINN hybrid** — augment the standard CE loss with PDE-violation penalties
   (mass conservation, Rothermel surface-spread residuals). The paper's
   PhysFire-WM is a diffusion transformer with an embedded PDE simulator.

## Important caveats

- **Out-of-time validation only.** Do not switch to K-fold "for stability".
- **Apply minority injection to the training set only.** Never the test set.
- **Threshold tuning matters.** 0.60 is the paper-validated default; tune it on
  your operational cost curve (false alarms vs. missed ignitions).
- **The synthetic dataset is for plumbing, not for science.** Numbers from
  `--synthetic` runs are inflated because the regimes are linearly separable.
