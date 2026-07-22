# West Coast Wildfire Prediction — Engineering Writeup

How I built an operational two-stage wildfire prediction pipeline for the US
West Coast, what went wrong along the way, and how I fixed it. ~1,000 words.

---

## The problem

California is on fire more often, more intensely, and earlier in the season
than it was a decade ago. Operational tools used by the National Interagency
Fire Center and CAL FIRE are excellent at "is this region currently
dangerous?" — they fuse weather forecasts, fuel moisture sensors, and
historical fire return intervals into broad red-flag warnings. They are less
good at answering a more specific question: **for this particular 2 km cell
on this particular day, what is the probability of a fire starting in the
next week, and if one does, where will it be tomorrow?**

That question matters because the cost of being wrong is asymmetric: missed
ignitions become disasters, false alarms cost crew shifts. A useful model
has to be precise enough to triage where to pre-position resources without
crying wolf.

I built two complementary models, each on the timescale where it adds the
most value.

## The architecture — two timescales, two model families

**Ignition model** answers "WHERE could a fire start?" on a 7-day horizon.
This is a per-cell binary classification problem on a coarse spatial grid
where the inputs are structured tabular features (weather, drought,
vegetation, topography). XGBoost is the right tool here — it dominates
tabular benchmarks, handles non-linear feature interactions naturally, and
gives interpretable feature importances I can sanity-check against the
wildfire physics literature.

**Spread model** answers "GIVEN a fire is burning, where will it be
tomorrow?" This is a per-pixel image segmentation problem on a fine spatial
grid where the inputs are aligned multi-channel raster patches. U-Net is the
canonical architecture for this; the published Next Day Wildfire Spread
paper (Huot et al., ICDM 2022) uses essentially the same setup. I trained
my U-Net on the same dataset and confirmed it reproduces the paper's
~0.30 AUC-PR baseline.

Splitting the problem this way means each model gets data it's actually
suited for, and a downstream consumer can choose to run ignition only
(fast, daily triage) or chain ignition → spread (deeper analysis on
high-risk cells).

## Three engineering decisions worth calling out

The interesting parts of this project are not the model architectures —
those came straight out of papers. The interesting parts are the
production-data plumbing problems that anyone integrating Earth Engine
imagery would hit on day one and the fixes that took real iteration to land.

### 1. The sentinel-mask trick for sparse GEE collections

The first time I ran a real extraction, every other window crashed with
`Image.multiply: If one image has no bands, the other must also have no
bands. Got 0 and 1.` It turned out that SMAP, KBDI, MODIS NDVI, and FIRMS
all have data gaps — for any given 7-day window over the West Coast, at
least one of those collections is often empty. Calling `.mean()` on an
empty `ImageCollection` returns an image with **zero bands**, and the
downstream `.multiply(0.0001).rename("ndvi")` then crashes because you
can't rename bands on a 0-band image.

The naive fix is `ee.Algorithms.If(coll.size().gt(0), coll.mean(), 0)` —
but `ee.Algorithms.If` wraps the result in a `ComputedObject` that GEE
treats as fully masked, killing every downstream pixel. The fix that
actually works is to prepend a **fully-masked sentinel image** with the
right band name to every collection. Masked pixels don't contribute to
`.mean()` so the band always exists with the real data when present, and
falls back to the masked sentinel when not. See `_safe_mean()` in
[`wildfire/data_extraction.py`](wildfire/data_extraction.py).

### 2. Paged sampling for GEE's 5000-element `getInfo()` cap

Once the empty-collection bug was fixed, I bumped `numPixels=50000` to get
better coverage of the West Coast. Every window started returning empty
DataFrames again. After some staring at GEE error messages, I figured out
that `FeatureCollection.getInfo()` silently aborts at 5000 features on
free-tier accounts — `numPixels=50000` was hitting that ceiling and
returning nothing.

The workaround is **paged sampling**: instead of one giant `.sample()`,
do N independent `.sample()` calls with different seeds, each under the
cap, then concatenate and deduplicate by `(longitude, latitude)`. With
10 pages at 4,000 pixels each I get 32,000 unique cells per window (10×
the density at the same per-call cost) while staying well clear of the
limit. See `extract_window()` in the same file.

### 3. Data-driven threshold tuning from the PR curve

The model's first real-data evaluation reported `ROC-AUC 0.749` but zero
true positives in the confusion matrix. That seemed wrong until I realized
the hardcoded operational threshold of `0.60` was way above the model's
calibrated probability range — fire-positives are so rare (~0.1% of cells)
that almost no prediction crosses 0.60 even when the ranking is perfect.

The fix was to tune the operational threshold from the PR curve during
training and persist it to the model metadata. The first version of this
tuned on the test set — which quietly inflated the reported
precision/recall, since the threshold was chosen to look best on the same
rows it was then scored on. The current protocol holds out a dedicated
validation year: the threshold is tuned there, frozen, and the test year
sees it exactly once. I report three candidates — F1-optimal, Youden's J,
and "largest threshold meeting 10% precision" — all computed on
validation; the F1-optimal one ships in the model metadata and is what
`predict.py` and the dashboard apply. See `_tune_threshold()` in
[`wildfire/train.py`](wildfire/train.py).

## Honest results

| Model | Metric | Value | Baseline / paper |
|---|---|---|---|
| XGBoost ignition (2 km, May-Oct 2020-2026) | ROC-AUC (2025 out-of-time) | 0.785 | operational ceiling ~0.85 with 1 km cells |
| XGBoost ignition (2 km, May-Oct 2020-2026) | Average Precision | 0.0055 | base rate ~0.0011 → ~5× above chance |
| XGBoost ignition (2 km, May-Oct 2020-2026) | Recall @ val-tuned threshold 0.161 | 3.1% (precision 1.6%, 0.2% alert rate) | leak-free operating point |
| U-Net spread | ROC-AUC (NDWS test) | 0.903 | matches Huot et al. 2022 |
| U-Net spread | AUC-PR | 0.295 | matches Huot et al. 2022 (~0.30) |

The U-Net is at published-baseline quality. The XGBoost evaluates at
**0.785 ROC-AUC** under the leak-free protocol on May-October data: train
2020-2023 (24,518 real fire positives), threshold tuned on 2024
validation, 2025 test touched exactly once — then the production model
refits on everything including the 2026 season-to-date (~37K positives,
~24.5M cells). Honest notes: extending coverage from Aug-Oct to May-Oct
was driven by a real failure — the fire-season-only model had never seen
a July day, and when asked to score one it over-fired on 3.9% of cells;
the May-Oct model alerts on 0.2% at its tuned threshold and its AUC
*improved* to 0.785. AP reads lower than the season-only run simply
because the shoulder months halve the base rate. And the old "12.5%
precision at threshold 0.43" headline did NOT survive the leak fix — a
precision-targeted threshold chosen on validation does not transfer to
the test year, which is exactly the optimism the leaky protocol was
hiding. Going to 1 km would likely push AUC into the 0.82+ range and is
queued as future work — mainly limited by GEE quota, not algorithm.

## What would push this further

1. **1 km cell resolution** — the 4 km → 2 km move delivered the predicted
   ROC-AUC lift; 1 km is the next step, limited mainly by GEE sampling
   quota rather than algorithm.
2. **NDWI canopy moisture from Sentinel-2** — directly measures live fuel
   moisture, currently approximated by KBDI + soil moisture. 1-line GEE
   addition, expected +1-2 points.
3. **EMIT hyperspectral** — NASA's ISS-mounted hyperspectral imager covers
   the West Coast partially. Bands at 970, 1200, 1450, 1940 nm directly
   measure equivalent water thickness — the gold-standard fuel-moisture
   signal. Bigger lift if coverage is verified.
4. **Stratified sampling near MTBS fire perimeters** — biased toward the
   ignition-prone manifold, 10× more positives per GEE call.

## Repository

Full code, two-model pipeline, dashboard CLI, satellite-style HUD heatmap,
14-step commit history showing the engineering journey:
[github.com/raahimnawaz/wildfire](https://github.com/raahimnawaz/wildfire).
