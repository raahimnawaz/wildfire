"""
Central configuration for the West Coast wildfire predictor.

Key choices follow the project's research doc ("Comprehensive Analysis of
Wildfire Prediction"), adapted where real-data iteration disagreed with it:
  - West Coast bbox and Aug-Oct fire-season window
  - 2 km cell resolution (the doc's 4 km blurred out too many positive labels)
  - Out-of-time train/val/test split by year (random K-fold leaks temporally)
  - The operational decision threshold is tuned on the validation year during
    training and stored in model metadata; decision_threshold below is only a
    last-resort fallback for legacy metadata
  - Feature list spanning ERA5-Land, SRTM, SMAP L4, GRIDMET ERC, MODIS NDVI
  - The label is a FORECAST: a new FIRMS detection in the week after the
    feature window, in a cell with no detection in the two weeks before it
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "wildfire_data"
MODEL_DIR = PROJECT_ROOT / "wildfire_models"


@dataclass(frozen=True)
class WildfireConfig:
    # Geography — US West Coast (CA + OR + WA, includes coastal NV bleed)
    west_coast_bbox: tuple = (-124.73, 32.53, -114.13, 49.00)  # (W, S, E, N)
    crs: str = "EPSG:4326"
    cell_scale_m: int = 2000  # 4km blurred too much; 2km gives 4x more positive labels per fire

    # Temporal — fire season Aug 1 to Oct 31 (covers Santa Ana / Diablo wind events)
    fire_season_months: tuple = (8, 9, 10)
    # Months the trained model has actually seen (fire season + May-Jul
    # shoulder season). Dashboard warns when scoring outside this range.
    model_coverage_months: tuple = (5, 6, 7, 8, 9, 10)
    # Seasons available on disk / to extract. Real training passes explicit
    # parquets (train = all but last two, val = second-last, test = last);
    # train_year/test_year only drive the synthetic-mode split.
    data_years: tuple = (2020, 2021, 2022, 2023, 2024, 2025)
    train_year: int = 2022
    test_year: int = 2023

    # GEE sampling
    gee_num_pixels: int = 4_000   # per-call cap (GEE getInfo() hard limit is 5000)
    gee_pages_per_window: int = 40  # 4x bumped for 2km: keeps coverage at ~30% of cells
    gee_tile_scale: int = 4
    gee_seed: int = 42

    # Features — keep names stable; train/predict both consume this list
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "temp_c",          # ERA5 2m temperature, Kelvin -> Celsius
            "dewpoint_c",      # ERA5 2m dewpoint, Kelvin -> Celsius
            "rh_pct",          # derived from temp + dewpoint
            "vpd_kpa",         # vapor pressure deficit (paper-aligned drought driver)
            "wind_speed_ms",   # sqrt(u^2 + v^2) at 10 m
            "wind_dir_deg",    # atan2(u, v) bearing
            "soil_moist_surf", # SMAP L4 0-5 cm volumetric, m^3/m^3
            "soil_moist_root", # ERA5 root-zone (28-100 cm)
            "erc",             # GRIDMET Energy Release Component (NFDRS fuel dryness)
            "ndvi",            # MODIS MOD13Q1 16-day NDVI, -1..1
            "elevation_m",     # SRTM
            "slope_deg",       # derived from SRTM
            "aspect_deg",      # derived from SRTM
            "month",           # 8, 9, 10
            "day_of_year",
        ]
    )
    target_column: str = "fire_occurred"

    # Forecast label. fire_occurred = 1 when FIRMS detects fire in the
    # `label_horizon_days` AFTER the feature window, in a cell with no
    # detection in the `label_quiet_days` before the window ends. Cells that
    # were already burning are dropped from training: "will it ignite" has no
    # meaning for them.
    label_horizon_days: int = 7
    label_quiet_days: int = 14

    # A band whose source returned nothing for more than this share of a
    # window's cells is an outage, not data -- extraction raises instead of
    # handing the model a constant.
    max_fill_fraction: float = 0.2

    # Class-imbalance handling (research doc's domain-guided injection)
    minority_oversample_factor: int = 30   # duplicate fire rows N times before perturbation
    majority_downsample_ratio: int = 10    # final neg:pos ratio cap
    extreme_temp_bump_c: float = 10.0
    extreme_wind_bump_ms: float = 8.0
    # ERC has no fixed ceiling (it depends on fuel model and region), so the
    # "extreme" floor is the training set's own upper quantile.
    extreme_erc_quantile: float = 0.95
    extreme_rh_ceiling_pct: float = 15.0

    # XGBoost hyperparameters — research doc's reference baseline
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_tree_method: str = "hist"
    xgb_device: str = "cpu"   # set to "cuda" if a GPU + cuDF stack is available

    # Legacy fallback ONLY. The live operational threshold is tuned on the
    # validation year during training and read from feature_meta.json
    # (see predict.resolve_threshold). This constant applies just when metadata
    # predates threshold tuning.
    decision_threshold: float = 0.60

    # Persistence
    data_dir: Path = DATA_DIR
    model_dir: Path = MODEL_DIR
    model_filename: str = "west_coast_wildfire_model.json"
    feature_meta_filename: str = "feature_meta.json"

    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    def feature_meta_path(self) -> Path:
        return self.model_dir / self.feature_meta_filename


CFG = WildfireConfig()

# Make sure runtime directories exist (cheap; safe to import repeatedly)
CFG.data_dir.mkdir(parents=True, exist_ok=True)
CFG.model_dir.mkdir(parents=True, exist_ok=True)
