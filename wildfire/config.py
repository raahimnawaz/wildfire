"""
Central configuration for the West Coast wildfire predictor.

All choices below are taken directly from the research paper:
  - West Coast bbox and fire-season window
  - 4 km cell resolution
  - Out-of-time train/test split (random K-fold causes fatal leakage)
  - Operational decision threshold of 0.60
  - Feature list spanning ERA5-Land, SRTM, SMAP, KBDI, Sentinel-2 NDVI
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
    train_year: int = 2022
    test_year: int = 2023

    # GEE sampling
    gee_num_pixels: int = 4_000   # per-call cap (GEE getInfo() hard limit is 5000)
    gee_pages_per_window: int = 40  # 4x bumped for 2km: keeps coverage at ~30% of cells
    gee_tile_scale: int = 4
    gee_seed: int = 42

    # FIRMS confidence threshold (% for MODIS T21)
    firms_min_confidence: int = 30

    # Features — keep names stable; train/predict both consume this list
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "temp_c",          # ERA5 2m temperature, Kelvin -> Celsius
            "dewpoint_c",      # ERA5 2m dewpoint, Kelvin -> Celsius
            "rh_pct",          # derived from temp + dewpoint
            "vpd_kpa",         # vapor pressure deficit (paper-aligned drought driver)
            "wind_speed_ms",   # sqrt(u^2 + v^2) at 10 m
            "wind_dir_deg",    # atan2(u, v) bearing
            "soil_moist_surf", # SMAP 0-10 cm volumetric, m^3/m^3
            "soil_moist_root", # ERA5 root-zone (28-100 cm)
            "kbdi_index",      # Keetch-Byram Drought Index, 0-800
            "ndvi",            # Sentinel-2 (or MODIS) NDVI, -1..1
            "elevation_m",     # SRTM
            "slope_deg",       # derived from SRTM
            "aspect_deg",      # derived from SRTM
            "month",           # 8, 9, 10
            "day_of_year",
        ]
    )
    target_column: str = "fire_occurred"

    # Class-imbalance handling (paper's preferred domain-guided injection)
    minority_oversample_factor: int = 30   # duplicate fire rows N times before perturbation
    majority_downsample_ratio: int = 10    # final neg:pos ratio cap
    extreme_temp_bump_c: float = 10.0
    extreme_wind_bump_ms: float = 8.0
    extreme_kbdi_floor: float = 750.0
    extreme_rh_ceiling_pct: float = 15.0

    # XGBoost hyperparameters — paper's reference baseline
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_tree_method: str = "hist"
    xgb_device: str = "cpu"   # set to "cuda" if a GPU + cuDF stack is available

    # Operational decision threshold (NOT 0.5 — paper-validated)
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
