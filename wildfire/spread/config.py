"""Configuration for the ConvLSTM spread forecaster."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ..config import PROJECT_ROOT


SPREAD_DATA_DIR  = PROJECT_ROOT / "wildfire_data" / "spread"
SPREAD_MODEL_DIR = PROJECT_ROOT / "wildfire_models" / "spread"


@dataclass(frozen=True)
class SpreadConfig:
    # Patch geometry
    patch_size: int = 64               # 64x64 cells, matches Next Day Wildfire Spread benchmark
    cell_resolution_m: int = 1000      # NDWS uses 1 km cells
    lookback_days: int = 4             # ConvLSTM input timesteps

    # Channels — order is meaningful; train/predict must agree. Named for what
    # they hold in Next Day Wildfire Spread, the data the model trains on.
    # (Until schema v2 these were called temp_c/dewpoint_c/wind_u_ms/
    # wind_v_ms/kbdi_index/soil_moist while holding tmmx/tmmn/vs/vs*cos(th)/
    # PDSI/specific humidity, and the serving path filled them with what the
    # names said. The order is unchanged, so older checkpoints still load.)
    feature_channels: List[str] = field(default_factory=lambda: [
        "elevation",       # SRTM, m
        "slope_deg",       # zeros: NDWS carries no slope
        "aspect_deg",      # zeros: NDWS carries no aspect
        "NDVI",            # VIIRS VNP13A1, raw units (x1e4)
        "tmmx_c",          # GRIDMET daily max temperature, °C
        "tmmn_c",          # GRIDMET daily min temperature, °C
        "vs",              # GRIDMET wind speed, m/s
        "vs_cos_th",       # vs * cos(th), th = GRIDMET wind-from direction
        "pdsi",            # GRIDMET Palmer Drought Severity Index
        "sph",             # GRIDMET specific humidity, kg/kg
        "PrevFireMask",    # MOD14A1 fire mask on day t: -1 unknown, 0, 1 fire
    ])

    # Network shape
    hidden_channels:  tuple = (32, 64)
    kernel_size:      int   = 3
    use_batch_norm:   bool  = True

    # Training
    batch_size:    int   = 8
    epochs:        int   = 20
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-5
    fire_pos_weight: float = 5.0       # weighted BCE: fire pixels rarer
    dice_loss_weight: float = 0.5

    # Data split
    train_fraction: float = 0.8
    val_fraction:   float = 0.1
    # Remainder is test

    # Operational threshold for the per-pixel sigmoid
    decision_threshold: float = 0.5

    # Persistence
    data_dir:        Path = SPREAD_DATA_DIR
    model_dir:       Path = SPREAD_MODEL_DIR
    model_filename:  str  = "convlstm_spread.pt"
    meta_filename:   str  = "convlstm_meta.json"

    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    def meta_path(self) -> Path:
        return self.model_dir / self.meta_filename


SPREAD_CFG = SpreadConfig()
SPREAD_CFG.data_dir.mkdir(parents=True, exist_ok=True)
SPREAD_CFG.model_dir.mkdir(parents=True, exist_ok=True)
