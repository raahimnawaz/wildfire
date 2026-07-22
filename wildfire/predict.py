"""
Score new observations with a trained West Coast wildfire model.

Two ways to call it:
  1. Pass a CSV/parquet of feature rows (with raw GEE column names OR the
     final feature set) and get back the same file with `fire_probability`
     and `fire_alert` columns appended.
  2. Programmatic: from wildfire.predict import score_dataframe; score_dataframe(df).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG
from .features import derive_all


def _load_model():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise SystemExit("xgboost is not installed.")

    if not CFG.model_path().exists():
        raise SystemExit(
            f"No trained model at {CFG.model_path()}.\n"
            "Train one first:  python -m wildfire.train --synthetic"
        )
    model = XGBClassifier()
    model.load_model(str(CFG.model_path()))

    meta = {}
    if CFG.feature_meta_path().exists():
        meta = json.loads(CFG.feature_meta_path().read_text())
    return model, meta


def resolve_threshold(meta: dict) -> float:
    """
    Operational threshold precedence: validation-tuned value in current
    metadata -> legacy metadata locations -> CFG fallback. Explicit None
    checks so a legitimate 0.0 threshold would not fall through.
    """
    for candidate in (
        meta.get("operational_threshold"),
        (meta.get("metrics") or {}).get("operational_threshold"),
        meta.get("decision_threshold"),
    ):
        if candidate is not None:
            return float(candidate)
    return float(CFG.decision_threshold)


def score_dataframe(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Append fire_probability and fire_alert columns. Non-destructive."""
    model, meta = _load_model()
    if threshold is None:
        # Operational threshold tuned on the validation year during training.
        threshold = resolve_threshold(meta)

    # If the dataframe already has the engineered features we still want to be
    # idempotent — derive_all is safe to call.
    derived = derive_all(df.copy())
    X = derived[CFG.feature_columns].astype("float32").values
    proba = model.predict_proba(X)[:, 1]

    # Return the DERIVED dataframe (raw cols + engineered cols) with
    # predictions appended. Downstream consumers (heatmap, dashboard) read
    # temp_c, wind_speed_ms, etc. — those only exist after derive_all.
    out = derived.copy()
    out["fire_probability"] = proba
    out["fire_alert"]       = (proba >= threshold).astype(np.int8)
    return out


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_any(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Score wildfire ignition probability.")
    parser.add_argument("--input",  type=Path, required=True,
                        help="CSV or parquet of feature rows.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the scored output.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the decision threshold (default: model metadata).")
    parser.add_argument("--top", type=int, default=10,
                        help="Print the top-N highest-risk rows after scoring.")
    args = parser.parse_args()

    df  = _read_any(args.input)
    out = score_dataframe(df, threshold=args.threshold)
    _write_any(out, args.output)

    n_alert = int(out["fire_alert"].sum())
    print(f"\nScored {len(out):,} rows -> {args.output}")
    print(f"  fire_alert positives : {n_alert:,}")
    if args.top > 0:
        cols = [c for c in ("longitude", "latitude", "date",
                            "temp_c", "wind_speed_ms", "kbdi_index",
                            "fire_probability") if c in out.columns]
        top = out.sort_values("fire_probability", ascending=False).head(args.top)
        print(f"\nTop {args.top} highest-risk rows:")
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
