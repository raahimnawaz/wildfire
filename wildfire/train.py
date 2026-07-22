"""
Train the West Coast wildfire ignition predictor.

Pipeline:
  1. Load train / validation / test parquets (or synthesize them).
  2. Out-of-time splits — train years < validation year < test year. If no
     validation file is given, the last ~20% of training DATES are held out
     (still out-of-time, never a random row split).
  3. Domain-guided minority injection on the training set ONLY.
  4. Fit XGBoost binary classifier with research-doc baseline hyperparameters.
  5. Tune the operational threshold on the VALIDATION set (never the test
     set), then evaluate once on the untouched test set at that frozen
     threshold.
  6. Optionally (--refit-all) refit on train+val+test for the production
     model — standard practice once evaluation is locked in: the reported
     metrics come from the held-out protocol, the shipped model uses every
     season available.
  7. Persist model JSON + feature metadata for the predict.py CLI.

Usage:
    # Run end-to-end on synthetic data (no GEE required):
    python -m wildfire.train --synthetic

    # Full real-data protocol: train 2020-2023, tune threshold on 2024,
    # report on 2025, then refit the production model on all six seasons:
    python -m wildfire.train \\
        --train wildfire_data/raw_2020.parquet wildfire_data/raw_2021.parquet \\
                wildfire_data/raw_2022.parquet wildfire_data/raw_2023.parquet \\
        --val   wildfire_data/raw_2024.parquet \\
        --test  wildfire_data/raw_2025.parquet \\
        --refit-all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CFG
from .features import derive_all
from .imbalance import rebalance_training_set, scale_pos_weight


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features, then slim to just what training needs (features +
    target + date) with float32 downcasting — keeps six seasons (~11.5M rows)
    comfortably inside memory."""
    df = derive_all(df)
    keep = [c for c in CFG.feature_columns + [CFG.target_column] if c in df.columns]
    out = df[keep].copy()
    if "date" in df.columns:
        # Parquets written by different pipeline eras store date at different
        # datetime64 resolutions (ns vs us); normalize so concat works.
        out["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    for c in CFG.feature_columns:
        if c in out.columns and out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    return out


def _years_of(df: pd.DataFrame) -> List[int]:
    if "date" not in df.columns or df.empty:
        return []
    return sorted(pd.to_datetime(df["date"]).dt.year.unique().tolist())


def _temporal_val_split(df: pd.DataFrame, frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out the LAST `frac` of distinct dates as validation. Out-of-time
    within the training window, so threshold tuning never sees test data and
    never peeks forward. Falls back to a seeded random split only if the
    date-based one can't produce both classes on both sides.
    """
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        order = np.sort(dates.unique())
        if len(order) >= 2:
            idx = max(1, min(len(order) - 1, int(round(len(order) * (1.0 - frac)))))
            cutoff = order[idx]
            train, val = df[dates < cutoff], df[dates >= cutoff]
            if (not train.empty and not val.empty
                    and train[CFG.target_column].nunique() == 2
                    and val[CFG.target_column].nunique() == 2):
                return train.copy(), val.copy()
    rng = np.random.default_rng(CFG.gee_seed)
    mask = rng.random(len(df)) < frac
    return df[~mask].copy(), df[mask].copy()


def _load_concat(paths: List[Path], label: str) -> pd.DataFrame:
    dfs = []
    for p in paths:
        print(f"[train] loading {p} -> {label}")
        dfs.append(_prep(pd.read_parquet(p)))
    out = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    if len(paths) > 1:
        print(f"[train] concatenated {len(paths)} {label} parquets: {len(out):,} rows total")
    return out


def _load_or_synthesize(synthetic: bool,
                        train_paths: Optional[List[Path]],
                        val_paths: Optional[List[Path]],
                        test_paths: Optional[List[Path]]
                        ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    if synthetic:
        from .synthetic_data import make_synthetic_dataset
        print("[train] generating synthetic dataset (no GEE needed) ...")
        df = make_synthetic_dataset(n_total=200_000)
        years = pd.to_datetime(df["date"]).dt.year
        train_df = _prep(df[years == CFG.train_year].copy())
        test_df  = _prep(df[years == CFG.test_year].copy())
        return train_df, None, test_df

    if not train_paths or not test_paths:
        raise SystemExit("Must pass --train and --test parquet paths (or --synthetic).")
    train_df = _load_concat(train_paths, "training set")
    val_df = _load_concat(val_paths, "validation set (threshold tuning)") if val_paths else None
    test_df = _load_concat(test_paths, "test set")
    return train_df, val_df, test_df


def _tune_threshold(y_true, proba, strategy: str = "f1") -> dict:
    """
    Find the operational threshold from the PR curve on the VALIDATION set.

    Strategies:
      * "f1": maximize fire-class F1 — balanced precision/recall.
      * "youden": maximize Youden's J on the ROC curve (TPR - FPR).
      * "precision@0.10": find the largest threshold that still hits
        >=10% precision (false-alarm-bounded operation).

    Returns a dict with the tuned threshold + diagnostics for every strategy.
    """
    from sklearn.metrics import precision_recall_curve, roc_curve

    precisions, recalls, pr_thr = precision_recall_curve(y_true, proba)
    # precision_recall_curve drops the last point so it matches thresholds
    f1 = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    if len(f1) == 0:
        return {"f1": {"threshold": 0.5}, "youden": {"threshold": 0.5}, "precision@0.10": {"threshold": 0.5}}

    best_f1_idx = int(np.nanargmax(f1))
    f1_choice = {
        "threshold": float(pr_thr[best_f1_idx]),
        "f1":        float(f1[best_f1_idx]),
        "precision": float(precisions[best_f1_idx]),
        "recall":    float(recalls[best_f1_idx]),
    }

    fpr, tpr, roc_thr = roc_curve(y_true, proba)
    youden = tpr - fpr
    best_y_idx = int(np.argmax(youden))
    youden_choice = {
        "threshold": float(roc_thr[best_y_idx]),
        "tpr":       float(tpr[best_y_idx]),
        "fpr":       float(fpr[best_y_idx]),
    }

    # Precision-floor: largest threshold that still gets >= 10% precision
    target_p = 0.10
    mask = precisions[:-1] >= target_p
    if mask.any():
        # pick the highest-threshold point that meets the floor (maximizes precision)
        chosen = int(np.argmax(pr_thr[mask]))
        candidates = np.where(mask)[0]
        idx = candidates[chosen]
        prec_choice = {
            "threshold": float(pr_thr[idx]),
            "precision": float(precisions[idx]),
            "recall":    float(recalls[idx]),
        }
    else:
        prec_choice = {"threshold": float(pr_thr[-1]) if len(pr_thr) else 0.5,
                       "precision": 0.0, "recall": 0.0,
                       "note": "no threshold reaches target precision 0.10"}

    out = {"f1": f1_choice, "youden": youden_choice, "precision@0.10": prec_choice}
    chosen = out.get(strategy, out["f1"])
    out["chosen_strategy"] = strategy
    out["chosen_threshold"] = chosen["threshold"]
    return out


def _metrics_at(y_true, proba, threshold: float, label: str) -> dict:
    """Threshold-free metrics + classification report at a FIXED threshold."""
    from sklearn.metrics import (
        roc_auc_score,
        classification_report,
        confusion_matrix,
        average_precision_score,
    )

    preds = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    ap  = average_precision_score(y_true, proba)
    cm  = confusion_matrix(y_true, preds).tolist()
    report = classification_report(
        y_true, preds, target_names=["no_fire", "fire"], output_dict=True, zero_division=0
    )

    print(f"\n=== {label} ===")
    print(f"  rows                : {len(y_true):,}")
    print(f"  positive rate       : {y_true.mean():.5f}")
    print(f"  ROC-AUC             : {auc:.4f}")
    print(f"  Average Precision   : {ap:.4f}  (positive base rate {y_true.mean():.4f})")
    print(f"  threshold applied   : {threshold:.4f}")
    print(f"  confusion matrix    : {cm}")
    print(classification_report(y_true, preds, target_names=["no_fire", "fire"], zero_division=0))

    return {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "classification_report": report,
        "n_rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def _fit_xgb(X, y, spw: float, eval_set=None):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators     = CFG.xgb_n_estimators,
        learning_rate    = CFG.xgb_learning_rate,
        max_depth        = CFG.xgb_max_depth,
        subsample        = CFG.xgb_subsample,
        colsample_bytree = CFG.xgb_colsample_bytree,
        tree_method      = CFG.xgb_tree_method,
        device           = CFG.xgb_device,
        scale_pos_weight = spw,
        eval_metric      = "auc",
        objective        = "binary:logistic",
        random_state     = CFG.gee_seed,
        n_jobs           = -1,
    )
    model.fit(X, y, eval_set=eval_set, verbose=False)
    return model


def _xy(df: pd.DataFrame):
    X = df[CFG.feature_columns].astype("float32").values
    y = df[CFG.target_column].astype("int8").values
    return X, y


def train(synthetic: bool = False,
          train_paths: Optional[List[Path]] = None,
          val_paths:  Optional[List[Path]] = None,
          test_paths: Optional[List[Path]] = None,
          refit_all:  bool = False,
          refit_extra_paths: Optional[List[Path]] = None) -> dict:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        raise SystemExit(
            "xgboost is not installed. Install with:\n"
            "    pip install -r wildfire/requirements.txt"
        )

    train_df, val_df, test_df = _load_or_synthesize(synthetic, train_paths, val_paths, test_paths)

    if val_df is None:
        train_df, val_df = _temporal_val_split(train_df)
        print(f"[train] no --val given: held out last-by-date "
              f"{len(val_df):,}/{len(val_df) + len(train_df):,} training rows as validation")

    train_years, val_years, test_years = _years_of(train_df), _years_of(val_df), _years_of(test_df)
    print(f"[train] train years {train_years}: {len(train_df):,} rows "
          f"(positives: {(train_df[CFG.target_column] == 1).sum():,})")
    print(f"[train] val   years {val_years}: {len(val_df):,} rows "
          f"(positives: {(val_df[CFG.target_column] == 1).sum():,})")
    print(f"[train] test  years {test_years}: {len(test_df):,} rows "
          f"(positives: {(test_df[CFG.target_column] == 1).sum():,})")

    rebalanced = rebalance_training_set(train_df)
    spw = scale_pos_weight(rebalanced)
    print(f"[train] rebalanced rows: {len(rebalanced):,}  "
          f"(positives: {(rebalanced[CFG.target_column] == 1).sum():,})")
    print(f"[train] scale_pos_weight: {spw:.3f}")

    X_train, y_train = _xy(rebalanced)
    X_val,   y_val   = _xy(val_df)
    X_test,  y_test  = _xy(test_df)

    print("[train] fitting XGBoost (validation set monitors AUC) ...")
    model = _fit_xgb(X_train, y_train, spw, eval_set=[(X_val, y_val)])

    # Threshold tuning happens on the VALIDATION set only; the test set sees
    # the frozen threshold exactly once, so reported precision/recall are
    # honest out-of-time numbers.
    proba_val = model.predict_proba(X_val)[:, 1]
    tuning = _tune_threshold(y_val, proba_val, strategy="f1")
    operational_threshold = float(tuning["chosen_threshold"])
    print(f"\n[train] operational threshold tuned on validation: "
          f"{operational_threshold:.4f} (strategy: {tuning['chosen_strategy']})")
    print("  threshold candidates:")
    for k, v in tuning.items():
        if isinstance(v, dict):
            print(f"    {k:18s} {v}")

    val_metrics = _metrics_at(y_val, proba_val, operational_threshold,
                              f"Validation ({val_years or 'holdout'}) — threshold source")
    val_metrics["threshold_tuning"] = tuning

    proba_test = model.predict_proba(X_test)[:, 1]
    test_metrics = _metrics_at(y_test, proba_test, operational_threshold,
                               f"Out-of-time test ({test_years}) at frozen threshold")

    if refit_all:
        # Extra files (e.g. the current year's partial season) join ONLY the
        # production refit — never the evaluation protocol above, which must
        # not train on anything newer than the test year.
        extra_df = (_load_concat(refit_extra_paths, "production-refit extra")
                    if refit_extra_paths else None)
        extra_years = _years_of(extra_df) if extra_df is not None else []
        production_years = sorted(set(train_years + val_years + test_years + extra_years))
        print(f"\n[train] --refit-all: refitting production model on all data "
              f"{production_years} (evaluation above stays the reported protocol) ...")
        frames = [train_df, val_df, test_df] + ([extra_df] if extra_df is not None else [])
        full = pd.concat(frames, ignore_index=True)
        del train_df, val_df, test_df, extra_df, rebalanced, X_train, y_train
        rebalanced_full = rebalance_training_set(full)
        del full
        spw_full = scale_pos_weight(rebalanced_full)
        X_full, y_full = _xy(rebalanced_full)
        del rebalanced_full
        print(f"[train] production fit: {len(y_full):,} rebalanced rows, "
              f"scale_pos_weight {spw_full:.3f}")
        production_model = _fit_xgb(X_full, y_full, spw_full)
    else:
        production_model = model
        production_years = sorted(set(train_years + val_years))

    importance = dict(zip(
        CFG.feature_columns,
        [float(v) for v in production_model.feature_importances_],
    ))
    importance_sorted = dict(sorted(importance.items(), key=lambda kv: -kv[1]))
    print("\n=== Top feature importances (production model) ===")
    for k, v in list(importance_sorted.items())[:10]:
        print(f"  {k:18s} {v:.4f}")

    production_model.save_model(str(CFG.model_path()))
    feature_meta = {
        "feature_columns": CFG.feature_columns,
        "operational_threshold": operational_threshold,
        "threshold_strategy": tuning["chosen_strategy"],
        "threshold_tuned_on": "validation",
        "decision_threshold": CFG.decision_threshold,  # legacy fallback only
        "trained_on": "synthetic" if synthetic else "gee",
        "data": {
            "train_years": train_years,
            "val_years": val_years,
            "test_years": test_years,
            "refit_all": bool(refit_all),
            "production_years": production_years,
        },
        # Legacy display keys (heatmap KPI panel and older consumers)
        "train_year": (f"{train_years[0]}-{train_years[-1]}" if len(train_years) > 1
                       else (str(train_years[0]) if train_years else None)),
        "test_year": test_years[-1] if test_years else None,
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
            # Aliases kept for predict.py / heatmap back-compat
            "operational_threshold": operational_threshold,
            "roc_auc": test_metrics["roc_auc"],
            "average_precision": test_metrics["average_precision"],
        },
        "feature_importance": importance_sorted,
    }
    CFG.feature_meta_path().write_text(json.dumps(feature_meta, indent=2))
    print(f"\n[train] saved model    -> {CFG.model_path()}")
    print(f"[train] saved metadata -> {CFG.feature_meta_path()}")

    return feature_meta


def main():
    parser = argparse.ArgumentParser(description="Train the West Coast wildfire ignition model.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on a synthetic dataset (no GEE auth required).")
    parser.add_argument("--train", type=Path, nargs="+", default=None,
                        help="One or more parquets of GEE-extracted training data (concatenated).")
    parser.add_argument("--val", type=Path, nargs="+", default=None,
                        help="Parquet(s) used ONLY for threshold tuning + fit monitoring. "
                             "Default: last ~20%% of training dates.")
    parser.add_argument("--test", type=Path, nargs="+", default=None,
                        help="Parquet(s) of GEE-extracted test data (touched once, at the frozen threshold).")
    parser.add_argument("--refit-all", action="store_true",
                        help="After evaluation, refit the production model on train+val+test combined.")
    parser.add_argument("--refit-extra", type=Path, nargs="*", default=None,
                        help="Extra parquet(s) added ONLY to the production refit (e.g. the "
                             "current year's partial season). Ignored without --refit-all.")
    args = parser.parse_args()
    train(synthetic=args.synthetic, train_paths=args.train,
          val_paths=args.val, test_paths=args.test,
          refit_all=args.refit_all, refit_extra_paths=args.refit_extra)


if __name__ == "__main__":
    main()
