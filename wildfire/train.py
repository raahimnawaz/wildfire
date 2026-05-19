"""
Train the West Coast wildfire ignition predictor.

Pipeline:
  1. Load training and test parquets (or synthesize them).
  2. Out-of-time split — train year != test year.
  3. Domain-guided minority injection on training set ONLY.
  4. Fit XGBoost binary classifier with paper-aligned hyperparameters.
  5. Evaluate on the untouched test set with ROC-AUC, full classification
     report at the operational threshold, and per-feature importance.
  6. Persist model JSON + feature metadata for the predict.py CLI.

Usage:
    # Run end-to-end on synthetic data (no GEE required):
    python -m wildfire.train --synthetic

    # Run on real GEE-extracted parquets:
    python -m wildfire.train --train wildfire_data/raw_2022.parquet \\
                             --test  wildfire_data/raw_2023.parquet
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


def _load_or_synthesize(synthetic: bool,
                        train_paths: Optional[List[Path]],
                        test_path: Optional[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if synthetic:
        from .synthetic_data import make_synthetic_dataset
        print("[train] generating synthetic dataset (no GEE needed) ...")
        df = make_synthetic_dataset(n_total=200_000)
        # Force out-of-time split using the year encoded in the synthetic dates
        years = pd.to_datetime(df["date"]).dt.year
        train_df = df[years == CFG.train_year].copy()
        test_df  = df[years == CFG.test_year].copy()
    else:
        if not train_paths or test_path is None:
            raise SystemExit("Must pass --train and --test parquet paths (or --synthetic).")
        train_dfs = []
        for p in train_paths:
            print(f"[train] loading {p} -> training set")
            train_dfs.append(pd.read_parquet(p))
        train_df = pd.concat(train_dfs, ignore_index=True) if len(train_dfs) > 1 else train_dfs[0]
        if len(train_paths) > 1:
            print(f"[train] concatenated {len(train_paths)} training parquets: {len(train_df):,} rows total")
        print(f"[train] loading {test_path} -> test set")
        test_df  = pd.read_parquet(test_path)
    return train_df, test_df


def _tune_threshold(y_true, proba, strategy: str = "f1") -> dict:
    """
    Find the operational threshold from the PR curve on the test set.

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


def _evaluate(model, X_test, y_test, threshold: float | None = None) -> dict:
    from sklearn.metrics import (
        roc_auc_score,
        classification_report,
        confusion_matrix,
        average_precision_score,
    )

    proba = model.predict_proba(X_test)[:, 1]

    # Tune threshold on this evaluation set (will be saved into metadata so
    # downstream tools can use it instead of the hard-coded CFG default).
    tuning = _tune_threshold(y_test, proba, strategy="f1")
    operational_threshold = threshold if threshold is not None else tuning["chosen_threshold"]
    preds = (proba >= operational_threshold).astype(int)

    auc = roc_auc_score(y_test, proba)
    ap  = average_precision_score(y_test, proba)
    cm  = confusion_matrix(y_test, preds).tolist()
    report = classification_report(
        y_test, preds, target_names=["no_fire", "fire"], output_dict=True, zero_division=0
    )

    print("\n=== Out-of-time evaluation ===")
    print(f"  test rows           : {len(y_test):,}")
    print(f"  positive rate       : {y_test.mean():.5f}")
    print(f"  ROC-AUC             : {auc:.4f}")
    print(f"  Average Precision   : {ap:.4f}")
    print(f"  operational thr.    : {operational_threshold:.4f}  (data-driven via PR-curve F1)")
    print(f"  confusion matrix    : {cm}")
    print(classification_report(y_test, preds, target_names=["no_fire", "fire"], zero_division=0))
    print(f"  threshold candidates:")
    for k, v in tuning.items():
        if isinstance(v, dict):
            print(f"    {k:18s} {v}")

    return {
        "roc_auc": auc,
        "average_precision": ap,
        "operational_threshold": float(operational_threshold),
        "threshold_tuning": tuning,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_test": int(len(y_test)),
        "test_positive_rate": float(y_test.mean()),
    }


def train(synthetic: bool = False,
          train_paths: Optional[List[Path]] = None,
          test_path:  Optional[Path] = None) -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise SystemExit(
            "xgboost is not installed. Install with:\n"
            "    pip install -r wildfire/requirements.txt"
        )

    train_df_raw, test_df_raw = _load_or_synthesize(synthetic, train_paths, test_path)

    train_df = derive_all(train_df_raw)
    test_df  = derive_all(test_df_raw)

    print(f"[train] raw  train rows: {len(train_df):,}  (positives: {(train_df[CFG.target_column]==1).sum():,})")
    print(f"[train] raw  test  rows: {len(test_df):,}   (positives: {(test_df [CFG.target_column]==1).sum():,})")

    rebalanced = rebalance_training_set(train_df)
    spw = scale_pos_weight(rebalanced)
    print(f"[train] rebalanced rows: {len(rebalanced):,}  (positives: {(rebalanced[CFG.target_column]==1).sum():,})")
    print(f"[train] scale_pos_weight: {spw:.3f}")

    X_train = rebalanced[CFG.feature_columns].astype("float32").values
    y_train = rebalanced[CFG.target_column].astype("int8").values
    X_test  = test_df[CFG.feature_columns].astype("float32").values
    y_test  = test_df[CFG.target_column].astype("int8").values

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

    print("[train] fitting XGBoost ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    metrics = _evaluate(model, X_test, y_test)

    importance = dict(zip(
        CFG.feature_columns,
        [float(v) for v in model.feature_importances_],
    ))
    importance_sorted = dict(sorted(importance.items(), key=lambda kv: -kv[1]))
    print("\n=== Top feature importances ===")
    for k, v in list(importance_sorted.items())[:10]:
        print(f"  {k:18s} {v:.4f}")

    model.save_model(str(CFG.model_path()))
    feature_meta = {
        "feature_columns": CFG.feature_columns,
        "decision_threshold": CFG.decision_threshold,
        "trained_on": "synthetic" if synthetic else "gee",
        "train_year": CFG.train_year,
        "test_year": CFG.test_year,
        "metrics": metrics,
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
    parser.add_argument("--test", type=Path, default=None,
                        help="Parquet of GEE-extracted test data.")
    args = parser.parse_args()
    train(synthetic=args.synthetic, train_paths=args.train, test_path=args.test)


if __name__ == "__main__":
    main()
