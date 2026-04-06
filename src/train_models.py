"""
Train and evaluate classifiers on chess move features.

V1-V3: Decision Tree + Random Forest
V4: Decision Tree + Random Forest + XGBoost + threshold tuning

Reads data/features/features[_v*].csv, performs a 70/15/15 stratified split,
runs GridSearchCV for hyperparameter tuning on each model, evaluates on the
held-out test set, and persists trained models + results.

Output:
  data/models[_v*]/decision_tree.joblib
  data/models[_v*]/random_forest.joblib
  data/models[_v*]/xgboost.joblib        (V4 only)
  data/models[_v*]/thresholds.json       (V4 only)
  data/models[_v*]/results.csv
  data/models[_v*]/split_info.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

INPUT_CSV_V1 = Path("data/features/features.csv")
INPUT_CSV_V2 = Path("data/features/features_v2.csv")
INPUT_CSV_V3 = Path("data/features/features_v3.csv")
OUTPUT_DIR_V1 = Path("data/models")
OUTPUT_DIR_V2 = Path("data/models_v2")
OUTPUT_DIR_V3 = Path("data/models_v3")
OUTPUT_DIR_V4 = Path("data/models_v4")

RANDOM_STATE = 42


# ── Data loading & splitting ──────────────────────────────────────

def load_and_split(path: Path) -> tuple:
    """Load features CSV and return (X_train, X_val, X_test, y_train, y_val, y_test)."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1] - 1} features")

    X = df.drop(columns=["label"])
    y = (df["label"] == "ruim").astype(int)  # 1 = ruim, 0 = bom

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"  Train : {len(X_train):,}  (bom={int((y_train==0).sum()):,}  ruim={int((y_train==1).sum()):,})")
    print(f"  Val   : {len(X_val):,}  (bom={int((y_val==0).sum()):,}  ruim={int((y_val==1).sum()):,})")
    print(f"  Test  : {len(X_test):,}  (bom={int((y_test==0).sum()):,}  ruim={int((y_test==1).sum()):,})")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Model evaluation helper ──────────────────────────────────────

def evaluate(name: str, model, X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> dict:
    """Predict on X, print report, return metrics dict."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y, y_pred)
    f1_good = f1_score(y, y_pred, pos_label=0)
    f1_bad = f1_score(y, y_pred, pos_label=1)
    auc = roc_auc_score(y, y_proba) if y_proba is not None else float("nan")

    report = classification_report(y, y_pred, target_names=["bom", "ruim"])
    cm = confusion_matrix(y, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name}  —  Test set evaluation")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (bom) : {f1_good:.4f}")
    print(f"  F1 (ruim): {f1_bad:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{report}")
    print(f"Confusion matrix:\n{cm}")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "f1_bom": round(f1_good, 4),
        "f1_ruim": round(f1_bad, 4),
        "roc_auc": round(auc, 4),
        "recall_ruim": round(cm[1, 1] / (cm[1, 0] + cm[1, 1]), 4) if cm.shape == (2, 2) else float("nan"),
        "precision_ruim": round(cm[1, 1] / (cm[0, 1] + cm[1, 1]), 4) if cm.shape == (2, 2) else float("nan"),
    }


# ── Decision Tree ─────────────────────────────────────────────────

def train_decision_tree(X_train, y_train, feature_names: list[str]) -> DecisionTreeClassifier:
    print("\n>>> Training Decision Tree (GridSearchCV) …")
    param_grid = {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    best = grid.best_estimator_
    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")
    print(f"  Time        : {elapsed:.1f}s")

    rules = export_text(best, feature_names=feature_names, max_depth=4)
    print(f"\n  Top rules (depth ≤ 4):\n{rules[:2000]}")

    return best


# ── Random Forest ─────────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("\n>>> Training Random Forest (GridSearchCV) …")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 5, 10],
    }

    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")
    print(f"  Time        : {elapsed:.1f}s")

    return grid.best_estimator_


# ── XGBoost (V4) ──────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier

    print("\n>>> Training XGBoost (GridSearchCV) …")

    pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 5],
    }

    grid = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")
    print(f"  Time        : {elapsed:.1f}s")

    return grid.best_estimator_


# ── Threshold tuning ──────────────────────────────────────────────

def find_best_threshold(model, X_val, y_val) -> tuple[float, float]:
    """Find threshold that maximizes F1-ruim on the validation set."""
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.15, 0.70, 0.01)
    best_t, best_f1 = 0.50, 0.0

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return round(float(best_t), 2), round(float(best_f1), 4)


def evaluate_with_threshold(name: str, model, X, y, threshold: float) -> dict:
    """Evaluate model using a custom probability threshold."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    f1_good = f1_score(y, y_pred, pos_label=0)
    f1_bad = f1_score(y, y_pred, pos_label=1)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name}  —  threshold={threshold:.2f}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (bom) : {f1_good:.4f}")
    print(f"  F1 (ruim): {f1_bad:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    report = classification_report(y, y_pred, target_names=["bom", "ruim"])
    print(f"\n{report}")
    print(f"Confusion matrix:\n{cm}")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "f1_bom": round(f1_good, 4),
        "f1_ruim": round(f1_bad, 4),
        "roc_auc": round(auc, 4),
        "recall_ruim": round(cm[1, 1] / (cm[1, 0] + cm[1, 1]), 4) if cm.shape == (2, 2) else float("nan"),
        "precision_ruim": round(cm[1, 1] / (cm[0, 1] + cm[1, 1]), 4) if cm.shape == (2, 2) else float("nan"),
        "threshold": threshold,
    }


# ── Feature importance ────────────────────────────────────────────

def print_feature_importance(name: str, model, feature_names: list[str], top_n: int = 15) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    print(f"\n  Top {top_n} features — {name}:")
    for rank, idx in enumerate(indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s}  {importances[idx]:.4f}")


# ── Main pipeline ─────────────────────────────────────────────────

def run(v2: bool = False, v3: bool = False, v4: bool = False) -> None:
    if v4:
        input_csv, output_dir, tag = INPUT_CSV_V3, OUTPUT_DIR_V4, "V4"
    elif v3:
        input_csv, output_dir, tag = INPUT_CSV_V3, OUTPUT_DIR_V3, "V3"
    elif v2:
        input_csv, output_dir, tag = INPUT_CSV_V2, OUTPUT_DIR_V2, "V2"
    else:
        input_csv, output_dir, tag = INPUT_CSV_V1, OUTPUT_DIR_V1, "V1"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Pipeline {tag}  —  input: {input_csv}")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(input_csv)
    feature_names = list(X_train.columns)

    split_info = pd.DataFrame({
        "split": ["train", "val", "test"],
        "total": [len(X_train), len(X_val), len(X_test)],
        "bom": [int((y_train == 0).sum()), int((y_val == 0).sum()), int((y_test == 0).sum())],
        "ruim": [int((y_train == 1).sum()), int((y_val == 1).sum()), int((y_test == 1).sum())],
    })
    split_info.to_csv(output_dir / "split_info.csv", index=False)

    # ── Train models
    dt = train_decision_tree(X_train, y_train, feature_names)
    rf = train_random_forest(X_train, y_train)

    # ── Save models
    joblib.dump(dt, output_dir / "decision_tree.joblib")
    joblib.dump(rf, output_dir / "random_forest.joblib")

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    results = []

    if v4:
        # ── Train XGBoost
        xgb = train_xgboost(X_train, y_train)
        joblib.dump(xgb, output_dir / "xgboost.joblib")

        # ── Threshold tuning on validation set
        print(f"\n{'='*60}")
        print("  Threshold tuning on validation set")
        print(f"{'='*60}")

        thresholds = {}
        for name, model in [("Decision Tree", dt), ("Random Forest", rf), ("XGBoost", xgb)]:
            best_t, best_f1 = find_best_threshold(model, X_val, y_val)
            thresholds[name] = best_t
            print(f"  {name:20s}  best_threshold={best_t:.2f}  val_F1={best_f1:.4f}")

        with open(output_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)

        # ── Evaluate all models with default threshold (0.50)
        print(f"\n{'='*60}")
        print("  Test set evaluation — default threshold (0.50)")
        print(f"{'='*60}")
        results.append(evaluate("Decision Tree", dt, X_test, y_test, feature_names))
        results.append(evaluate("Random Forest", rf, X_test, y_test, feature_names))
        results.append(evaluate("XGBoost", xgb, X_test, y_test, feature_names))

        # ── Evaluate all models with tuned thresholds
        print(f"\n{'='*60}")
        print("  Test set evaluation — tuned thresholds")
        print(f"{'='*60}")
        for name, model in [("Decision Tree", dt), ("Random Forest", rf), ("XGBoost", xgb)]:
            t = thresholds[name]
            r = evaluate_with_threshold(f"{name} (t={t:.2f})", model, X_test, y_test, t)
            results.append(r)

        print_feature_importance("Decision Tree", dt, feature_names)
        print_feature_importance("Random Forest", rf, feature_names)
        print_feature_importance("XGBoost", xgb, feature_names)
    else:
        # ── V1/V2/V3: evaluate with default threshold
        results.append(evaluate("Decision Tree", dt, X_test, y_test, feature_names))
        results.append(evaluate("Random Forest", rf, X_test, y_test, feature_names))

        print_feature_importance("Decision Tree", dt, feature_names)
        print_feature_importance("Random Forest", rf, feature_names)

    print(f"\nModels saved to {output_dir}/")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "results.csv", index=False)
    print(f"\n{'='*60}")
    print(f"Results summary saved to {output_dir / 'results.csv'}")
    print(df_results.to_string(index=False))


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifiers on chess features.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--v2", action="store_true", help="Use V2 features (52 features incl. tactical)")
    group.add_argument("--v3", action="store_true", help="Use V3 features (67 features incl. look-ahead)")
    group.add_argument("--v4", action="store_true", help="V3 features + XGBoost + threshold tuning")
    args = parser.parse_args()
    run(v2=args.v2, v3=args.v3, v4=args.v4)


if __name__ == "__main__":
    main()
