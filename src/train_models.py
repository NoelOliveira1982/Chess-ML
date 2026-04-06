"""
Train and evaluate two classifiers (Decision Tree + Random Forest) on chess
move features.

Reads data/features/features.csv, performs a 70/15/15 stratified split,
runs GridSearchCV for hyperparameter tuning on each model, evaluates on the
held-out test set, and persists trained models + results.

Output:
  data/models/decision_tree.joblib
  data/models/random_forest.joblib
  data/models/results.csv          (per-model metrics summary)
  data/models/split_info.csv       (split sizes and class distributions)
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

INPUT_CSV = Path("data/features/features.csv")
OUTPUT_DIR = Path("data/models")

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


# ── Feature importance ────────────────────────────────────────────

def print_feature_importance(name: str, model, feature_names: list[str], top_n: int = 15) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    print(f"\n  Top {top_n} features — {name}:")
    for rank, idx in enumerate(indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s}  {importances[idx]:.4f}")


# ── Main pipeline ─────────────────────────────────────────────────

def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(INPUT_CSV)
    feature_names = list(X_train.columns)

    split_info = pd.DataFrame({
        "split": ["train", "val", "test"],
        "total": [len(X_train), len(X_val), len(X_test)],
        "bom": [int((y_train == 0).sum()), int((y_val == 0).sum()), int((y_test == 0).sum())],
        "ruim": [int((y_train == 1).sum()), int((y_val == 1).sum()), int((y_test == 1).sum())],
    })
    split_info.to_csv(OUTPUT_DIR / "split_info.csv", index=False)

    # ── Train models
    dt = train_decision_tree(X_train, y_train, feature_names)
    rf = train_random_forest(X_train, y_train)

    # ── Save models
    joblib.dump(dt, OUTPUT_DIR / "decision_tree.joblib")
    joblib.dump(rf, OUTPUT_DIR / "random_forest.joblib")
    print(f"\nModels saved to {OUTPUT_DIR}/")

    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # ── Evaluate on test set
    results = []
    results.append(evaluate("Decision Tree", dt, X_test, y_test, feature_names))
    results.append(evaluate("Random Forest", rf, X_test, y_test, feature_names))

    print_feature_importance("Decision Tree", dt, feature_names)
    print_feature_importance("Random Forest", rf, feature_names)

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\n{'='*60}")
    print(f"Results summary saved to {OUTPUT_DIR / 'results.csv'}")
    print(df_results.to_string(index=False))


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Decision Tree + Random Forest on chess features.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
