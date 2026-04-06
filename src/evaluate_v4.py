"""
Evaluate V4 models and generate V1-vs-V2-vs-V3-vs-V4 comparison artifacts.

Reads:
  data/models/decision_tree.joblib        (V1)
  data/models/random_forest.joblib        (V1)
  data/models_v2/decision_tree.joblib     (V2)
  data/models_v2/random_forest.joblib     (V2)
  data/models_v3/decision_tree.joblib     (V3)
  data/models_v3/random_forest.joblib     (V3)
  data/models_v4/decision_tree.joblib     (V4)
  data/models_v4/random_forest.joblib     (V4)
  data/models_v4/xgboost.joblib           (V4)
  data/models_v4/thresholds.json          (V4)
  data/features/features.csv              (V1)
  data/features/features_v2.csv           (V2)
  data/features/features_v3.csv           (V3 + V4)
  data/labeled/moves_labeled.csv

Output (all saved to data/evaluation_v4/):
  confusion_matrices_v4.png
  feature_importance_xgb_v4.png
  feature_importance_3way_v4.png
  roc_curves_v4.png
  precision_recall_curves_v4.png
  threshold_analysis_v4.png
  error_analysis_fp_v4.csv
  error_analysis_fn_v4.csv
  v1_v2_v3_v4_metrics.csv
  v1_v2_v3_v4_deltas.csv
  v1_v2_v3_v4_comparison.png
  v1_v2_v3_v4_roc_overlay.png
  v1_v2_v3_v4_pr_overlay.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

MODELS_DIR_V1 = Path("data/models")
MODELS_DIR_V2 = Path("data/models_v2")
MODELS_DIR_V3 = Path("data/models_v3")
MODELS_DIR_V4 = Path("data/models_v4")
FEATURES_V1 = Path("data/features/features.csv")
FEATURES_V2 = Path("data/features/features_v2.csv")
FEATURES_V3 = Path("data/features/features_v3.csv")
LABELED_CSV = Path("data/labeled/moves_labeled.csv")
OUTPUT_DIR = Path("data/evaluation_v4")

RANDOM_STATE = 42

FEATURE_TRANSLATION = {
    "white_pawns": "Peões brancos",
    "white_knights": "Cavalos brancos",
    "white_bishops": "Bispos brancos",
    "white_rooks": "Torres brancas",
    "white_queens": "Damas brancas",
    "black_pawns": "Peões pretos",
    "black_knights": "Cavalos pretos",
    "black_bishops": "Bispos pretos",
    "black_rooks": "Torres pretas",
    "black_queens": "Damas pretas",
    "material_diff": "Diferença material",
    "legal_moves_player": "Lances legais (jogador)",
    "legal_moves_opponent": "Lances legais (adversário)",
    "mobility_diff": "Diferença de mobilidade",
    "player_castled": "Jogador rocou",
    "opponent_castled": "Adversário rocou",
    "player_can_castle": "Pode rocar",
    "king_pawn_shield": "Escudo de peões do rei",
    "player_doubled_pawns": "Peões dobrados",
    "player_isolated_pawns": "Peões isolados",
    "player_passed_pawns": "Peões passados (jogador)",
    "opponent_passed_pawns": "Peões passados (adversário)",
    "player_center_control": "Controle do centro (jogador)",
    "opponent_center_control": "Controle do centro (adversário)",
    "player_center_occupation": "Ocupação do centro",
    "is_capture": "É captura",
    "is_check": "Dá xeque",
    "is_promotion": "É promoção",
    "moved_piece": "Peça movida",
    "move_to_center": "Move para o centro",
    "move_to_extended_center": "Move para centro expandido",
    "move_number": "Número do lance",
    "is_white": "Joga de brancas",
    # V2 tactical features
    "hanging_pieces_player": "Peças indefesas (jogador)",
    "hanging_pieces_opponent": "Peças indefesas (adversário)",
    "hanging_value_player": "Valor indefeso (jogador)",
    "hanging_value_opponent": "Valor indefeso (adversário)",
    "min_attacker_vs_piece_player": "Menor atacante vs peça (jogador)",
    "threats_against_player": "Ameaças contra jogador",
    "threats_against_opponent": "Ameaças contra adversário",
    "max_threat_value_player": "Maior ameaça (jogador)",
    "max_threat_value_opponent": "Maior ameaça (adversário)",
    "pinned_pieces_player": "Peças cravadas (jogador)",
    "pinned_pieces_opponent": "Peças cravadas (adversário)",
    "king_attackers_player": "Atacantes do rei (jogador)",
    "king_attackers_opponent": "Atacantes do rei (adversário)",
    "king_open_files_player": "Colunas abertas do rei (jogador)",
    "king_escape_squares_player": "Casas de fuga do rei (jogador)",
    "total_attacks_player": "Ataques totais (jogador)",
    "total_attacks_opponent": "Ataques totais (adversário)",
    "contested_squares": "Casas disputadas",
    "undefended_pieces_player": "Peças sem defesa (jogador)",
    # V3 look-ahead features
    "delta_hanging_player": "Δ peças indefesas (jogador)",
    "delta_hanging_opponent": "Δ peças indefesas (adversário)",
    "delta_hanging_value_player": "Δ valor indefeso (jogador)",
    "delta_threats_against_player": "Δ ameaças contra jogador",
    "delta_mobility_player": "Δ mobilidade (jogador)",
    "delta_mobility_opponent": "Δ mobilidade (adversário)",
    "delta_contested_squares": "Δ casas disputadas",
    "delta_king_attackers_player": "Δ atacantes do rei (jogador)",
    "opponent_best_capture_value": "Melhor captura adversário",
    "opponent_can_check": "Adversário pode dar xeque",
    "opponent_num_good_captures": "Nº capturas boas adversário",
    "created_hanging_self": "Criou peça indefesa",
    "see_of_move": "SEE do lance",
    "worst_see_against_player": "Pior SEE contra jogador",
    "is_losing_capture": "É captura perdedora",
}

TACTICAL_FEATURES = [
    "hanging_pieces_player", "hanging_pieces_opponent",
    "hanging_value_player", "hanging_value_opponent",
    "min_attacker_vs_piece_player",
    "threats_against_player", "threats_against_opponent",
    "max_threat_value_player", "max_threat_value_opponent",
    "pinned_pieces_player", "pinned_pieces_opponent",
    "king_attackers_player", "king_attackers_opponent",
    "king_open_files_player", "king_escape_squares_player",
    "total_attacks_player", "total_attacks_opponent",
    "contested_squares", "undefended_pieces_player",
]

LOOKAHEAD_FEATURES = [
    "delta_hanging_player", "delta_hanging_opponent",
    "delta_hanging_value_player", "delta_threats_against_player",
    "delta_mobility_player", "delta_mobility_opponent",
    "delta_contested_squares", "delta_king_attackers_player",
    "opponent_best_capture_value", "opponent_can_check",
    "opponent_num_good_captures", "created_hanging_self",
    "see_of_move", "worst_see_against_player", "is_losing_capture",
]


def _translate(name: str) -> str:
    return FEATURE_TRANSLATION.get(name, name)


def _feature_group(name: str) -> str:
    if name in LOOKAHEAD_FEATURES:
        return "v3"
    if name in TACTICAL_FEATURES:
        return "v2"
    return "v1"


GROUP_COLORS = {"v1": "#4C72B0", "v2": "#C44E52", "v3": "#55A868"}
GROUP_LABELS = {
    "v1": "V1 (posicional)",
    "v2": "V2 (tática)",
    "v3": "V3 (look-ahead)",
}


# ── Data loading ──────────────────────────────────────────────────

def _split(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = (df["label"] == "ruim").astype(int)
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X, y, df.index,
        test_size=0.15, stratify=y, random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=0.176, stratify=y_temp, random_state=RANDOM_STATE,
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "idx_test": idx_test, "feature_names": list(X.columns),
    }


def load_all():
    df_v1 = pd.read_csv(FEATURES_V1)
    df_v2 = pd.read_csv(FEATURES_V2)
    df_v3 = pd.read_csv(FEATURES_V3)
    df_labeled = pd.read_csv(LABELED_CSV)

    dt_v1 = joblib.load(MODELS_DIR_V1 / "decision_tree.joblib")
    rf_v1 = joblib.load(MODELS_DIR_V1 / "random_forest.joblib")
    dt_v2 = joblib.load(MODELS_DIR_V2 / "decision_tree.joblib")
    rf_v2 = joblib.load(MODELS_DIR_V2 / "random_forest.joblib")
    dt_v3 = joblib.load(MODELS_DIR_V3 / "decision_tree.joblib")
    rf_v3 = joblib.load(MODELS_DIR_V3 / "random_forest.joblib")
    dt_v4 = joblib.load(MODELS_DIR_V4 / "decision_tree.joblib")
    rf_v4 = joblib.load(MODELS_DIR_V4 / "random_forest.joblib")
    xgb_v4 = joblib.load(MODELS_DIR_V4 / "xgboost.joblib")

    with open(MODELS_DIR_V4 / "thresholds.json") as f:
        thresholds_v4 = json.load(f)

    data_v1 = _split(df_v1)
    data_v2 = _split(df_v2)
    data_v3 = _split(df_v3)
    data_v4 = _split(df_v3)  # V4 uses the same features as V3

    return {
        "v1": {"dt": dt_v1, "rf": rf_v1, **data_v1},
        "v2": {"dt": dt_v2, "rf": rf_v2, **data_v2},
        "v3": {"dt": dt_v3, "rf": rf_v3, **data_v3},
        "v4": {"dt": dt_v4, "rf": rf_v4, "xgb": xgb_v4,
               "thresholds": thresholds_v4, **data_v4},
        "df_labeled": df_labeled,
    }


# ── Metrics helpers ───────────────────────────────────────────────

def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_bom": f1_score(y_test, y_pred, pos_label=0),
        "f1_ruim": f1_score(y_test, y_pred, pos_label=1),
        "recall_ruim": recall_score(y_test, y_pred, pos_label=1),
        "precision_ruim": precision_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


# ── V4 confusion matrices ────────────────────────────────────────

def plot_confusion_matrices_v4(dt, rf, xgb, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, model, name in [
        (axes[0], dt, "Decision Tree V4"),
        (axes[1], rf, "Random Forest V4"),
        (axes[2], xgb, "XGBoost V4"),
    ]:
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=["bom", "ruim"],
            cmap="Blues", ax=ax, values_format="d",
        )
        ax.set_title(f"Matriz de Confusão — {name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ confusion_matrices_v4.png")


# ── V4 feature importance ────────────────────────────────────────

def plot_feature_importance_xgb_v4(model, feature_names, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    labels = [_translate(feature_names[i]) for i in indices]
    values = importances[indices]
    colors = [GROUP_COLORS[_feature_group(feature_names[i])] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância")
    ax.set_title("Top 15 Features — XGBoost V4", fontsize=14)

    legend_handles = [
        Patch(facecolor=c, label=l) for c, l in
        [(GROUP_COLORS["v1"], GROUP_LABELS["v1"]),
         (GROUP_COLORS["v2"], GROUP_LABELS["v2"]),
         (GROUP_COLORS["v3"], GROUP_LABELS["v3"])]
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_xgb_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ feature_importance_xgb_v4.png")


def plot_feature_importance_3way_v4(dt, rf, xgb, feature_names, top_n=15):
    imp_dt = dt.feature_importances_
    imp_rf = rf.feature_importances_
    imp_xgb = xgb.feature_importances_
    all_idx = (
        set(np.argsort(imp_dt)[::-1][:top_n])
        | set(np.argsort(imp_rf)[::-1][:top_n])
        | set(np.argsort(imp_xgb)[::-1][:top_n])
    )
    combined = sorted(
        all_idx, key=lambda i: imp_rf[i] + imp_dt[i] + imp_xgb[i], reverse=True,
    )[:top_n]

    labels = [_translate(feature_names[i]) for i in combined]
    vals_dt = [imp_dt[i] for i in combined]
    vals_rf = [imp_rf[i] for i in combined]
    vals_xgb = [imp_xgb[i] for i in combined]

    y_pos = np.arange(top_n)
    height = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(y_pos - height, vals_dt, height, label="Decision Tree V4", color="#4C72B0")
    ax.barh(y_pos, vals_rf, height, label="Random Forest V4", color="#DD8452")
    ax.barh(y_pos + height, vals_xgb, height, label="XGBoost V4", color="#2CA02C")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância")
    ax.set_title("Comparação de Feature Importance — DT V4 vs RF V4 vs XGB V4", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_3way_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ feature_importance_3way_v4.png")


# ── V4 ROC & PR curves ───────────────────────────────────────────

def plot_roc_v4(dt, rf, xgb, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(dt, X_test, y_test, ax=ax, name="Decision Tree V4")
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest V4")
    RocCurveDisplay.from_estimator(xgb, X_test, y_test, ax=ax, name="XGBoost V4")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random baseline")
    ax.set_title("Curva ROC — V4 — Classe 'ruim'", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ roc_curves_v4.png")


def plot_pr_v4(dt, rf, xgb, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(dt, X_test, y_test, ax=ax, name="Decision Tree V4")
    PrecisionRecallDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest V4")
    PrecisionRecallDisplay.from_estimator(xgb, X_test, y_test, ax=ax, name="XGBoost V4")
    prevalence = y_test.mean()
    ax.axhline(y=prevalence, color="k", linestyle="--", alpha=0.5,
               label=f"Prevalência ({prevalence:.3f})")
    ax.set_title("Curva Precision-Recall — V4 — Classe 'ruim'", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curves_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ precision_recall_curves_v4.png")


# ── V4 threshold analysis ────────────────────────────────────────

def plot_threshold_analysis_v4(dt, rf, xgb, X_test, y_test, thresholds):
    thresholds_sweep = np.arange(0.15, 0.71, 0.01)

    model_info = [
        ("Decision Tree", dt, "#4C72B0"),
        ("Random Forest", rf, "#DD8452"),
        ("XGBoost", xgb, "#2CA02C"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, model, color in model_info:
        proba = model.predict_proba(X_test)[:, 1]
        f1_scores = []
        for t in thresholds_sweep:
            y_pred_t = (proba >= t).astype(int)
            f1_scores.append(f1_score(y_test, y_pred_t, pos_label=1, zero_division=0))
        ax.plot(thresholds_sweep, f1_scores, color=color, label=name, linewidth=2)

    for name, _, color in model_info:
        if name in thresholds:
            opt_t = thresholds[name]
            ax.axvline(x=opt_t, color=color, linestyle="--", alpha=0.7,
                       label=f"{name} thresh={opt_t:.2f}")

    ax.axhline(y=0.50, color="gray", linestyle="--", alpha=0.5, label="F1 = 0.50 (meta)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-score (classe ruim)")
    ax.set_title("Análise de Threshold — V4 — F1-ruim vs Threshold", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_analysis_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ threshold_analysis_v4.png")


# ── V4 error analysis ────────────────────────────────────────────

def error_analysis_v4(model, model_name, X_test, y_test, idx_test,
                      df_labeled, feature_names, n=10):
    y_pred = model.predict(X_test)
    fp_mask = (y_pred == 1) & (y_test.values == 0)
    fn_mask = (y_pred == 0) & (y_test.values == 1)
    fp_indices = idx_test[fp_mask]
    fn_indices = idx_test[fn_mask]

    rng = np.random.RandomState(RANDOM_STATE)
    fp_sample = rng.choice(fp_indices, size=min(n, len(fp_indices)), replace=False)
    fn_sample = rng.choice(fn_indices, size=min(n, len(fn_indices)), replace=False)

    def build_rows(indices, error_type):
        rows = []
        for idx in indices:
            labeled = df_labeled.iloc[idx]
            feat_row = X_test.loc[idx] if idx in X_test.index else None
            top_features = ""
            if feat_row is not None:
                imp = model.feature_importances_
                top_idx = np.argsort(imp)[::-1][:5]
                top_features = "; ".join(
                    f"{feature_names[i]}={feat_row.iloc[i]}" for i in top_idx
                )
            rows.append({
                "error_type": error_type,
                "model": model_name,
                "game_site": labeled.get("game_site", ""),
                "move_number": labeled.get("move_number", ""),
                "color": labeled.get("color", ""),
                "fen_before": labeled.get("fen_before", ""),
                "move_san": labeled.get("move_san", ""),
                "move_uci": labeled.get("move_uci", ""),
                "delta_cp": labeled.get("delta_cp", ""),
                "label_real": labeled.get("label", ""),
                "label_pred": "ruim" if error_type == "FP" else "bom",
                "top_features": top_features,
            })
        return rows

    return build_rows(fp_sample, "FP"), build_rows(fn_sample, "FN")


# ── V1 vs V2 vs V3 vs V4 comparison ──────────────────────────────

def build_comparison_table(all_data):
    rows = []
    for version in ["V1", "V2", "V3"]:
        vdata = all_data[version.lower()]
        for name, key in [("Decision Tree", "dt"), ("Random Forest", "rf")]:
            m = compute_metrics(vdata[key], vdata["X_test"], vdata["y_test"])
            m["model"] = f"{name} {version}"
            m["version"] = version
            m["algo"] = name
            rows.append(m)

    v4 = all_data["v4"]
    for name, key in [("Decision Tree", "dt"), ("Random Forest", "rf"),
                       ("XGBoost", "xgb")]:
        m = compute_metrics(v4[key], v4["X_test"], v4["y_test"])
        m["model"] = f"{name} V4"
        m["version"] = "V4"
        m["algo"] = name
        rows.append(m)

    df = pd.DataFrame(rows)

    delta_rows = []
    for algo in ["Decision Tree", "Random Forest"]:
        v1 = df[(df["algo"] == algo) & (df["version"] == "V1")].iloc[0]
        v2 = df[(df["algo"] == algo) & (df["version"] == "V2")].iloc[0]
        v3 = df[(df["algo"] == algo) & (df["version"] == "V3")].iloc[0]
        v4_row = df[(df["algo"] == algo) & (df["version"] == "V4")].iloc[0]
        for metric in ["accuracy", "f1_bom", "f1_ruim", "recall_ruim",
                        "precision_ruim", "roc_auc"]:
            delta_rows.append({
                "algo": algo,
                "metric": metric,
                "v1": round(v1[metric], 4),
                "v2": round(v2[metric], 4),
                "v3": round(v3[metric], 4),
                "v4": round(v4_row[metric], 4),
                "delta_v2_v1": round(v2[metric] - v1[metric], 4),
                "delta_v3_v2": round(v3[metric] - v2[metric], 4),
                "delta_v4_v3": round(v4_row[metric] - v3[metric], 4),
                "delta_v4_v1": round(v4_row[metric] - v1[metric], 4),
                "delta_v2_v1_pp": round((v2[metric] - v1[metric]) * 100, 2),
                "delta_v3_v2_pp": round((v3[metric] - v2[metric]) * 100, 2),
                "delta_v4_v3_pp": round((v4_row[metric] - v3[metric]) * 100, 2),
                "delta_v4_v1_pp": round((v4_row[metric] - v1[metric]) * 100, 2),
            })

    # XGBoost only exists in V4
    xgb_v4 = df[(df["algo"] == "XGBoost") & (df["version"] == "V4")].iloc[0]
    for metric in ["accuracy", "f1_bom", "f1_ruim", "recall_ruim",
                    "precision_ruim", "roc_auc"]:
        delta_rows.append({
            "algo": "XGBoost",
            "metric": metric,
            "v1": None,
            "v2": None,
            "v3": None,
            "v4": round(xgb_v4[metric], 4),
            "delta_v2_v1": None,
            "delta_v3_v2": None,
            "delta_v4_v3": None,
            "delta_v4_v1": None,
            "delta_v2_v1_pp": None,
            "delta_v3_v2_pp": None,
            "delta_v4_v3_pp": None,
            "delta_v4_v1_pp": None,
        })

    df_delta = pd.DataFrame(delta_rows)
    df.to_csv(OUTPUT_DIR / "v1_v2_v3_v4_metrics.csv", index=False)
    df_delta.to_csv(OUTPUT_DIR / "v1_v2_v3_v4_deltas.csv", index=False)
    print("  ✓ v1_v2_v3_v4_metrics.csv")
    print("  ✓ v1_v2_v3_v4_deltas.csv")
    return df, df_delta


def plot_v1_v2_v3_v4_comparison(all_data):
    """9-bar grouped chart: DT V1/V2/V3/V4, RF V1/V2/V3/V4, XGB V4."""
    metric_keys = ["accuracy", "f1_ruim", "recall_ruim", "precision_ruim", "roc_auc"]
    metric_labels = ["Accuracy", "F1 (ruim)", "Recall (ruim)", "Precision (ruim)", "ROC-AUC"]

    models = [
        ("DT V1", all_data["v1"]["dt"], all_data["v1"]),
        ("DT V2", all_data["v2"]["dt"], all_data["v2"]),
        ("DT V3", all_data["v3"]["dt"], all_data["v3"]),
        ("DT V4", all_data["v4"]["dt"], all_data["v4"]),
        ("RF V1", all_data["v1"]["rf"], all_data["v1"]),
        ("RF V2", all_data["v2"]["rf"], all_data["v2"]),
        ("RF V3", all_data["v3"]["rf"], all_data["v3"]),
        ("RF V4", all_data["v4"]["rf"], all_data["v4"]),
        ("XGB V4", all_data["v4"]["xgb"], all_data["v4"]),
    ]

    colors = [
        "#C6DBEF", "#6BAED6", "#2171B5", "#08519C",
        "#FDCDAC", "#FC8D62", "#D7301F", "#7F2704",
        "#2CA02C",
    ]

    all_vals = {}
    for label, model, vdata in models:
        m = compute_metrics(model, vdata["X_test"], vdata["y_test"])
        all_vals[label] = [m[k] for k in metric_keys]

    x = np.arange(len(metric_labels))
    n_bars = len(models)
    width = 0.10
    fig, ax = plt.subplots(figsize=(17, 7))

    for i, (label, _, _) in enumerate(models):
        vals = all_vals[label]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x + (n_bars - 1) / 2 * width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparação V1 vs V2 vs V3 vs V4 — Conjunto de Teste", fontsize=14)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v1_v2_v3_v4_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ v1_v2_v3_v4_comparison.png")


def plot_v1_v2_v3_v4_roc_overlay(all_data):
    fig, ax = plt.subplots(figsize=(10, 7))

    pairs = [
        ("DT V1", all_data["v1"]["dt"], all_data["v1"], "#C6DBEF", ":"),
        ("DT V2", all_data["v2"]["dt"], all_data["v2"], "#6BAED6", "--"),
        ("DT V3", all_data["v3"]["dt"], all_data["v3"], "#2171B5", "-."),
        ("DT V4", all_data["v4"]["dt"], all_data["v4"], "#08519C", "-"),
        ("RF V1", all_data["v1"]["rf"], all_data["v1"], "#FDCDAC", ":"),
        ("RF V2", all_data["v2"]["rf"], all_data["v2"], "#FC8D62", "--"),
        ("RF V3", all_data["v3"]["rf"], all_data["v3"], "#D7301F", "-."),
        ("RF V4", all_data["v4"]["rf"], all_data["v4"], "#7F2704", "-"),
        ("XGB V4", all_data["v4"]["xgb"], all_data["v4"], "#2CA02C", "-"),
    ]
    for label, model, vdata, color, ls in pairs:
        RocCurveDisplay.from_estimator(
            model, vdata["X_test"], vdata["y_test"],
            ax=ax, name=label, linestyle=ls, color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("Curva ROC — V1 vs V2 vs V3 vs V4 — Classe 'ruim'", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v1_v2_v3_v4_roc_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ v1_v2_v3_v4_roc_overlay.png")


def plot_v1_v2_v3_v4_pr_overlay(all_data):
    fig, ax = plt.subplots(figsize=(10, 7))

    pairs = [
        ("DT V1", all_data["v1"]["dt"], all_data["v1"], "#C6DBEF", ":"),
        ("DT V2", all_data["v2"]["dt"], all_data["v2"], "#6BAED6", "--"),
        ("DT V3", all_data["v3"]["dt"], all_data["v3"], "#2171B5", "-."),
        ("DT V4", all_data["v4"]["dt"], all_data["v4"], "#08519C", "-"),
        ("RF V1", all_data["v1"]["rf"], all_data["v1"], "#FDCDAC", ":"),
        ("RF V2", all_data["v2"]["rf"], all_data["v2"], "#FC8D62", "--"),
        ("RF V3", all_data["v3"]["rf"], all_data["v3"], "#D7301F", "-."),
        ("RF V4", all_data["v4"]["rf"], all_data["v4"], "#7F2704", "-"),
        ("XGB V4", all_data["v4"]["xgb"], all_data["v4"], "#2CA02C", "-"),
    ]
    for label, model, vdata, color, ls in pairs:
        PrecisionRecallDisplay.from_estimator(
            model, vdata["X_test"], vdata["y_test"],
            ax=ax, name=label, linestyle=ls, color=color,
        )

    prevalence = all_data["v4"]["y_test"].mean()
    ax.axhline(y=prevalence, color="k", linestyle=":", alpha=0.4,
               label=f"Prevalência ({prevalence:.3f})")
    ax.set_title("Curva Precision-Recall — V1 vs V2 vs V3 vs V4 — Classe 'ruim'",
                 fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v1_v2_v3_v4_pr_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ v1_v2_v3_v4_pr_overlay.png")


# ── Summary printer ──────────────────────────────────────────────

def print_summary(df_metrics, df_deltas):
    print(f"\n{'='*80}")
    print("  COMPARAÇÃO V1 vs V2 vs V3 vs V4 — Resumo")
    print(f"{'='*80}\n")

    display_cols = ["model", "accuracy", "f1_ruim", "recall_ruim",
                    "precision_ruim", "roc_auc"]
    print(df_metrics[display_cols].to_string(index=False))

    print(f"\n{'─'*80}")
    print("  Deltas (V4 − V3) e (V4 − V1):\n")

    for algo in ["Decision Tree", "Random Forest"]:
        sub = df_deltas[df_deltas["algo"] == algo]
        print(f"  {algo}:")
        for _, row in sub.iterrows():
            s43 = "+" if row["delta_v4_v3_pp"] >= 0 else ""
            s41 = "+" if row["delta_v4_v1_pp"] >= 0 else ""
            print(f"    {row['metric']:20s}  V1={row['v1']:.4f}  V2={row['v2']:.4f}  "
                  f"V3={row['v3']:.4f}  V4={row['v4']:.4f}  "
                  f"ΔV4-V3={s43}{row['delta_v4_v3_pp']:.2f}pp  "
                  f"ΔV4-V1={s41}{row['delta_v4_v1_pp']:.2f}pp")
        print()

    sub_xgb = df_deltas[df_deltas["algo"] == "XGBoost"]
    print("  XGBoost (V4 only):")
    for _, row in sub_xgb.iterrows():
        print(f"    {row['metric']:20s}  V4={row['v4']:.4f}")
    print()


# ── Main ─────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})

    print("Loading all data and models (V1 + V2 + V3 + V4) …")
    all_data = load_all()
    v4 = all_data["v4"]
    df_labeled = all_data["df_labeled"]

    print(f"  V1: {len(all_data['v1']['feature_names'])} features, "
          f"test={len(all_data['v1']['X_test']):,}")
    print(f"  V2: {len(all_data['v2']['feature_names'])} features, "
          f"test={len(all_data['v2']['X_test']):,}")
    print(f"  V3: {len(all_data['v3']['feature_names'])} features, "
          f"test={len(all_data['v3']['X_test']):,}")
    print(f"  V4: {len(v4['feature_names'])} features, "
          f"test={len(v4['X_test']):,} (same features as V3)\n")

    # ── V4-specific evaluation ───
    print("Generating V4 evaluation plots …")
    plot_confusion_matrices_v4(v4["dt"], v4["rf"], v4["xgb"],
                                v4["X_test"], v4["y_test"])
    plot_feature_importance_xgb_v4(v4["xgb"], v4["feature_names"])
    plot_feature_importance_3way_v4(v4["dt"], v4["rf"], v4["xgb"],
                                    v4["feature_names"])
    plot_roc_v4(v4["dt"], v4["rf"], v4["xgb"], v4["X_test"], v4["y_test"])
    plot_pr_v4(v4["dt"], v4["rf"], v4["xgb"], v4["X_test"], v4["y_test"])

    print("\nThreshold analysis …")
    plot_threshold_analysis_v4(v4["dt"], v4["rf"], v4["xgb"],
                                v4["X_test"], v4["y_test"], v4["thresholds"])

    print("\nV4 error analysis (XGBoost) …")
    fp, fn = error_analysis_v4(
        v4["xgb"], "XGBoost V4", v4["X_test"], v4["y_test"], v4["idx_test"],
        df_labeled, v4["feature_names"], n=10,
    )
    pd.DataFrame(fp).to_csv(OUTPUT_DIR / "error_analysis_fp_v4.csv", index=False)
    pd.DataFrame(fn).to_csv(OUTPUT_DIR / "error_analysis_fn_v4.csv", index=False)
    print("  ✓ error_analysis_fp_v4.csv")
    print("  ✓ error_analysis_fn_v4.csv")

    # ── V1 vs V2 vs V3 vs V4 comparison ───
    print("\nGenerating V1 vs V2 vs V3 vs V4 comparison …")
    df_metrics, df_deltas = build_comparison_table(all_data)
    plot_v1_v2_v3_v4_comparison(all_data)
    plot_v1_v2_v3_v4_roc_overlay(all_data)
    plot_v1_v2_v3_v4_pr_overlay(all_data)

    print_summary(df_metrics, df_deltas)

    print(f"\n{'='*80}")
    print(f"  Todos os artefatos salvos em {OUTPUT_DIR}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    run()
