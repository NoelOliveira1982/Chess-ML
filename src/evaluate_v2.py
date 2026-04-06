"""
Evaluate V2 models and generate V1-vs-V2 comparison artifacts.

Reads:
  data/models/decision_tree.joblib        (V1)
  data/models/random_forest.joblib        (V1)
  data/models_v2/decision_tree.joblib     (V2)
  data/models_v2/random_forest.joblib     (V2)
  data/features/features.csv              (V1)
  data/features/features_v2.csv           (V2)
  data/models/results.csv                 (V1 metrics)
  data/models_v2/results.csv              (V2 metrics)
  data/labeled/moves_labeled.csv

Output (all saved to data/evaluation_v2/):
  confusion_matrices_v2.png
  feature_importance_dt_v2.png
  feature_importance_rf_v2.png
  feature_importance_comparison_v2.png
  roc_curves_v2.png
  precision_recall_curves_v2.png
  learning_curves_v2.png
  decision_tree_rules_v2.txt
  decision_tree_rules_chess_v2.txt
  error_analysis_fp_v2.csv
  error_analysis_fn_v2.csv
  v1_vs_v2_metrics.csv
  v1_vs_v2_comparison.png
  v1_vs_v2_roc_overlay.png
  v1_vs_v2_pr_overlay.png
  v1_vs_v2_feature_importance_new.png
  tactical_features_analysis.csv
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import export_text

MODELS_DIR_V1 = Path("data/models")
MODELS_DIR_V2 = Path("data/models_v2")
FEATURES_V1 = Path("data/features/features.csv")
FEATURES_V2 = Path("data/features/features_v2.csv")
LABELED_CSV = Path("data/labeled/moves_labeled.csv")
OUTPUT_DIR = Path("data/evaluation_v2")

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


def _translate(name: str) -> str:
    return FEATURE_TRANSLATION.get(name, name)


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
    df_labeled = pd.read_csv(LABELED_CSV)

    dt_v1 = joblib.load(MODELS_DIR_V1 / "decision_tree.joblib")
    rf_v1 = joblib.load(MODELS_DIR_V1 / "random_forest.joblib")
    dt_v2 = joblib.load(MODELS_DIR_V2 / "decision_tree.joblib")
    rf_v2 = joblib.load(MODELS_DIR_V2 / "random_forest.joblib")

    data_v1 = _split(df_v1)
    data_v2 = _split(df_v2)

    return {
        "v1": {"dt": dt_v1, "rf": rf_v1, **data_v1},
        "v2": {"dt": dt_v2, "rf": rf_v2, **data_v2},
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


# ── V2 confusion matrices ────────────────────────────────────────

def plot_confusion_matrices_v2(dt, rf, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, name in [
        (axes[0], dt, "Decision Tree V2"),
        (axes[1], rf, "Random Forest V2"),
    ]:
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=["bom", "ruim"],
            cmap="Blues", ax=ax, values_format="d",
        )
        ax.set_title(f"Matriz de Confusão — {name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ confusion_matrices_v2.png")


# ── V2 feature importance ────────────────────────────────────────

def plot_feature_importance_v2(model, feature_names, title, filename, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    labels = [_translate(feature_names[i]) for i in indices]
    values = importances[indices]
    is_tactical = [feature_names[i] in TACTICAL_FEATURES for i in indices]
    colors = ["#C44E52" if t else "#4C72B0" for t in is_tactical]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini)")
    ax.set_title(title, fontsize=14)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4C72B0", label="Feature V1 (posicional)"),
        Patch(facecolor="#C44E52", label="Feature V2 (tática)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_feature_importance_comparison_v2(dt, rf, feature_names, top_n=15):
    imp_dt = dt.feature_importances_
    imp_rf = rf.feature_importances_
    all_idx = set(np.argsort(imp_dt)[::-1][:top_n]) | set(np.argsort(imp_rf)[::-1][:top_n])
    combined = sorted(all_idx, key=lambda i: imp_rf[i] + imp_dt[i], reverse=True)[:top_n]

    labels = [_translate(feature_names[i]) for i in combined]
    vals_dt = [imp_dt[i] for i in combined]
    vals_rf = [imp_rf[i] for i in combined]

    y_pos = np.arange(top_n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y_pos - height / 2, vals_dt, height, label="Decision Tree V2", color="#4C72B0")
    ax.barh(y_pos + height / 2, vals_rf, height, label="Random Forest V2", color="#DD8452")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini)")
    ax.set_title("Comparação de Feature Importance — DT V2 vs RF V2", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_comparison_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ feature_importance_comparison_v2.png")


# ── V2 ROC & PR curves ───────────────────────────────────────────

def plot_roc_v2(dt, rf, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(dt, X_test, y_test, ax=ax, name="Decision Tree V2")
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest V2")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random baseline")
    ax.set_title("Curva ROC — V2 — Classe 'ruim'", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ roc_curves_v2.png")


def plot_pr_v2(dt, rf, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(dt, X_test, y_test, ax=ax, name="Decision Tree V2")
    PrecisionRecallDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest V2")
    prevalence = y_test.mean()
    ax.axhline(y=prevalence, color="k", linestyle="--", alpha=0.5,
               label=f"Prevalência ({prevalence:.3f})")
    ax.set_title("Curva Precision-Recall — V2 — Classe 'ruim'", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curves_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ precision_recall_curves_v2.png")


# ── V2 learning curves ───────────────────────────────────────────

def plot_learning_curves_v2(dt, rf, X_train, y_train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    for ax, model, name in [
        (axes[0], dt, "Decision Tree V2"),
        (axes[1], rf, "Random Forest V2"),
    ]:
        sizes, train_sc, val_sc = learning_curve(
            model, X_train, y_train, cv=5,
            train_sizes=fracs, scoring="f1", n_jobs=-1,
        )
        t_mean, t_std = train_sc.mean(axis=1), train_sc.std(axis=1)
        v_mean, v_std = val_sc.mean(axis=1), val_sc.std(axis=1)

        ax.fill_between(sizes, t_mean - t_std, t_mean + t_std, alpha=0.1, color="#4C72B0")
        ax.fill_between(sizes, v_mean - v_std, v_mean + v_std, alpha=0.1, color="#DD8452")
        ax.plot(sizes, t_mean, "o-", color="#4C72B0", label="Treino")
        ax.plot(sizes, v_mean, "o-", color="#DD8452", label="Validação (CV)")
        ax.set_xlabel("Tamanho do treino")
        ax.set_ylabel("F1-score")
        ax.set_title(f"Curva de Aprendizado — {name}", fontsize=13)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curves_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ learning_curves_v2.png")


# ── V2 DT rules ──────────────────────────────────────────────────

def extract_rules_v2(dt, feature_names):
    rules_raw = export_text(dt, feature_names=feature_names, max_depth=5)
    with open(OUTPUT_DIR / "decision_tree_rules_v2.txt", "w") as f:
        f.write(rules_raw)
    print("  ✓ decision_tree_rules_v2.txt")

    lines = rules_raw.split("\n")
    translated = []
    for line in lines:
        tl = line
        for eng, pt in FEATURE_TRANSLATION.items():
            tl = tl.replace(eng, pt)
        tl = re.sub(r"class: 0", "→ BOM", tl)
        tl = re.sub(r"class: 1", "→ RUIM", tl)
        translated.append(tl)

    header = textwrap.dedent("""\
    ═══════════════════════════════════════════════════════════
    Regras da Árvore de Decisão V2 — Traduzidas para xadrez
    ═══════════════════════════════════════════════════════════

    Legenda:
      - "Casas disputadas": nº de casas atacadas por ambos os lados
      - "Valor indefeso (adversário)": valor material de peças hanging
      - "Peças cravadas": peças que não podem mover sem expor peça mais valiosa
      - "Atacantes do rei": nº de peças inimigas atacando zona do rei
      - Demais features: ver legenda V1

    ═══════════════════════════════════════════════════════════

    """)
    with open(OUTPUT_DIR / "decision_tree_rules_chess_v2.txt", "w") as f:
        f.write(header + "\n".join(translated))
    print("  ✓ decision_tree_rules_chess_v2.txt")


# ── V2 error analysis ────────────────────────────────────────────

def error_analysis_v2(model, model_name, X_test, y_test, idx_test, df_labeled, feature_names, n=10):
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


# ── V1 vs V2 comparison ──────────────────────────────────────────

def build_comparison_table(all_data):
    """Build a metrics comparison table with deltas."""
    rows = []
    for version, vdata in [("V1", all_data["v1"]), ("V2", all_data["v2"])]:
        for name, key in [("Decision Tree", "dt"), ("Random Forest", "rf")]:
            m = compute_metrics(vdata[key], vdata["X_test"], vdata["y_test"])
            m["model"] = f"{name} {version}"
            m["version"] = version
            m["algo"] = name
            rows.append(m)

    df = pd.DataFrame(rows)

    delta_rows = []
    for algo in ["Decision Tree", "Random Forest"]:
        v1 = df[(df["algo"] == algo) & (df["version"] == "V1")].iloc[0]
        v2 = df[(df["algo"] == algo) & (df["version"] == "V2")].iloc[0]
        for metric in ["accuracy", "f1_bom", "f1_ruim", "recall_ruim", "precision_ruim", "roc_auc"]:
            delta_rows.append({
                "algo": algo,
                "metric": metric,
                "v1": round(v1[metric], 4),
                "v2": round(v2[metric], 4),
                "delta": round(v2[metric] - v1[metric], 4),
                "delta_pp": round((v2[metric] - v1[metric]) * 100, 2),
            })

    df_delta = pd.DataFrame(delta_rows)
    df.to_csv(OUTPUT_DIR / "v1_vs_v2_metrics.csv", index=False)
    df_delta.to_csv(OUTPUT_DIR / "v1_vs_v2_deltas.csv", index=False)
    print("  ✓ v1_vs_v2_metrics.csv")
    print("  ✓ v1_vs_v2_deltas.csv")
    return df, df_delta


def plot_v1_vs_v2_comparison(all_data):
    """4-bar grouped chart: DT V1, DT V2, RF V1, RF V2."""
    metric_keys = ["accuracy", "f1_ruim", "recall_ruim", "precision_ruim", "roc_auc"]
    metric_labels = ["Accuracy", "F1 (ruim)", "Recall (ruim)", "Precision (ruim)", "ROC-AUC"]

    models = [
        ("DT V1", all_data["v1"]["dt"], all_data["v1"]),
        ("DT V2", all_data["v2"]["dt"], all_data["v2"]),
        ("RF V1", all_data["v1"]["rf"], all_data["v1"]),
        ("RF V2", all_data["v2"]["rf"], all_data["v2"]),
    ]

    colors = ["#8DA0CB", "#4C72B0", "#F4A582", "#DD8452"]

    all_vals = {}
    for label, model, vdata in models:
        m = compute_metrics(model, vdata["X_test"], vdata["y_test"])
        all_vals[label] = [m[k] for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (label, _, _) in enumerate(models):
        vals = all_vals[label]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Comparação V1 vs V2 — Conjunto de Teste", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v1_vs_v2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ v1_vs_v2_comparison.png")


def plot_v1_vs_v2_roc_overlay(all_data):
    """Overlay ROC curves: RF V1 vs RF V2 (+ DT V1 vs DT V2)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    pairs = [
        ("DT V1", all_data["v1"]["dt"], all_data["v1"], "#8DA0CB", "--"),
        ("DT V2", all_data["v2"]["dt"], all_data["v2"], "#4C72B0", "-"),
        ("RF V1", all_data["v1"]["rf"], all_data["v1"], "#F4A582", "--"),
        ("RF V2", all_data["v2"]["rf"], all_data["v2"], "#DD8452", "-"),
    ]
    for label, model, vdata, color, ls in pairs:
        RocCurveDisplay.from_estimator(
            model, vdata["X_test"], vdata["y_test"],
            ax=ax, name=label, linestyle=ls, color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("Curva ROC — V1 vs V2 — Classe 'ruim'", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v1_vs_v2_roc_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ v1_vs_v2_roc_overlay.png")


def plot_v1_vs_v2_pr_overlay(all_data):
    """Overlay PR curves: V1 vs V2."""
    fig, ax = plt.subplots(figsize=(8, 7))

    pairs = [
        ("DT V1", all_data["v1"]["dt"], all_data["v1"], "#8DA0CB", "--"),
        ("DT V2", all_data["v2"]["dt"], all_data["v2"], "#4C72B0", "-"),
        ("RF V1", all_data["v1"]["rf"], all_data["v1"], "#F4A582", "--"),
        ("RF V2", all_data["v2"]["rf"], all_data["v2"], "#DD8452", "-"),
    ]
    for label, model, vdata, color, ls in pairs:
        PrecisionRecallDisplay.from_estimator(
            model, vdata["X_test"], vdata["y_test"],
            ax=ax, name=label, linestyle=ls, color=color,
        )

    prevalence = all_data["v2"]["y_test"].mean()
    ax.axhline(y=prevalence, color="k", linestyle=":", alpha=0.4,
               label=f"Prevalência ({prevalence:.3f})")
    ax.set_title("Curva Precision-Recall — V1 vs V2 — Classe 'ruim'", fontsize=14)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v1_vs_v2_pr_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ v1_vs_v2_pr_overlay.png")


def plot_new_features_importance(rf_v2, feature_names_v2):
    """Show importance of the 19 new tactical features in RF V2."""
    imp = rf_v2.feature_importances_
    tactical_idx = [i for i, f in enumerate(feature_names_v2) if f in TACTICAL_FEATURES]
    tactical_imp = [(feature_names_v2[i], imp[i]) for i in tactical_idx]
    tactical_imp.sort(key=lambda x: x[1], reverse=True)

    labels = [_translate(f) for f, _ in tactical_imp]
    values = [v for _, v in tactical_imp]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(labels)), values, color="#C44E52")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini) no Random Forest V2")
    ax.set_title("Contribuição das 19 Features Táticas (V2)", fontsize=14)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v1_vs_v2_feature_importance_new.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ v1_vs_v2_feature_importance_new.png")

    df_tact = pd.DataFrame(tactical_imp, columns=["feature", "importance_rf_v2"])
    df_tact["feature_pt"] = df_tact["feature"].map(_translate)
    df_tact.to_csv(OUTPUT_DIR / "tactical_features_analysis.csv", index=False)
    print("  ✓ tactical_features_analysis.csv")


# ── Summary printer ──────────────────────────────────────────────

def print_summary(df_metrics, df_deltas):
    print(f"\n{'='*75}")
    print("  COMPARAÇÃO V1 vs V2 — Resumo")
    print(f"{'='*75}\n")

    print(df_metrics[["model", "accuracy", "f1_ruim", "recall_ruim",
                       "precision_ruim", "roc_auc"]].to_string(index=False))

    print(f"\n{'─'*75}")
    print("  Deltas (V2 − V1):\n")

    for algo in ["Decision Tree", "Random Forest"]:
        sub = df_deltas[df_deltas["algo"] == algo]
        print(f"  {algo}:")
        for _, row in sub.iterrows():
            sign = "+" if row["delta_pp"] >= 0 else ""
            print(f"    {row['metric']:20s}  V1={row['v1']:.4f}  V2={row['v2']:.4f}  "
                  f"Δ={sign}{row['delta_pp']:.2f}pp")
        print()


# ── Main ─────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})

    print("Loading all data and models (V1 + V2) …")
    all_data = load_all()
    v2 = all_data["v2"]
    df_labeled = all_data["df_labeled"]

    print(f"  V1: {len(all_data['v1']['feature_names'])} features, "
          f"test={len(all_data['v1']['X_test']):,}")
    print(f"  V2: {len(v2['feature_names'])} features, "
          f"test={len(v2['X_test']):,}\n")

    # ── V2-specific evaluation ───
    print("Generating V2 evaluation plots …")
    plot_confusion_matrices_v2(v2["dt"], v2["rf"], v2["X_test"], v2["y_test"])
    plot_feature_importance_v2(v2["dt"], v2["feature_names"],
                               "Top 15 Features — Decision Tree V2",
                               "feature_importance_dt_v2.png")
    plot_feature_importance_v2(v2["rf"], v2["feature_names"],
                               "Top 15 Features — Random Forest V2",
                               "feature_importance_rf_v2.png")
    plot_feature_importance_comparison_v2(v2["dt"], v2["rf"], v2["feature_names"])
    plot_roc_v2(v2["dt"], v2["rf"], v2["X_test"], v2["y_test"])
    plot_pr_v2(v2["dt"], v2["rf"], v2["X_test"], v2["y_test"])

    print("\nExtracting V2 decision tree rules …")
    extract_rules_v2(v2["dt"], v2["feature_names"])

    print("\nComputing V2 learning curves …")
    plot_learning_curves_v2(v2["dt"], v2["rf"], v2["X_train"], v2["y_train"])

    print("\nV2 error analysis …")
    all_fp, all_fn = [], []
    for name, key in [("Decision Tree V2", "dt"), ("Random Forest V2", "rf")]:
        fp, fn = error_analysis_v2(
            v2[key], name, v2["X_test"], v2["y_test"], v2["idx_test"],
            df_labeled, v2["feature_names"], n=10,
        )
        all_fp.extend(fp)
        all_fn.extend(fn)
    pd.DataFrame(all_fp).to_csv(OUTPUT_DIR / "error_analysis_fp_v2.csv", index=False)
    pd.DataFrame(all_fn).to_csv(OUTPUT_DIR / "error_analysis_fn_v2.csv", index=False)
    print("  ✓ error_analysis_fp_v2.csv")
    print("  ✓ error_analysis_fn_v2.csv")

    # ── V1 vs V2 comparison ───
    print("\nGenerating V1 vs V2 comparison …")
    df_metrics, df_deltas = build_comparison_table(all_data)
    plot_v1_vs_v2_comparison(all_data)
    plot_v1_vs_v2_roc_overlay(all_data)
    plot_v1_vs_v2_pr_overlay(all_data)
    plot_new_features_importance(v2["rf"], v2["feature_names"])

    print_summary(df_metrics, df_deltas)

    print(f"\n{'='*75}")
    print(f"  Todos os artefatos salvos em {OUTPUT_DIR}/")
    print(f"{'='*75}")


if __name__ == "__main__":
    run()
