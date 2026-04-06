"""
Evaluate trained models: confusion matrices, feature importance, ROC/PR
curves, decision-tree rule translation, and qualitative error analysis.

Reads:
  data/models/decision_tree.joblib
  data/models/random_forest.joblib
  data/models/feature_names.json
  data/features/features.csv
  data/labeled/moves_labeled.csv

Output (all saved to data/evaluation/):
  confusion_matrix_dt.png
  confusion_matrix_rf.png
  feature_importance_dt.png
  feature_importance_rf.png
  feature_importance_comparison.png
  model_comparison.png
  roc_curves.png
  precision_recall_curves.png
  learning_curves.png
  decision_tree_rules.txt
  decision_tree_rules_chess.txt
  error_analysis_fp.csv
  error_analysis_fn.csv
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
    roc_auc_score,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import export_text

MODELS_DIR = Path("data/models")
FEATURES_CSV = Path("data/features/features.csv")
LABELED_CSV = Path("data/labeled/moves_labeled.csv")
OUTPUT_DIR = Path("data/evaluation")

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
}

PIECE_NAMES = {1: "Peão", 2: "Cavalo", 3: "Bispo", 4: "Torre", 5: "Dama", 6: "Rei"}


# ── Data loading ──────────────────────────────────────────────────

def load_data():
    """Load features + labels, reproduce the same train/test split."""
    df_feat = pd.read_csv(FEATURES_CSV)
    df_labeled = pd.read_csv(LABELED_CSV)

    X = df_feat.drop(columns=["label"])
    y = (df_feat["label"] == "ruim").astype(int)

    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X, y, df_feat.index,
        test_size=0.15, stratify=y, random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=0.176, stratify=y_temp, random_state=RANDOM_STATE,
    )

    feature_names = list(X.columns)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "idx_test": idx_test,
        "feature_names": feature_names,
        "df_labeled": df_labeled,
        "X_full": X, "y_full": y,
    }


def load_models():
    dt = joblib.load(MODELS_DIR / "decision_tree.joblib")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    return dt, rf


# ── Confusion matrices ───────────────────────────────────────────

def plot_confusion_matrices(dt, rf, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model, name in [
        (axes[0], dt, "Decision Tree"),
        (axes[1], rf, "Random Forest"),
    ]:
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=["bom", "ruim"],
            cmap="Blues", ax=ax,
            values_format="d",
        )
        ax.set_title(f"Matriz de Confusão — {name}", fontsize=13)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ confusion_matrices.png")


# ── Feature importance ───────────────────────────────────────────

def plot_feature_importance(model, feature_names, title, filename, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    labels = [FEATURE_TRANSLATION.get(feature_names[i], feature_names[i])
              for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), values, color="#4C72B0")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini)")
    ax.set_title(title, fontsize=14)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_feature_importance_comparison(dt, rf, feature_names, top_n=15):
    """Side-by-side comparison of feature importances."""
    imp_dt = dt.feature_importances_
    imp_rf = rf.feature_importances_

    all_indices = set(np.argsort(imp_dt)[::-1][:top_n]) | set(np.argsort(imp_rf)[::-1][:top_n])
    combined = sorted(all_indices, key=lambda i: imp_rf[i] + imp_dt[i], reverse=True)[:top_n]

    labels = [FEATURE_TRANSLATION.get(feature_names[i], feature_names[i])
              for i in combined]
    vals_dt = [imp_dt[i] for i in combined]
    vals_rf = [imp_rf[i] for i in combined]

    y_pos = np.arange(top_n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y_pos - height / 2, vals_dt, height, label="Decision Tree", color="#4C72B0")
    ax.barh(y_pos + height / 2, vals_rf, height, label="Random Forest", color="#DD8452")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini)")
    ax.set_title("Comparação de Feature Importance — DT vs RF", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ feature_importance_comparison.png")


# ── Model comparison bar chart ───────────────────────────────────

def plot_model_comparison(dt, rf, X_test, y_test):
    metrics = {}
    for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 (bom)": f1_score(y_test, y_pred, pos_label=0),
            "F1 (ruim)": f1_score(y_test, y_pred, pos_label=1),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }

    metric_names = list(next(iter(metrics.values())).keys())
    x = np.arange(len(metric_names))
    width = 0.30

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, vals) in enumerate(metrics.items()):
        values = [vals[m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=model_name)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos — Conjunto de Teste", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ model_comparison.png")


# ── ROC curves ───────────────────────────────────────────────────

def plot_roc_curves(dt, rf, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(dt, X_test, y_test, ax=ax, name="Decision Tree")
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random baseline")
    ax.set_title("Curva ROC — Classe 'ruim'", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ roc_curves.png")


# ── Precision-Recall curves ──────────────────────────────────────

def plot_pr_curves(dt, rf, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(dt, X_test, y_test, ax=ax, name="Decision Tree")
    PrecisionRecallDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")
    prevalence = y_test.mean()
    ax.axhline(y=prevalence, color="k", linestyle="--", alpha=0.5, label=f"Prevalência ({prevalence:.3f})")
    ax.set_title("Curva Precision-Recall — Classe 'ruim'", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ precision_recall_curves.png")


# ── Learning curves ──────────────────────────────────────────────

def plot_learning_curves(dt, rf, X_train, y_train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train_sizes_frac = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    for ax, model, name in [
        (axes[0], dt, "Decision Tree"),
        (axes[1], rf, "Random Forest"),
    ]:
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5,
            train_sizes=train_sizes_frac,
            scoring="f1", n_jobs=-1,
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#4C72B0")
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="#DD8452")
        ax.plot(sizes, train_mean, "o-", color="#4C72B0", label="Treino")
        ax.plot(sizes, val_mean, "o-", color="#DD8452", label="Validação (CV)")

        ax.set_xlabel("Tamanho do treino")
        ax.set_ylabel("F1-score")
        ax.set_title(f"Curva de Aprendizado — {name}", fontsize=13)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ learning_curves.png")


# ── Decision tree rules ──────────────────────────────────────────

def extract_and_translate_rules(dt, feature_names):
    rules_raw = export_text(dt, feature_names=feature_names, max_depth=5)
    with open(OUTPUT_DIR / "decision_tree_rules.txt", "w") as f:
        f.write(rules_raw)
    print("  ✓ decision_tree_rules.txt")

    translated = translate_rules(rules_raw)
    with open(OUTPUT_DIR / "decision_tree_rules_chess.txt", "w") as f:
        f.write(translated)
    print("  ✓ decision_tree_rules_chess.txt")

    return rules_raw


def translate_rules(rules_text: str) -> str:
    """Translate sklearn rule text into chess-friendly language."""
    lines = rules_text.split("\n")
    translated_lines = []

    for line in lines:
        tl = line
        for eng, pt in FEATURE_TRANSLATION.items():
            tl = tl.replace(eng, pt)

        tl = re.sub(r"class: 0", "→ BOM", tl)
        tl = re.sub(r"class: 1", "→ RUIM", tl)

        translated_lines.append(tl)

    header = textwrap.dedent("""\
    ═══════════════════════════════════════════════════════════
    Regras da Árvore de Decisão — Traduzidas para xadrez
    ═══════════════════════════════════════════════════════════

    Legenda:
      - "Diferença material <= -3.0": jogador com ≥3 pontos a menos
      - "Lances legais (jogador)": mobilidade/atividade das peças
      - "É captura": o lance captura uma peça adversária
      - "Número do lance": fase da partida (abertura/meio-jogo/final)
      - "Peça movida": 1=Peão, 2=Cavalo, 3=Bispo, 4=Torre, 5=Dama, 6=Rei
      - "Escudo de peões do rei": peões protegendo o rei (0–3)

    ═══════════════════════════════════════════════════════════

    """)

    return header + "\n".join(translated_lines)


# ── Qualitative error analysis ───────────────────────────────────

def error_analysis(model, model_name, X_test, y_test, idx_test, df_labeled, feature_names, n=10):
    """Find FP and FN examples and annotate with game context."""
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
            labeled_row = df_labeled.iloc[idx]
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
                "game_site": labeled_row.get("game_site", ""),
                "move_number": labeled_row.get("move_number", ""),
                "color": labeled_row.get("color", ""),
                "fen_before": labeled_row.get("fen_before", ""),
                "move_san": labeled_row.get("move_san", ""),
                "move_uci": labeled_row.get("move_uci", ""),
                "delta_cp": labeled_row.get("delta_cp", ""),
                "label_real": labeled_row.get("label", ""),
                "label_pred": "ruim" if error_type == "FP" else "bom",
                "top_features": top_features,
            })
        return rows

    fp_rows = build_rows(fp_sample, "FP")
    fn_rows = build_rows(fn_sample, "FN")

    return fp_rows, fn_rows


def print_error_examples(fp_rows, fn_rows, model_name):
    """Print readable error analysis to stdout."""
    print(f"\n{'='*70}")
    print(f"  Análise qualitativa de erros — {model_name}")
    print(f"{'='*70}")

    print(f"\n  FALSOS POSITIVOS (modelo disse 'ruim', mas o lance é bom):")
    print(f"  {'—'*60}")
    for i, row in enumerate(fp_rows[:5], 1):
        print(f"\n  FP-{i}:")
        print(f"    Partida : {row['game_site']}")
        print(f"    Lance   : {row['move_san']} (#{row['move_number']}, {row['color']})")
        print(f"    FEN     : {row['fen_before']}")
        print(f"    Delta   : {row['delta_cp']} cp")
        print(f"    Features: {row['top_features']}")

    print(f"\n  FALSOS NEGATIVOS (modelo disse 'bom', mas o lance é ruim):")
    print(f"  {'—'*60}")
    for i, row in enumerate(fn_rows[:5], 1):
        print(f"\n  FN-{i}:")
        print(f"    Partida : {row['game_site']}")
        print(f"    Lance   : {row['move_san']} (#{row['move_number']}, {row['color']})")
        print(f"    FEN     : {row['fen_before']}")
        print(f"    Delta   : {row['delta_cp']} cp")
        print(f"    Features: {row['top_features']}")


# ── Summary table ────────────────────────────────────────────────

def print_comparison_table(dt, rf, X_test, y_test):
    print(f"\n{'='*70}")
    print("  Tabela comparativa — Conjunto de teste")
    print(f"{'='*70}\n")

    rows = []
    for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Modelo": name,
            "Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
            "F1 (bom)": f"{f1_score(y_test, y_pred, pos_label=0):.4f}",
            "F1 (ruim)": f"{f1_score(y_test, y_pred, pos_label=1):.4f}",
            "Recall (ruim)": f"{(y_pred[y_test == 1] == 1).mean():.4f}",
            "ROC-AUC": f"{roc_auc_score(y_test, y_proba):.4f}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
        y_pred = model.predict(X_test)
        print(f"\n  {name} — Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["bom", "ruim"]))


# ── Main ─────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})

    print("Loading data and models …")
    data = load_data()
    dt, rf = load_models()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    idx_test = data["idx_test"]
    feature_names = data["feature_names"]
    df_labeled = data["df_labeled"]

    print(f"  Test set: {len(X_test):,} samples ({(y_test == 1).sum():,} ruim)\n")

    print("Generating plots …")
    plot_confusion_matrices(dt, rf, X_test, y_test)
    plot_feature_importance(dt, feature_names,
                           "Top 15 Features — Decision Tree",
                           "feature_importance_dt.png")
    plot_feature_importance(rf, feature_names,
                           "Top 15 Features — Random Forest",
                           "feature_importance_rf.png")
    plot_feature_importance_comparison(dt, rf, feature_names)
    plot_model_comparison(dt, rf, X_test, y_test)
    plot_roc_curves(dt, rf, X_test, y_test)
    plot_pr_curves(dt, rf, X_test, y_test)

    print("\nExtracting decision tree rules …")
    extract_and_translate_rules(dt, feature_names)

    print("\nComputing learning curves (may take a few minutes) …")
    plot_learning_curves(dt, rf, X_train, y_train)

    print("\nRunning error analysis …")
    all_fp, all_fn = [], []
    for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
        fp, fn = error_analysis(model, name, X_test, y_test, idx_test,
                                df_labeled, feature_names, n=10)
        print_error_examples(fp, fn, name)
        all_fp.extend(fp)
        all_fn.extend(fn)

    pd.DataFrame(all_fp).to_csv(OUTPUT_DIR / "error_analysis_fp.csv", index=False)
    pd.DataFrame(all_fn).to_csv(OUTPUT_DIR / "error_analysis_fn.csv", index=False)
    print("\n  ✓ error_analysis_fp.csv")
    print("  ✓ error_analysis_fn.csv")

    print_comparison_table(dt, rf, X_test, y_test)

    print(f"\n{'='*70}")
    print(f"  Todos os artefatos salvos em {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    run()
