"""
Shared plotting and analysis utilities for chess move classifier notebooks.

All functions receive data/config as arguments — no global state.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pointbiserialr
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve
from sklearn.tree import plot_tree

RANDOM_SEED = 42

FEATURE_TRANSLATION = {
    # V1
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
    "material_diff": "Dif. material",
    "legal_moves_player": "Legais (jogador)",
    "legal_moves_opponent": "Legais (adversário)",
    "mobility_diff": "Dif. mobilidade",
    "player_castled": "Rocou",
    "opponent_castled": "Adv. rocou",
    "player_can_castle": "Pode rocar",
    "king_pawn_shield": "Escudo rei",
    "player_doubled_pawns": "Peões dobrados",
    "player_isolated_pawns": "Peões isolados",
    "player_passed_pawns": "Passados (jog.)",
    "opponent_passed_pawns": "Passados (adv.)",
    "player_center_control": "Centro (jog.)",
    "opponent_center_control": "Centro (adv.)",
    "player_center_occupation": "Ocupação centro",
    "is_capture": "É captura",
    "is_check": "Dá xeque",
    "is_promotion": "É promoção",
    "moved_piece": "Peça movida",
    "move_to_center": "Move p/ centro",
    "move_to_extended_center": "Move p/ centro exp.",
    "move_number": "Nº do lance",
    "is_white": "Joga de brancas",
    # V2
    "hanging_pieces_player": "Indefesas (jog.)",
    "hanging_pieces_opponent": "Indefesas (adv.)",
    "hanging_value_player": "Val. indefeso (jog.)",
    "hanging_value_opponent": "Val. indefeso (adv.)",
    "min_attacker_vs_piece_player": "Menor atac. vs peça",
    "threats_against_player": "Ameaças (jog.)",
    "threats_against_opponent": "Ameaças (adv.)",
    "max_threat_value_player": "Maior ameaça (jog.)",
    "max_threat_value_opponent": "Maior ameaça (adv.)",
    "pinned_pieces_player": "Cravadas (jog.)",
    "pinned_pieces_opponent": "Cravadas (adv.)",
    "king_attackers_player": "Atacantes rei (jog.)",
    "king_attackers_opponent": "Atacantes rei (adv.)",
    "king_open_files_player": "Col. abertas rei",
    "king_escape_squares_player": "Fuga do rei",
    "total_attacks_player": "Ataques totais (jog.)",
    "total_attacks_opponent": "Ataques totais (adv.)",
    "contested_squares": "Casas disputadas",
    "undefended_pieces_player": "Sem defesa (jog.)",
    # V3
    "delta_hanging_player": "Δ indefesas (jog.)",
    "delta_hanging_opponent": "Δ indefesas (adv.)",
    "delta_hanging_value_player": "Δ val. indefeso (jog.)",
    "delta_threats_against_player": "Δ ameaças (jog.)",
    "delta_mobility_player": "Δ mobilidade (jog.)",
    "delta_mobility_opponent": "Δ mobilidade (adv.)",
    "delta_contested_squares": "Δ casas disputadas",
    "delta_king_attackers_player": "Δ atacantes rei (jog.)",
    "opponent_best_capture_value": "Melhor captura adv.",
    "opponent_can_check": "Adv. pode dar xeque",
    "opponent_num_good_captures": "Nº capturas boas adv.",
    "created_hanging_self": "Criou peça indefesa",
    "see_of_move": "SEE do lance",
    "worst_see_against_player": "Pior SEE contra jog.",
    "is_losing_capture": "É captura perdedora",
}


def translate(name: str) -> str:
    """Translate a feature name to Portuguese."""
    return FEATURE_TRANSLATION.get(name, name)


# ── Data overview ────────────────────────────────────────────────


def print_dataset_stats(df_filtered: pd.DataFrame) -> None:
    n_games = df_filtered["game_site"].nunique()
    n_moves = len(df_filtered)
    avg_moves = n_moves / n_games
    print(f"{'='*50}")
    print(f"  DATASET FILTRADO")
    print(f"{'='*50}")
    print(f"  Partidas       : {n_games:,}")
    print(f"  Lances (total) : {n_moves:,}")
    print(f"  Lances/partida : {avg_moves:.1f}")
    print(f"  Colunas        : {list(df_filtered.columns)}")


def plot_rating_distribution(df_filtered: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col, title, color in [
        (axes[0], "white_elo", "Rating das Brancas", "#4C72B0"),
        (axes[1], "black_elo", "Rating das Pretas", "#DD8452"),
    ]:
        elos = df_filtered.groupby("game_site")[col].first()
        ax.hist(elos, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Elo")
        ax.set_ylabel("Partidas")
        ax.set_title(title)
        ax.axvline(elos.mean(), color="red", linestyle="--",
                    label=f"Média: {elos.mean():.0f}")
        ax.legend()
    plt.suptitle("Distribuição de Ratings no Dataset Filtrado",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def print_labeling_stats(df_scored: pd.DataFrame,
                         df_labeled: pd.DataFrame) -> None:
    n_bom = (df_labeled["label"] == "bom").sum()
    n_ruim = (df_labeled["label"] == "ruim").sum()
    n_desc = len(df_scored) - len(df_labeled)
    print(f"{'='*50}")
    print(f"  ROTULAGEM")
    print(f"{'='*50}")
    print(f"  Lances avaliados      : {len(df_scored):,}")
    print(f"  Bom (δ ≥ −50 cp)      : {n_bom:,} ({n_bom/len(df_scored)*100:.1f}%)")
    print(f"  Descartado (cinzenta) : {n_desc:,} ({n_desc/len(df_scored)*100:.1f}%)")
    print(f"  Ruim (δ ≤ −150 cp)    : {n_ruim:,} ({n_ruim/len(df_scored)*100:.1f}%)")
    print(f"  ──────────────────────")
    print(f"  Dataset final         : {len(df_labeled):,} (bom + ruim)")
    print(f"  Ratio bom:ruim        : {n_bom/n_ruim:.2f}:1")


def plot_labeling_analysis(df_scored: pd.DataFrame,
                           df_labeled: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    deltas = df_scored["delta_cp"].clip(-2000, 2000)
    ax.hist(deltas, bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(-50, color="green", linestyle="--", linewidth=1.5,
               label="Limiar bom (−50)")
    ax.axvline(-150, color="red", linestyle="--", linewidth=1.5,
               label="Limiar ruim (−150)")
    ax.set_xlabel("Delta (centipawns)")
    ax.set_ylabel("Lances")
    ax.set_title("Distribuição de Delta CP")
    ax.legend(fontsize=9)

    ax = axes[1]
    counts = df_scored["label"].value_counts()
    colors_map = {"bom": "#55A868", "ruim": "#C44E52", "descartado": "#8C8C8C"}
    order = ["bom", "descartado", "ruim"]
    bars = ax.bar(order, [counts.get(c, 0) for c in order],
                  color=[colors_map.get(c, "gray") for c in order],
                  edgecolor="white")
    for bar, c in zip(bars, order):
        val = counts.get(c, 0)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 500,
                f"{val:,}\n({val/len(df_scored)*100:.1f}%)",
                ha="center", fontsize=10)
    ax.set_ylabel("Lances")
    ax.set_title("Distribuição por Classe")

    ax = axes[2]
    bom_delta = df_labeled[df_labeled["label"] == "bom"]["delta_cp"].clip(-3000, 3000)
    ruim_delta = df_labeled[df_labeled["label"] == "ruim"]["delta_cp"].clip(-3000, 3000)
    bp = ax.boxplot([bom_delta, ruim_delta], labels=["Bom", "Ruim"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#55A868")
    bp["boxes"][1].set_facecolor("#C44E52")
    ax.set_ylabel("Delta (centipawns)")
    ax.set_title("Delta CP por Classe")

    plt.suptitle("Análise da Rotulagem", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ── Features ─────────────────────────────────────────────────────


def print_features_stats(df_features: pd.DataFrame, config) -> None:
    feature_cols = [c for c in df_features.columns if c != "label"]
    print(f"{'='*50}")
    print(f"  FEATURES — V{config.version}")
    print(f"{'='*50}")
    print(f"  Fonte       : {config.features_csv}")
    print(f"  Linhas      : {len(df_features):,}")
    print(f"  Features    : {len(feature_cols)}")
    print(f"  Valores nulos: {df_features[feature_cols].isnull().sum().sum()}")
    print(f"\nDistribuição do label:")
    print(df_features["label"].value_counts().to_string())


def plot_correlation_matrix(df_features: pd.DataFrame,
                            feature_cols: list[str]) -> None:
    corr = df_features[feature_cols].corr()
    labels_pt = [translate(c) for c in feature_cols]

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0,
        xticklabels=labels_pt, yticklabels=labels_pt,
        annot=False, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Correlação de Pearson"},
        ax=ax,
    )
    ax.set_title("Mapa de Correlação entre Features", fontsize=15, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()

    high_corr = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = abs(corr.iloc[i, j])
            if r > 0.7:
                high_corr.append((feature_cols[i], feature_cols[j],
                                  corr.iloc[i, j]))
    if high_corr:
        print("Pares com |correlação| > 0.7:")
        for a, b, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"  {translate(a):25s}  ↔  {translate(b):25s}  r = {r:+.3f}")
    else:
        print("Nenhum par de features com |correlação| > 0.7.")


# ── Model evaluation ─────────────────────────────────────────────


def print_split_info(X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    print(f"{'='*50}")
    print(f"  SPLIT DOS DADOS")
    print(f"{'='*50}")
    for name, Xs, ys in [("Treino", X_train, y_train),
                         ("Validação", X_val, y_val),
                         ("Teste", X_test, y_test)]:
        n = len(ys)
        n_ruim = int(ys.sum())
        print(f"  {name:10s}: {n:>6,} amostras  "
              f"(bom={n - n_ruim:,}, ruim={n_ruim:,}, "
              f"{n_ruim/n*100:.1f}% ruim)")


def print_model_params(dt, rf, config) -> None:
    print(f"\n{'='*55}")
    print(f"  DECISION TREE V{config.version} — Melhores hiperparâmetros")
    print(f"{'='*55}")
    print(f"  criterion        : {dt.criterion}")
    print(f"  max_depth        : {dt.max_depth}")
    print(f"  min_samples_leaf : {dt.min_samples_leaf}")
    print(f"  class_weight     : {dt.class_weight}")
    print(f"\n{'='*55}")
    print(f"  RANDOM FOREST V{config.version} — Melhores hiperparâmetros")
    print(f"{'='*55}")
    print(f"  n_estimators     : {rf.n_estimators}")
    print(f"  max_depth        : {rf.max_depth}")
    print(f"  min_samples_leaf : {rf.min_samples_leaf}")
    print(f"  class_weight     : {rf.class_weight}")


def print_classification_reports(dt, rf, X_test, y_test) -> None:
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    print("=" * 60)
    print("  DECISION TREE — Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred_dt,
                                target_names=["bom", "ruim"]))
    print("=" * 60)
    print("  RANDOM FOREST — Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred_rf,
                                target_names=["bom", "ruim"]))


def build_results_table(dt, rf, X_test, y_test) -> pd.DataFrame:
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
            "Precision (ruim)": f"{(y_test[y_pred == 1] == 1).mean():.4f}",
            "ROC-AUC": f"{roc_auc_score(y_test, y_proba):.4f}",
        })
    return pd.DataFrame(rows)


def plot_confusion_matrices(dt, rf, X_test, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, name in [(axes[0], dt, "Decision Tree"),
                            (axes[1], rf, "Random Forest")]:
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=["bom", "ruim"],
            cmap="Blues", ax=ax, values_format="d",
        )
        ax.set_title(f"Matriz de Confusão — {name}", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list[str],
                            title: str, color: str, ax,
                            top_n: int = 15) -> None:
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    labels = [translate(feature_names[i]) for i in idx]
    values = imp[idx]
    bars = ax.barh(range(top_n), values, color=color)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini)")
    ax.set_title(title, fontsize=13)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)


def plot_feature_importances_side_by_side(dt, rf,
                                          feature_names: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_feature_importance(dt, feature_names,
                            "Top 15 Features — Decision Tree",
                            "#4C72B0", axes[0])
    plot_feature_importance(rf, feature_names,
                            "Top 15 Features — Random Forest",
                            "#DD8452", axes[1])
    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves(dt, rf, X_test, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    RocCurveDisplay.from_estimator(dt, X_test, y_test, ax=ax,
                                   name="Decision Tree")
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax,
                                   name="Random Forest")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório")
    ax.set_title("Curva ROC — Classe 'ruim'", fontsize=13)
    ax.legend()

    ax = axes[1]
    PrecisionRecallDisplay.from_estimator(dt, X_test, y_test, ax=ax,
                                          name="Decision Tree")
    PrecisionRecallDisplay.from_estimator(rf, X_test, y_test, ax=ax,
                                          name="Random Forest")
    prevalence = y_test.mean()
    ax.axhline(y=prevalence, color="k", linestyle="--", alpha=0.5,
               label=f"Prevalência ({prevalence:.3f})")
    ax.set_title("Curva Precision-Recall — Classe 'ruim'", fontsize=13)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(dt, rf, X_test, y_test) -> None:
    metrics_data = {}
    for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
        yp = model.predict(X_test)
        yproba = model.predict_proba(X_test)[:, 1]
        metrics_data[name] = {
            "Accuracy": accuracy_score(y_test, yp),
            "F1 (bom)": f1_score(y_test, yp, pos_label=0),
            "F1 (ruim)": f1_score(y_test, yp, pos_label=1),
            "Recall (ruim)": (yp[y_test == 1] == 1).mean(),
            "ROC-AUC": roc_auc_score(y_test, yproba),
        }

    metric_names = list(next(iter(metrics_data.values())).keys())
    x = np.arange(len(metric_names))
    width = 0.30

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model_name, vals) in enumerate(metrics_data.items()):
        values = [vals[m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=model_name,
                      color=["#4C72B0", "#DD8452"][i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos — Conjunto de Teste", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(dt, rf, X_train, y_train) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train_sizes_frac = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    for ax, model, name in [(axes[0], dt, "Decision Tree"),
                            (axes[1], rf, "Random Forest")]:
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5,
            train_sizes=train_sizes_frac,
            scoring="f1", n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color="#4C72B0")
        ax.fill_between(sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color="#DD8452")
        ax.plot(sizes, train_mean, "o-", color="#4C72B0", label="Treino")
        ax.plot(sizes, val_mean, "o-", color="#DD8452",
                label="Validação (CV)")
        ax.set_xlabel("Tamanho do treino")
        ax.set_ylabel("F1-score")
        ax.set_title(f"Curva de Aprendizado — {name}", fontsize=13)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Interpretation ───────────────────────────────────────────────


def plot_decision_tree(dt, feature_names: list[str]) -> None:
    translated = [translate(f) for f in feature_names]
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        dt, feature_names=translated,
        class_names=["Bom", "Ruim"],
        filled=True, rounded=True,
        max_depth=3, fontsize=9, ax=ax,
        impurity=False, proportion=True,
    )
    ax.set_title("Árvore de Decisão (primeiros 3 níveis)",
                 fontsize=15, pad=15)
    plt.tight_layout()
    plt.show()


def print_tree_rules(config) -> None:
    rules_file = config.eval_dir / f"decision_tree_rules_chess{config.rules_suffix}.txt"
    if not rules_file.exists():
        rules_file = Path("data/evaluation/decision_tree_rules_chess.txt")
    if rules_file.exists():
        rules_text = rules_file.read_text()
        print(rules_text[:3000])
        if len(rules_text) > 3000:
            print("\n... (regras continuam)")
    else:
        print(f"Arquivo de regras não encontrado: {rules_file}")


def show_error_examples(config, model_suffix: str = "") -> None:
    """Show FP/FN error examples from pre-computed CSVs."""
    fp_file = config.eval_dir / f"error_analysis_fp{config.rules_suffix}.csv"
    fn_file = config.eval_dir / f"error_analysis_fn{config.rules_suffix}.csv"
    if not fp_file.exists():
        fp_file = Path("data/evaluation/error_analysis_fp.csv")
        fn_file = Path("data/evaluation/error_analysis_fn.csv")

    df_fp = pd.read_csv(fp_file)
    df_fn = pd.read_csv(fn_file)

    _model_suffix = {1: "", 2: " V2", 3: " V3"}
    suffix = _model_suffix.get(config.version, "")
    dt_name = f"Decision Tree{suffix}"
    rf_name = f"Random Forest{suffix}"

    for error_type, df in [("FP", df_fp), ("FN", df_fn)]:
        for model_name in [dt_name, rf_name]:
            _show_error_block(df, error_type, model_name)


def _show_error_block(df: pd.DataFrame, error_type: str,
                      model_name: str, n: int = 5) -> None:
    subset = df[df["model"] == model_name].head(n)
    tipo = ("FALSOS POSITIVOS" if error_type == "FP"
            else "FALSOS NEGATIVOS")
    desc = ('modelo disse "ruim", lance é bom' if error_type == "FP"
            else 'modelo disse "bom", lance é ruim')

    print(f"\n{'='*65}")
    print(f"  {tipo} — {model_name}")
    print(f"  ({desc})")
    print(f"{'='*65}")

    for i, (_, row) in enumerate(subset.iterrows(), 1):
        print(f"\n  {error_type}-{i}: {row['move_san']} "
              f"(lance #{int(row['move_number'])}, {row['color']})")
        print(f"    Delta: {row['delta_cp']} cp | "
              f"Partida: {row['game_site']}")
        feats = str(row.get("top_features", "")).split("; ")
        translated_feats = [
            f"{translate(f.split('=')[0])}={f.split('=')[1]}"
            for f in feats if "=" in f
        ]
        print(f"    Features: {'; '.join(translated_feats)}")


# ── Diagnostics ──────────────────────────────────────────────────


def plot_diagnostic(df_features: pd.DataFrame,
                    feature_cols: list[str],
                    version_label: str) -> None:
    """Point-biserial correlation + Cohen's d for feature-label separation."""
    y_binary = (df_features["label"] == "ruim").astype(int)

    corrs = []
    for col in feature_cols:
        r, _ = pointbiserialr(y_binary, df_features[col])
        mean_bom = df_features.loc[y_binary == 0, col].mean()
        mean_ruim = df_features.loc[y_binary == 1, col].mean()
        std_pool = np.sqrt(
            (df_features.loc[y_binary == 0, col].std() ** 2
             + df_features.loc[y_binary == 1, col].std() ** 2) / 2
        )
        d = (mean_ruim - mean_bom) / std_pool if std_pool > 0 else 0
        corrs.append({"feature": col, "r": r, "cohen_d": abs(d)})

    df_corr = pd.DataFrame(corrs).sort_values("cohen_d", ascending=False)
    top = df_corr.head(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    labels_pt = [translate(f) for f in top["feature"]]
    colors_r = ["#C44E52" if abs(r) >= 0.10 else "#8C8C8C"
                for r in top["r"]]
    ax.barh(range(len(top)), top["r"].abs().values, color=colors_r)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels_pt)
    ax.invert_yaxis()
    ax.set_xlabel("|Correlação point-biserial|")
    ax.set_title(f"Top 10 Features {version_label} — Correlação com Label",
                 fontsize=12)
    ax.axvline(0.10, color="green", linestyle="--", alpha=0.6,
               label="|r| = 0.10")
    ax.axvline(0.20, color="red", linestyle="--", alpha=0.6,
               label="|r| = 0.20")
    ax.legend(fontsize=8)

    ax = axes[1]
    colors_d = ["#DD8452" if d >= 0.20 else "#8C8C8C"
                for d in top["cohen_d"]]
    ax.barh(range(len(top)), top["cohen_d"].values, color=colors_d)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels_pt)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (separabilidade)")
    ax.set_title(f"Top 10 Features {version_label} — Separabilidade",
                 fontsize=12)
    ax.axvline(0.20, color="green", linestyle="--", alpha=0.6,
               label="d = 0.20 (pequeno)")
    ax.axvline(0.50, color="red", linestyle="--", alpha=0.6,
               label="d = 0.50 (médio)")
    ax.legend(fontsize=8)

    plt.suptitle(f"Diagnóstico {version_label}: Separabilidade features–label",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"Melhor feature: {top.iloc[0]['feature']} → "
          f"|r| = {abs(top.iloc[0]['r']):.3f}, "
          f"Cohen's d = {top.iloc[0]['cohen_d']:.3f}")
    n_above = sum(1 for _, row in df_corr.iterrows()
                  if abs(row["r"]) >= 0.10)
    print(f"Features com |r| ≥ 0.10: {n_above}")


# ── Comparison across versions ───────────────────────────────────


def plot_version_metrics_bars(configs: list, X_test, y_test) -> None:
    """Bar chart comparing all versions side by side."""
    metric_labels = {
        "accuracy": "Accuracy",
        "f1_ruim": "F1 (ruim)",
        "recall_ruim": "Recall (ruim)",
        "precision_ruim": "Precision (ruim)",
        "roc_auc": "ROC-AUC",
    }
    metrics_to_plot = list(metric_labels.keys())
    x = np.arange(len(metrics_to_plot))

    n_configs = len(configs)
    n_bars = n_configs * 2
    width = 0.80 / n_bars

    dt_colors = ["#C8D8E8", "#7BA3CC", "#4C72B0"]
    rf_colors = ["#F5D8B0", "#E8A86E", "#DD8452"]

    fig, ax = plt.subplots(figsize=(16, 6))
    bar_idx = 0
    for ci, cfg in enumerate(configs):
        dt, rf, fnames = cfg.load_models()
        X_sub = X_test[fnames]
        for model, name_prefix, color in [
            (dt, "DT", dt_colors[ci % len(dt_colors)]),
            (rf, "RF", rf_colors[ci % len(rf_colors)]),
        ]:
            yp = model.predict(X_sub)
            yproba = model.predict_proba(X_sub)[:, 1]
            values = [
                accuracy_score(y_test, yp),
                f1_score(y_test, yp, pos_label=1),
                (yp[y_test == 1] == 1).mean(),
                (y_test[yp == 1] == 1).mean() if (yp == 1).any() else 0,
                roc_auc_score(y_test, yproba),
            ]
            label = f"{name_prefix} V{cfg.version}"
            bars = ax.bar(x + bar_idx * width, values, width,
                          label=label, color=color,
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.006,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90 if n_bars > 4 else 0)
            bar_idx += 1

    ax.set_xticks(x + (n_bars - 1) * width / 2)
    ax.set_xticklabels([metric_labels[m] for m in metrics_to_plot])
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score")
    title_versions = " vs ".join(f"V{c.version}" for c in configs)
    ax.set_title(f"Evolução {title_versions} — Todas as Métricas",
                 fontsize=14)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_version_roc_pr_overlay(configs: list, X_test, y_test) -> None:
    """Overlay ROC and PR curves for multiple versions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    styles = [(":", 0.4), ("--", 0.6), ("-", 1.0)]

    for ci, cfg in enumerate(configs):
        dt, rf, fnames = cfg.load_models()
        X_sub = X_test[fnames]
        ls, alpha = styles[ci % len(styles)]
        lw = 2 if ci == len(configs) - 1 else 1

        RocCurveDisplay.from_estimator(
            dt, X_sub, y_test, ax=axes[0],
            name=f"DT V{cfg.version}", linestyle=ls, alpha=alpha,
            linewidth=lw)
        RocCurveDisplay.from_estimator(
            rf, X_sub, y_test, ax=axes[0],
            name=f"RF V{cfg.version}", linestyle=ls, alpha=alpha,
            linewidth=lw)
        PrecisionRecallDisplay.from_estimator(
            dt, X_sub, y_test, ax=axes[1],
            name=f"DT V{cfg.version}", linestyle=ls, alpha=alpha,
            linewidth=lw)
        PrecisionRecallDisplay.from_estimator(
            rf, X_sub, y_test, ax=axes[1],
            name=f"RF V{cfg.version}", linestyle=ls, alpha=alpha,
            linewidth=lw)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    title_versions = " vs ".join(f"V{c.version}" for c in configs)
    axes[0].set_title(f"Curvas ROC — {title_versions}", fontsize=13)
    axes[0].legend(loc="lower right", fontsize=8)

    prevalence = y_test.mean()
    axes[1].axhline(y=prevalence, color="k", linestyle="--", alpha=0.3,
                    label=f"Prevalência ({prevalence:.3f})")
    axes[1].set_title(f"Curvas Precision-Recall — {title_versions}",
                      fontsize=13)
    axes[1].legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_top_features_colored(rf, feature_names: list[str],
                              v1_names: list[str],
                              v2_names: list[str],
                              v3_new_features: list[str],
                              top_n: int = 20) -> None:
    """Top features bar chart colored by version origin."""
    color_map = {}
    for f in feature_names:
        if f in v3_new_features:
            color_map[f] = "#55A868"
        elif f in v2_names and f not in v1_names:
            color_map[f] = "#C44E52"
        else:
            color_map[f] = "#4C72B0"

    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [translate(feature_names[i]) for i in idx]
    values = imp[idx]
    colors = [color_map[feature_names[i]] for i in idx]
    bars = ax.barh(range(len(idx)), values, color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini)")
    ax.set_title(f"Top {top_n} Features — Random Forest V3", fontsize=13)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    legend_elements = [
        Patch(facecolor="#4C72B0", label="V1 — Posicionais"),
        Patch(facecolor="#C44E52", label="V2 — Táticas"),
        Patch(facecolor="#55A868", label="V3 — Look-ahead"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_tactical_features_importance(config) -> None:
    """Bar chart of the 19 tactical features' importance in RF V2."""
    csv_path = config.eval_dir / "tactical_features_analysis.csv"
    if not csv_path.exists():
        print(f"Arquivo não encontrado: {csv_path}")
        return

    df_tact = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_tact = [
        "#C44E52" if imp >= 0.03
        else "#DD8452" if imp >= 0.01
        else "#8C8C8C"
        for imp in df_tact["importance_rf_v2"]
    ]
    ax.barh(range(len(df_tact)), df_tact["importance_rf_v2"].values,
            color=colors_tact)
    ax.set_yticks(range(len(df_tact)))
    ax.set_yticklabels(df_tact["feature_pt"])
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini) — Random Forest V2")
    ax.set_title("Importância das 19 Features Táticas (V2)", fontsize=13)

    for i, (_, row) in enumerate(df_tact.iterrows()):
        ax.text(row["importance_rf_v2"] + 0.001, i,
                f"{row['importance_rf_v2']:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

    print(f"Feature tática #1: {df_tact.iloc[0]['feature_pt']} "
          f"(importância = {df_tact.iloc[0]['importance_rf_v2']:.4f})")


def print_version_deltas(deltas_csv: Path) -> None:
    """Print metric deltas between versions from pre-computed CSV."""
    df = pd.read_csv(deltas_csv)
    cols = [c for c in df.columns if c.startswith("v")]
    delta_cols = [c for c in df.columns if c.startswith("delta")]

    print(f"{'='*70}")
    print(f"  COMPARAÇÃO — Deltas por Métrica")
    print(f"{'='*70}")
    for algo in df["algo"].unique():
        subset = df[df["algo"] == algo]
        print(f"\n  {algo}:")
        for _, row in subset.iterrows():
            parts = []
            for vc in cols:
                parts.append(f"{vc.upper()}={row[vc]:.4f}")
            total_delta = row[delta_cols[-1]] if delta_cols else 0
            arrow = "↑" if total_delta > 0 else "↓"
            pp_col = [c for c in df.columns if c.endswith("_pp")]
            pp_val = abs(row[pp_col[-1]]) if pp_col else abs(total_delta * 100)
            print(f"    {row['metric']:18s}  {' → '.join(parts)}  "
                  f"({arrow} {pp_val:.2f} pp)")
