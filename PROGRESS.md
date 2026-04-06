# Progresso do Projeto

Arquivo de tracking para desenvolvimento iterativo. Atualizado automaticamente pelos commands de cada etapa.

## Status das etapas

| Etapa | Status | Última atualização | Observações |
|-------|--------|--------------------|-------------|
| 00 — Setup do ambiente | concluída | 2026-04-06 | Todas as dependências instaladas, Stockfish 18 OK |
| 01 — Coleta de dados | concluída | 2026-04-06 | PGN: `lichess_db_standard_rated_2015-01.pgn.zst` (~272 MiB); scripts `src/download_pgn.py`, `src/pgn_stream.py` |
| 02 — Filtragem e amostragem | concluída | 2026-04-06 | 452,929 scanned → 3,000 aceites (seed=42, rate=0.10); 136,620 lances mid-game; Elo médio 1561 |
| 03 — Rotulagem Stockfish | concluída | 2026-04-06 | Depth 15, 6 workers, ~174 min. 109,290 rotulados (92,197 bom / 17,093 ruim), 27,330 descartados. Ratio bom:ruim = 5.39:1 |
| 04 — Engenharia de features | concluída | 2026-04-06 | 33 features (7 grupos), 109,290 linhas, 4.1s com 6 workers, sem nulos |
| 05 — Treino dos modelos | concluída | 2026-04-06 | DT (gini, depth=7, leaf=20, CV-F1=0.3253) + RF (200 trees, depth=10, leaf=5, CV-F1=0.3499). Split 70/15/15 estratificado. RF superior em todas as métricas de teste. |
| 06 — Avaliação e interpretação | concluída | 2026-04-06 | 13 artefatos gerados (plots + CSVs + regras traduzidas). Confusion matrices, feature importance (DT vs RF), ROC/PR curves, learning curves, análise qualitativa de 10 FP + 10 FN por modelo. RF superior em accuracy (0.68 vs 0.62), F1-ruim (0.35 vs 0.33), AUC (0.68 vs 0.65). DT tem recall-ruim ligeiramente maior (0.60 vs 0.55). |
| 07 — Notebook final | concluída | 2026-04-06 | 39 células (24 code + 15 md), 8 seções. Pipeline autocontido (RERUN_PIPELINE flag). Inclui curvas de aprendizado. Execução ~28s. README.md criado. |
| **V2 — Upgrade** | | | |
| 08 — Features táticas | concluída | 2026-04-06 | 19 features táticas (G8–G12) adicionadas a `extract_features.py` com flag `--v2`. 52 features totais, 109,290 linhas, ~9s com 6 workers. Melhor nova feature: `contested_squares` (r=0.143, d=0.398) — supera a melhor V1. Sem NaN. |
| 09 — Re-treino V2 | concluída | 2026-04-06 | DT (gini, depth=7, leaf=1, CV-F1=0.3378) + RF (200 trees, depth=10, leaf=5, CV-F1=0.3638). Split idêntico à V1 (70/15/15). RF: accuracy 0.69 (+0.9pp), F1-ruim 0.37 (+1.4pp), AUC 0.71 (+2.4pp). `contested_squares` é feature #1 em ambos. |
| 10 — Avaliação V1 vs V2 | concluída | 2026-04-06 | 18 artefatos gerados em `data/evaluation_v2/`. Comparação V1→V2: RF accuracy +0.85pp (0.68→0.69), F1-ruim +1.39pp (0.35→0.37), AUC +2.42pp (0.68→0.71), recall-ruim +1.87pp (0.55→0.57). DT: accuracy +1.26pp, F1-ruim +1.49pp, AUC +2.57pp. `contested_squares` é feature #1 em ambos. Melhoria consistente mas modesta — confirma que features posicionais+táticas leves ajudam mas não resolvem o gap (F1-ruim ainda <0.40). |
| 11 — Notebook V2 | concluída | 2026-04-06 | 46 células (31 code + 15 md). Adicionadas seções: 6.6 Diagnóstico V1 (correlações, Cohen's d), 6.7 Comparação V1 vs V2 (tabela de deltas, barplot, ROC/PR overlay, features táticas), conclusão V2 com ciclo diagnóstico→melhoria. Execução ~34s. |
| **V3 — Look-ahead** | | | |
| 12 — Features look-ahead | concluída | 2026-04-06 | 15 features look-ahead (G13 delta 8, G14 resposta adversário 4, G15 SEE 3) adicionadas a `extract_features.py` com flag `--v3`. 67 features totais, 109,290 linhas, ~23s com 6 workers. Bug na SEE (negamax) detectado e corrigido. Melhores novas features: `opponent_best_capture_value` (r=0.147, d=0.383), `opponent_can_check` (r=0.120, d=0.335), `see_of_move` (r=-0.107, d=-0.334). 5 features com |r|>0.10. Sem NaN. |
| 13 — Re-treino V3 | concluída | 2026-04-06 | DT (entropy, depth=10, leaf=20, CV-F1=0.3826) + RF (200 trees, depth=15, leaf=10, CV-F1=0.4354). Split idêntico (70/15/15). RF: accuracy 0.7849 (+9.37pp vs V2), F1-ruim 0.4312 (+6.48pp), AUC 0.7678 (+5.99pp), precision-ruim 0.3676 (+9.78pp). DT: accuracy 0.6762 (+4.60pp), F1-ruim 0.3821 (+3.92pp), AUC 0.7176 (+4.32pp). `worst_see_against_player` é feature #1 em ambos. Look-ahead domina o top 5. |
| 14 — Avaliação V2 vs V3 | concluída | 2026-04-06 | 18 artefatos em `data/evaluation_v3/`. RF V3: accuracy 0.7849 (+9.37pp vs V2, +10.22pp vs V1), F1-ruim 0.4312 (+6.48pp vs V2, +7.87pp vs V1), AUC 0.7678 (+5.99pp vs V2, +8.41pp vs V1), precision-ruim 0.3676 (+9.78pp vs V2). recall-ruim caiu 4.95pp (trade-off precision↑/recall↓ → F1 sobe). DT V3: accuracy +4.60pp, F1-ruim +3.91pp, AUC +4.31pp vs V2. Look-ahead features dominam: `worst_see_against_player` (#1, 0.0611), `delta_mobility_opponent` (#2, 0.0478). Meta AUC ≥ 0.75 atingida (0.7678). Meta F1 ≥ 0.50 não atingida mas salto de +6.48pp é o maior do projeto. |
| 15 — Notebook V3 | concluída | 2026-04-06 | 53 células (32 code + 21 md). Novas seções: 6.8 Diagnóstico V2 (correlações V2 + insight "mesmo features para lances diferentes"), 6.9 Comparação V1→V2→V3 (tabela de deltas 3-way, barplot 6 barras, ROC/PR overlay 6 curvas, importância das 15 look-ahead features com cores V1/V2/V3). Conclusão reescrita com tabela V1×V2×V3, narrativa do triplo ciclo diagnóstico→melhoria, evolução das regras da DT (tipo→tensão→consequências). Execução ~42s com VERSION=3, RERUN_PIPELINE=False. |
| **V4 — XGBoost + Threshold** | | | |
| 16 — Documentação V4 | concluída | 2026-04-06 | Diagnóstico V3 atualizado com resultados reais (F1=0.43, AUC=0.77). Spec V4 criado: `docs/04-modelagem/upgrade-v4-xgboost.md`. Motivação: learning curves estáveis indicam teto no modelo, não nas features. XGBoost + threshold tuning sobre mesmas 67 features V3. Meta: F1 ≥ 0.50. |
| 17 — Treino V4 | concluída | 2026-04-06 | DT (entropy, depth=10, leaf=20, CV-F1=0.3826) + RF (200 trees, depth=15, leaf=10, CV-F1=0.4354) + XGB (300 trees, depth=7, lr=0.05, min_child=5, CV-F1=0.4487). XGB supera RF em CV-F1 (+1.33pp) e AUC (0.7809 vs 0.7678). Threshold tuning: DT=0.60, RF=0.52, XGB=0.61. XGB default: F1-ruim=0.4410 (+0.98pp vs RF), recall-ruim=0.617 (+9.55pp vs RF), AUC=0.7809 (+1.31pp). XGB com threshold: F1=0.4414, precision-ruim=0.4255 (primeira vez >0.40!). `is_losing_capture` é feature #1 do XGB (0.0785) — diferente do RF (#1=`worst_see_against_player`). |
| 18 — Avaliação V4 | concluída | 2026-04-06 | 15 artefatos em `data/evaluation_v4/`. Comparação 4-way (9 modelos: DT/RF V1–V4 + XGB V4). XGB V4: F1-ruim=0.4410 (+0.98pp vs RF V3), recall-ruim=0.6170 (+9.55pp vs RF V3), AUC=0.7809 (+1.31pp vs RF V3). DT V4 e RF V4 idênticos a V3 (mesmas features, mesmos hiperparâmetros ótimos). Threshold analysis: DT=0.60, RF=0.52, XGB=0.61. Feature importance XGB: `is_losing_capture` #1 (0.0785), diferente do RF (#1=`worst_see_against_player`). Error analysis (10 FP + 10 FN do XGB). Gráficos: comparison 9 barras, ROC/PR overlay 9 curvas, confusion matrices 3-panel, feature importance 3-way (DT vs RF vs XGB), threshold sweep. |
| 19 — Notebook V4 | concluída | 2026-04-06 | Notebook `comparacao.ipynb` atualizado para V1→V4: título, imports, modelo XGBoost, tabela comparativa com 9 modelos, deltas V4, barplot 9 barras, ROC/PR overlay com XGB, novas seções 10.1 (feature importance XGB com cores), 10.2 (threshold analysis sweep), 10.3 (métricas XGB default vs tuned), síntese atualizada. `notebook_utils.py` adaptado: `_unpack_models`, `plot_threshold_analysis`, `plot_xgb_feature_importance`. 26 células executam em ~8s. |

## Artefatos gerados

| Artefato | Caminho | Gerado em |
|----------|---------|-----------|
| requirements.txt | `requirements.txt` | 2026-04-06 |
| .gitignore | `.gitignore` | 2026-04-06 |
| Pastas do projeto | `data/`, `notebooks/`, `src/` | 2026-04-06 |
| Ambiente virtual | `.venv/` | 2026-04-06 |
| Lichess PGN (raw) | `data/raw/lichess_db_standard_rated_2015-01.pgn.zst` | 2026-04-06 |
| Download + streaming | `src/download_pgn.py`, `src/pgn_stream.py` | 2026-04-06 |
| Script de filtragem | `src/filter_games.py` | 2026-04-06 |
| CSV filtrado | `data/filtered/moves_filtered.csv` (136,620 linhas) | 2026-04-06 |
| Script de rotulagem | `src/label_moves.py` | 2026-04-06 |
| CSV com scores (todos) | `data/labeled/moves_all_scored.csv` (136,620 linhas) | 2026-04-06 |
| CSV rotulado (bom/ruim) | `data/labeled/moves_labeled.csv` (109,290 linhas) | 2026-04-06 |
| Script de features | `src/extract_features.py` | 2026-04-06 |
| CSV de features | `data/features/features.csv` (109,290 linhas, 33 features + label) | 2026-04-06 |
| Script de treino | `src/train_models.py` | 2026-04-06 |
| Modelo DT | `data/models/decision_tree.joblib` | 2026-04-06 |
| Modelo RF | `data/models/random_forest.joblib` | 2026-04-06 |
| Resultados | `data/models/results.csv` | 2026-04-06 |
| Split info | `data/models/split_info.csv` | 2026-04-06 |
| Feature names | `data/models/feature_names.json` | 2026-04-06 |
| Script de avaliação | `src/evaluate_models.py` | 2026-04-06 |
| Matrizes de confusão | `data/evaluation/confusion_matrices.png` | 2026-04-06 |
| Feature importance DT | `data/evaluation/feature_importance_dt.png` | 2026-04-06 |
| Feature importance RF | `data/evaluation/feature_importance_rf.png` | 2026-04-06 |
| Feature importance comparação | `data/evaluation/feature_importance_comparison.png` | 2026-04-06 |
| Comparação de modelos | `data/evaluation/model_comparison.png` | 2026-04-06 |
| Curvas ROC | `data/evaluation/roc_curves.png` | 2026-04-06 |
| Curvas Precision-Recall | `data/evaluation/precision_recall_curves.png` | 2026-04-06 |
| Curvas de aprendizado | `data/evaluation/learning_curves.png` | 2026-04-06 |
| Regras DT (raw) | `data/evaluation/decision_tree_rules.txt` | 2026-04-06 |
| Regras DT (xadrez) | `data/evaluation/decision_tree_rules_chess.txt` | 2026-04-06 |
| Análise de erros FP | `data/evaluation/error_analysis_fp.csv` | 2026-04-06 |
| Análise de erros FN | `data/evaluation/error_analysis_fn.csv` | 2026-04-06 |
| Notebook final | `notebooks/chess_move_classifier.ipynb` | 2026-04-06 |
| README | `README.md` | 2026-04-06 |
| Diagnóstico V1 | `docs/06-riscos-e-limitacoes/diagnostico-v1.md` | 2026-04-06 |
| Spec features táticas | `docs/03-features/features-taticas.md` | 2026-04-06 |
| CSV features V2 | `data/features/features_v2.csv` (109,290 linhas, 52 features + label) | 2026-04-06 |
| Modelo DT V2 | `data/models_v2/decision_tree.joblib` | 2026-04-06 |
| Modelo RF V2 | `data/models_v2/random_forest.joblib` | 2026-04-06 |
| Resultados V2 | `data/models_v2/results.csv` | 2026-04-06 |
| Split info V2 | `data/models_v2/split_info.csv` | 2026-04-06 |
| Feature names V2 | `data/models_v2/feature_names.json` | 2026-04-06 |
| Script de avaliação V2 | `src/evaluate_v2.py` | 2026-04-06 |
| Confusion matrices V2 | `data/evaluation_v2/confusion_matrices_v2.png` | 2026-04-06 |
| Feature importance DT V2 | `data/evaluation_v2/feature_importance_dt_v2.png` | 2026-04-06 |
| Feature importance RF V2 | `data/evaluation_v2/feature_importance_rf_v2.png` | 2026-04-06 |
| Feature importance comparação V2 | `data/evaluation_v2/feature_importance_comparison_v2.png` | 2026-04-06 |
| ROC curves V2 | `data/evaluation_v2/roc_curves_v2.png` | 2026-04-06 |
| PR curves V2 | `data/evaluation_v2/precision_recall_curves_v2.png` | 2026-04-06 |
| Learning curves V2 | `data/evaluation_v2/learning_curves_v2.png` | 2026-04-06 |
| Regras DT V2 (raw) | `data/evaluation_v2/decision_tree_rules_v2.txt` | 2026-04-06 |
| Regras DT V2 (xadrez) | `data/evaluation_v2/decision_tree_rules_chess_v2.txt` | 2026-04-06 |
| Erros FP V2 | `data/evaluation_v2/error_analysis_fp_v2.csv` | 2026-04-06 |
| Erros FN V2 | `data/evaluation_v2/error_analysis_fn_v2.csv` | 2026-04-06 |
| V1 vs V2 métricas | `data/evaluation_v2/v1_vs_v2_metrics.csv` | 2026-04-06 |
| V1 vs V2 deltas | `data/evaluation_v2/v1_vs_v2_deltas.csv` | 2026-04-06 |
| V1 vs V2 comparação (plot) | `data/evaluation_v2/v1_vs_v2_comparison.png` | 2026-04-06 |
| V1 vs V2 ROC overlay | `data/evaluation_v2/v1_vs_v2_roc_overlay.png` | 2026-04-06 |
| V1 vs V2 PR overlay | `data/evaluation_v2/v1_vs_v2_pr_overlay.png` | 2026-04-06 |
| Features táticas importância | `data/evaluation_v2/v1_vs_v2_feature_importance_new.png` | 2026-04-06 |
| Análise features táticas | `data/evaluation_v2/tactical_features_analysis.csv` | 2026-04-06 |
| Diagnóstico V2 | `docs/06-riscos-e-limitacoes/diagnostico-v2.md` | 2026-04-06 |
| Spec features look-ahead | `docs/03-features/features-lookahead.md` | 2026-04-06 |
| Diagnóstico V3 (evolução) | `docs/06-riscos-e-limitacoes/diagnostico-v3.md` | 2026-04-06 |
| CSV features V3 | `data/features/features_v3.csv` (109,290 linhas, 67 features + label) | 2026-04-06 |
| Modelo DT V3 | `data/models_v3/decision_tree.joblib` | 2026-04-06 |
| Modelo RF V3 | `data/models_v3/random_forest.joblib` | 2026-04-06 |
| Resultados V3 | `data/models_v3/results.csv` | 2026-04-06 |
| Split info V3 | `data/models_v3/split_info.csv` | 2026-04-06 |
| Feature names V3 | `data/models_v3/feature_names.json` | 2026-04-06 |
| Script de avaliação V3 | `src/evaluate_v3.py` | 2026-04-06 |
| Confusion matrices V3 | `data/evaluation_v3/confusion_matrices_v3.png` | 2026-04-06 |
| Feature importance DT V3 | `data/evaluation_v3/feature_importance_dt_v3.png` | 2026-04-06 |
| Feature importance RF V3 | `data/evaluation_v3/feature_importance_rf_v3.png` | 2026-04-06 |
| Feature importance comparação V3 | `data/evaluation_v3/feature_importance_comparison_v3.png` | 2026-04-06 |
| ROC curves V3 | `data/evaluation_v3/roc_curves_v3.png` | 2026-04-06 |
| PR curves V3 | `data/evaluation_v3/precision_recall_curves_v3.png` | 2026-04-06 |
| Learning curves V3 | `data/evaluation_v3/learning_curves_v3.png` | 2026-04-06 |
| Regras DT V3 (raw) | `data/evaluation_v3/decision_tree_rules_v3.txt` | 2026-04-06 |
| Regras DT V3 (xadrez) | `data/evaluation_v3/decision_tree_rules_chess_v3.txt` | 2026-04-06 |
| Erros FP V3 | `data/evaluation_v3/error_analysis_fp_v3.csv` | 2026-04-06 |
| Erros FN V3 | `data/evaluation_v3/error_analysis_fn_v3.csv` | 2026-04-06 |
| V1 vs V2 vs V3 métricas | `data/evaluation_v3/v1_v2_v3_metrics.csv` | 2026-04-06 |
| V1 vs V2 vs V3 deltas | `data/evaluation_v3/v1_v2_v3_deltas.csv` | 2026-04-06 |
| V1 vs V2 vs V3 comparação (plot) | `data/evaluation_v3/v1_v2_v3_comparison.png` | 2026-04-06 |
| V1 vs V2 vs V3 ROC overlay | `data/evaluation_v3/v1_v2_v3_roc_overlay.png` | 2026-04-06 |
| V1 vs V2 vs V3 PR overlay | `data/evaluation_v3/v1_v2_v3_pr_overlay.png` | 2026-04-06 |
| Features look-ahead importância | `data/evaluation_v3/v3_lookahead_feature_importance.png` | 2026-04-06 |
| Análise features look-ahead | `data/evaluation_v3/lookahead_features_analysis.csv` | 2026-04-06 |
| Spec V4 XGBoost | `docs/04-modelagem/upgrade-v4-xgboost.md` | 2026-04-06 |
| Modelo DT V4 | `data/models_v4/decision_tree.joblib` | 2026-04-06 |
| Modelo RF V4 | `data/models_v4/random_forest.joblib` | 2026-04-06 |
| Modelo XGBoost V4 | `data/models_v4/xgboost.joblib` | 2026-04-06 |
| Thresholds V4 | `data/models_v4/thresholds.json` | 2026-04-06 |
| Resultados V4 | `data/models_v4/results.csv` | 2026-04-06 |
| Split info V4 | `data/models_v4/split_info.csv` | 2026-04-06 |
| Feature names V4 | `data/models_v4/feature_names.json` | 2026-04-06 |
| Script de avaliação V4 | `src/evaluate_v4.py` | 2026-04-06 |
| Confusion matrices V4 | `data/evaluation_v4/confusion_matrices_v4.png` | 2026-04-06 |
| Feature importance XGB V4 | `data/evaluation_v4/feature_importance_xgb_v4.png` | 2026-04-06 |
| Feature importance 3-way V4 | `data/evaluation_v4/feature_importance_3way_v4.png` | 2026-04-06 |
| ROC curves V4 | `data/evaluation_v4/roc_curves_v4.png` | 2026-04-06 |
| PR curves V4 | `data/evaluation_v4/precision_recall_curves_v4.png` | 2026-04-06 |
| Threshold analysis V4 | `data/evaluation_v4/threshold_analysis_v4.png` | 2026-04-06 |
| Erros FP V4 | `data/evaluation_v4/error_analysis_fp_v4.csv` | 2026-04-06 |
| Erros FN V4 | `data/evaluation_v4/error_analysis_fn_v4.csv` | 2026-04-06 |
| V1→V4 métricas | `data/evaluation_v4/v1_v2_v3_v4_metrics.csv` | 2026-04-06 |
| V1→V4 deltas | `data/evaluation_v4/v1_v2_v3_v4_deltas.csv` | 2026-04-06 |
| V1→V4 comparação (plot) | `data/evaluation_v4/v1_v2_v3_v4_comparison.png` | 2026-04-06 |
| V1→V4 ROC overlay | `data/evaluation_v4/v1_v2_v3_v4_roc_overlay.png` | 2026-04-06 |
| V1→V4 PR overlay | `data/evaluation_v4/v1_v2_v3_v4_pr_overlay.png` | 2026-04-06 |

## Decisões tomadas durante o desenvolvimento

_(registrar aqui qualquer desvio da documentação original e justificativa)_

- **Desbalanceamento de classes (etapa 03):** ratio bom:ruim = 5.39:1. Será tratado na modelagem com class_weight='balanced' ou técnicas de reamostragem (SMOTE/undersampling). A proporção está dentro do esperado para jogadores 1400–1700 Lichess — a maioria dos lances não é um erro grave.
- **Multiprocessing na rotulagem:** usados 6 workers paralelos com caching de avaliações consecutivas para reduzir tempo (~174 min vs ~5h estimadas em single-thread).
- **Features extraídas (etapa 04):** 33 features em 7 grupos (material 11, mobilidade 3, segurança do rei 4, estrutura de peões 4, controle do centro 3, características do lance 6, contexto 2). Todas numéricas/inteiras, sem nulos. Extração levou ~4s com 6 workers.
- **Treino dos modelos (etapa 05):** Split estratificado 70/15/15 (76,546 / 16,350 / 16,394). Decision Tree: best = gini, depth=7, min_leaf=20 (CV-F1=0.3253). Random Forest: best = 200 trees, depth=10, min_leaf=5 (CV-F1=0.3499). Ambos com class_weight="balanced". RF ganhou em accuracy (0.68 vs 0.62), F1-ruim (0.35 vs 0.33) e AUC (0.68 vs 0.65). Top features: move_number, legal_moves_opponent, is_capture, legal_moves_player, material_diff. F1-ruim baixo é esperado dado o desbalanceamento 5.4:1 e a natureza difícil de prever erros apenas por features posicionais.
- **Avaliação e interpretação (etapa 06):** 13 artefatos de avaliação gerados. FPs típicos: lances quietos em posições complicadas; FNs: erros táticos sutis não capturados pelas features posicionais. Regras da DT: primeira decisão é se o lance é captura, depois mobilidade do adversário, depois fase da partida. Learning curves mostram gap treino-validação estável — melhoria viria de features táticas, não de mais dados.
- **Notebook final (etapa 07):** 39 células (24 code + 15 markdown) em 8 seções. Pipeline autocontido via `RERUN_PIPELINE` flag — chama funções reais dos scripts src/ para download, filtragem, rotulagem, features e treino. Por padrão carrega dados pré-computados (~28s). Inclui curvas de aprendizado, análise de erros (FP/FN) para ambos os modelos, e regras traduzidas da DT. README.md criado com instruções de reprodução e resumo de resultados.
- **Features táticas (etapa 08):** 19 features em 5 grupos (G8 peças indefesas 5, G9 capturas com ganho 4, G10 cravadas 2, G11 rei avançado 4, G12 tensão 4). Extração ~9s com 6 workers via `--v2` flag. Melhor nova feature: `contested_squares` (r=0.143, Cohen's d=0.398) — única feature a ultrapassar |r|=0.10 e d>0.30. As features de hanging/pins ficaram com d<0.15, abaixo do esperado. Features de tensão (G12) são o grupo mais forte.
- **Re-treino V2 (etapa 09):** Mesmos grids e split estratificado da V1. DT best: gini, depth=7, min_samples_leaf=1 (CV-F1=0.3378, +1.25pp). RF best: 200 trees, depth=10, min_leaf=5 (CV-F1=0.3638, +1.39pp). Test set: RF accuracy 0.6912 (+0.85pp), F1-ruim 0.3664 (+1.39pp), AUC 0.7079 (+2.42pp), precision-ruim 0.2698 (+1.09pp). DT accuracy 0.6302 (+1.26pp), F1-ruim 0.3429 (+1.49pp), AUC 0.6744 (+2.57pp). `contested_squares` subiu a feature #1 em ambos os modelos (DT: 0.2966, RF: 0.1144). `hanging_value_opponent` entrou no top 3 da DT. Melhoria consistente mas modesta — confirma diagnóstico V1: features posicionais + táticas leves melhoram, mas não resolvem o gap fundamental.
- **Avaliação V1 vs V2 (etapa 10):** 18 artefatos gerados em `data/evaluation_v2/`. Comparação completa lado a lado incluindo: confusion matrices V2, feature importance (com features táticas destacadas em vermelho), ROC/PR overlay V1 vs V2, learning curves V2, regras DT V2 traduzidas, análise de erros (FP/FN), tabela de deltas por métrica. Todas as métricas melhoraram de forma consistente: RF AUC +2.42pp é o maior ganho absoluto. Features táticas no RF V2: `contested_squares` (0.1144) é a #1 geral, seguida por `total_attacks_opponent` (0.0396) e `hanging_value_opponent` (0.0316). Das 19 features táticas, 6 entraram no top 15 do RF. Learning curves V2 continuam estáveis (sem overfitting), confirmando que o teto de performance está nas features, não no modelo ou dados.
- **Features look-ahead (etapa 12):** 15 features em 3 grupos (G13 delta 8, G14 resposta adversário 4, G15 SEE 3). Extração ~23s com 6 workers via `--v3` flag. Bug no negamax da SEE original (sempre retornava ≥0) detectado e corrigido — o algoritmo usava `max(gains[-2], -gains[-1] + gains[-2])` que era no-op; substituído pelo swap padrão `gain[d-1] = -max(-gain[d-1], gain[d])`. Após correção: `see_of_move` ∈ [-8, 9], `is_losing_capture` ∈ [0, 1]. Grupo mais forte: G14 (resposta do adversário) — `opponent_best_capture_value` (r=0.147, d=0.383) é a melhor feature V3, seguida de `opponent_can_check` (r=0.120, d=0.335) e `opponent_num_good_captures` (r=0.116, d=0.292). G15 (SEE) também forte: `worst_see_against_player` (r=-0.111, d=-0.284), `see_of_move` (r=-0.107, d=-0.334). G13 (deltas) mais fracos (|r|<0.06), possivelmente por redundância com features "antes do lance". 5 features com |r|>0.10, todas superiores à maioria das features V2.
- **Notebook V2 (etapa 11):** 46 células (31 code + 15 markdown). Novas seções: 6.6 Diagnóstico V1 (correlação point-biserial + Cohen's d para todas as 33 features, com limites de referência |r|=0.10 e d=0.20/0.50), 6.7 Comparação V1 vs V2 (tabela de deltas, barplot 4 barras DT/RF × V1/V2, overlay ROC/PR com modelos V1 carregados em paralelo, importância das 19 features táticas). Conclusão reescrita com tabela V1 vs V2, narrativa do ciclo diagnóstico→melhoria, limitações e trabalhos futuros atualizados. Execução ~34s com VERSION=2, RERUN_PIPELINE=False.
- **Re-treino V3 (etapa 13):** Mesmos grids e split estratificado da V1/V2. DT best: entropy, depth=10, min_samples_leaf=20 (CV-F1=0.3826, +4.48pp vs V2). RF best: 200 trees, depth=15, min_leaf=10 (CV-F1=0.4354, +7.16pp vs V2). Salto enorme em todas as métricas — maior upgrade do projeto. RF accuracy 0.7849 (+9.37pp vs V2, +9.72pp vs V1), F1-ruim 0.4312 (+6.48pp vs V2, +7.87pp vs V1), AUC 0.7678 (+5.99pp vs V2, +8.41pp vs V1), precision-ruim 0.3676 (+9.78pp vs V2). RF max_depth subiu de 10→15 — as features look-ahead permitem árvores mais profundas e informativas. Feature importance: `worst_see_against_player` é #1 em ambos (DT: 0.1466, RF: 0.0611), `see_of_move` #2 na DT (0.1002), `opponent_can_check` #3 na DT (0.0840). Look-ahead features (V3) ocupam 3 dos top 5 em ambos os modelos, superando `contested_squares` (V2) que era #1 anterior. Meta de F1-ruim ≥ 0.50 não atingida mas o salto de +6.48pp é muito significativo.
- **Notebook V3 (etapa 15):** 53 células (32 code + 21 markdown) em 8 seções + 4 subseções novas. CONFIG suporta VERSION=1/2/3 com paths automáticos para features, modelos e avaliação. Seção 6.8 (Diagnóstico V2): correlações point-biserial + Cohen's d das 52 features V2 + insight conceptual ("mesmo features para lances diferentes na mesma posição"). Seção 6.9 (Comparação V1→V2→V3): tabela completa de deltas 3-way, barplot de 6 barras (DT/RF × V1/V2/V3), overlay ROC/PR com 6 curvas (V1 pontilhadas, V2 tracejadas, V3 sólidas), importância das 15 look-ahead features com código de 3 cores (azul V1, vermelho V2, verde V3). Seção 7 atualizada: regras da DT V3 com interpretação orientada a consequências (1ª decisão = pior SEE vs é captura). Conclusão reescrita com tabela V1×V2×V3, narrativa do triplo ciclo científico, evolução das regras. Células V1 vs V2 (34-37) generalizadas com `VERSION >= 2` e carga explícita de modelos V1/V2 para funcionar com qualquer versão. Execução ~42s.
- **Avaliação V2 vs V3 (etapa 14):** 18 artefatos gerados em `data/evaluation_v3/`. Comparação completa V1→V2→V3 com 6-bar chart, ROC/PR overlay 6 curvas, feature importance com código de 3 cores (azul=V1, vermelho=V2, verde=V3). RF V3: accuracy +10.22pp vs V1, F1-ruim +7.87pp, AUC +8.41pp, precision-ruim +10.87pp. Nota: RF V3 recall-ruim caiu 4.95pp vs V2 (de 0.571 para 0.521) — trade-off precision↑/recall↓ típico de árvores mais profundas com features mais informativas, mas o F1 subiu 6.48pp porque a precision ganhou 9.78pp. DT V3 mantém recall alto (0.640, +2.30pp vs V2). Meta AUC ≥ 0.75 **atingida** (RF: 0.7678). Meta F1 ≥ 0.50 não atingida (RF: 0.4312), mas o salto é o maior de todo o projeto (+6.48pp) e o ciclo diagnóstico→melhoria está claramente demonstrado. Look-ahead features no RF V3: `worst_see_against_player` (#1, 0.0611), `delta_mobility_opponent` (#2, 0.0478), `opponent_best_capture_value` (#3, 0.0460), `delta_mobility_player` (#4, 0.0379), `see_of_move` (#5, 0.0378). 6 das 15 look-ahead features no top 15 geral.
