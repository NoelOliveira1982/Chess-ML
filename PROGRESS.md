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
| 12 — Features look-ahead | pendente | | Features delta (antes vs depois do lance), resposta do adversário, SEE. 15 features novas → 67 totais. Spec em `docs/03-features/features-lookahead.md`. NÃO requer re-rotulagem. |
| 13 — Re-treino V3 | pendente | | Treinar DT + RF com 67 features. Mesmos grids e split. |
| 14 — Avaliação V2 vs V3 | pendente | | Comparar métricas V1→V2→V3. Meta: F1-ruim ≥ 0.50. |
| 15 — Notebook V3 | pendente | | Atualizar notebook com seção V3 e comparação final. |

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
- **Notebook V2 (etapa 11):** 46 células (31 code + 15 markdown). Novas seções: 6.6 Diagnóstico V1 (correlação point-biserial + Cohen's d para todas as 33 features, com limites de referência |r|=0.10 e d=0.20/0.50), 6.7 Comparação V1 vs V2 (tabela de deltas, barplot 4 barras DT/RF × V1/V2, overlay ROC/PR com modelos V1 carregados em paralelo, importância das 19 features táticas). Conclusão reescrita com tabela V1 vs V2, narrativa do ciclo diagnóstico→melhoria, limitações e trabalhos futuros atualizados. Execução ~34s com VERSION=2, RERUN_PIPELINE=False.
