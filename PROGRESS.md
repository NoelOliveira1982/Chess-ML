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
| 07 — Notebook final | concluída | 2026-04-06 | 28 células (13 code + 15 markdown), 8 seções: intro, dados, rotulagem, features, treino, avaliação, exemplos interpretados, conclusões. Execução OK (~11s). |

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

## Decisões tomadas durante o desenvolvimento

_(registrar aqui qualquer desvio da documentação original e justificativa)_

- **Desbalanceamento de classes (etapa 03):** ratio bom:ruim = 5.39:1. Será tratado na modelagem com class_weight='balanced' ou técnicas de reamostragem (SMOTE/undersampling). A proporção está dentro do esperado para jogadores 1400–1700 Lichess — a maioria dos lances não é um erro grave.
- **Multiprocessing na rotulagem:** usados 6 workers paralelos com caching de avaliações consecutivas para reduzir tempo (~174 min vs ~5h estimadas em single-thread).
- **Features extraídas (etapa 04):** 33 features em 7 grupos (material 11, mobilidade 3, segurança do rei 4, estrutura de peões 4, controle do centro 3, características do lance 6, contexto 2). Todas numéricas/inteiras, sem nulos. Extração levou ~4s com 6 workers.
- **Treino dos modelos (etapa 05):** Split estratificado 70/15/15 (76,546 / 16,350 / 16,394). Decision Tree: best = gini, depth=7, min_leaf=20 (CV-F1=0.3253). Random Forest: best = 200 trees, depth=10, min_leaf=5 (CV-F1=0.3499). Ambos com class_weight="balanced". RF ganhou em accuracy (0.68 vs 0.62), F1-ruim (0.35 vs 0.33) e AUC (0.68 vs 0.65). Top features: move_number, legal_moves_opponent, is_capture, legal_moves_player, material_diff. F1-ruim baixo é esperado dado o desbalanceamento 5.4:1 e a natureza difícil de prever erros apenas por features posicionais.
- **Avaliação e interpretação (etapa 06):** 13 artefatos de avaliação gerados. FPs típicos: lances quietos em posições complicadas; FNs: erros táticos sutis não capturados pelas features posicionais. Regras da DT: primeira decisão é se o lance é captura, depois mobilidade do adversário, depois fase da partida. Learning curves mostram gap treino-validação estável — melhoria viria de features táticas, não de mais dados.
- **Notebook final (etapa 07):** 28 células organizadas em 8 seções. Carrega dados pré-computados (CSVs + modelos joblib), gera visualizações inline (histogramas, heatmap de correlação, matrizes de confusão, feature importance, ROC/PR curves, árvore visual, análise de erros). Tempo de execução: ~11s.
