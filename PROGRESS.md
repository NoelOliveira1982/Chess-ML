# Progresso do Projeto

Arquivo de tracking para desenvolvimento iterativo. Atualizado automaticamente pelos commands de cada etapa.

## Status das etapas

| Etapa | Status | Última atualização | Observações |
|-------|--------|--------------------|-------------|
| 00 — Setup do ambiente | concluída | 2026-04-06 | Todas as dependências instaladas, Stockfish 18 OK |
| 01 — Coleta de dados | concluída | 2026-04-06 | PGN: `lichess_db_standard_rated_2015-01.pgn.zst` (~272 MiB); scripts `src/download_pgn.py`, `src/pgn_stream.py` |
| 02 — Filtragem e amostragem | concluída | 2026-04-06 | 452,929 scanned → 3,000 aceites (seed=42, rate=0.10); 136,620 lances mid-game; Elo médio 1561 |
| 03 — Rotulagem Stockfish | concluída | 2026-04-06 | Depth 15, 6 workers, ~174 min. 109,290 rotulados (92,197 bom / 17,093 ruim), 27,330 descartados. Ratio bom:ruim = 5.39:1 |
| 04 — Engenharia de features | pendente | — | — |
| 05 — Treino dos modelos | pendente | — | — |
| 06 — Avaliação e interpretação | pendente | — | — |
| 07 — Notebook final | pendente | — | — |

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

## Decisões tomadas durante o desenvolvimento

_(registrar aqui qualquer desvio da documentação original e justificativa)_

- **Desbalanceamento de classes (etapa 03):** ratio bom:ruim = 5.39:1. Será tratado na modelagem com class_weight='balanced' ou técnicas de reamostragem (SMOTE/undersampling). A proporção está dentro do esperado para jogadores 1400–1700 Lichess — a maioria dos lances não é um erro grave.
- **Multiprocessing na rotulagem:** usados 6 workers paralelos com caching de avaliações consecutivas para reduzir tempo (~174 min vs ~5h estimadas em single-thread).
