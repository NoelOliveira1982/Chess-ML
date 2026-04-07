# Chess Move Classifier

Classificador supervisionado de qualidade de lances de xadrez (**bom** vs. **ruim**) para jogadores de nível intermediário (1200–1500 Chess.com / 1400–1700 Lichess).

Projeto acadêmico da disciplina de **Paradigmas de Aprendizagem de Máquina**.

## Resultados

O projeto evoluiu iterativamente em 4 versões, cada uma guiada por um ciclo de diagnóstico → melhoria:

| Versão | Abordagem | Modelo | Accuracy | F1 (ruim) | Recall (ruim) | Precision (ruim) | ROC-AUC |
|--------|-----------|--------|----------|-----------|---------------|-------------------|---------|
| V1 | 33 features posicionais | Decision Tree | 0.618 | 0.328 | 0.597 | 0.226 | 0.649 |
| V1 | | Random Forest | 0.683 | 0.353 | 0.552 | 0.259 | 0.684 |
| V2 | +19 features táticas (52) | Decision Tree | 0.630 | 0.343 | 0.618 | 0.238 | 0.674 |
| V2 | | Random Forest | 0.691 | 0.366 | 0.571 | 0.270 | 0.708 |
| V3 | +15 features look-ahead (67) | Decision Tree | 0.676 | 0.382 | 0.640 | 0.272 | 0.718 |
| V3 | | Random Forest | 0.785 | 0.431 | 0.522 | 0.368 | 0.768 |
| **V4** | **XGBoost + threshold tuning** | Decision Tree | 0.676 | 0.382 | 0.640 | 0.272 | 0.718 |
| **V4** | | Random Forest | 0.785 | 0.431 | 0.522 | 0.368 | 0.768 |
| **V4** | | **XGBoost** | **0.785** | **0.441** | **0.617** | **0.343** | **0.781** |

- **Dataset:** 109,290 lances de meio-jogo extraídos de 3,000 partidas do Lichess
- **67 features** em 15 grupos (posicionais, táticas, look-ahead)
- **Rotulagem:** Stockfish depth 15 — bom (δ ≥ −50 cp), ruim (δ ≤ −150 cp)
- **Melhor modelo:** XGBoost V4 — F1-ruim 0.441 (+8.8pp vs V1), AUC 0.781 (+9.7pp vs V1)

## Evolução do projeto

```
V1 (posicional)     → Diagnóstico: features descrevem posição, não o lance
V2 (+ tática)       → Diagnóstico: mesmas features para lances opostos na mesma posição
V3 (+ look-ahead)   → Diagnóstico: learning curves estáveis → teto no modelo, não nas features
V4 (+ XGBoost)      → Melhor modelo final (AUC 0.78, F1-ruim 0.44)
```

## Estrutura do projeto

```
Chess/
├── README.md
├── requirements.txt
├── PROGRESS.md
│
├── data/
│   ├── raw/                    # PGN bruto do Lichess (~272 MiB)
│   ├── filtered/               # Lances filtrados (136,620 linhas)
│   ├── labeled/                # Lances rotulados pelo Stockfish
│   ├── features/               # Features extraídas (V1: 33, V2: 52, V3/V4: 67)
│   ├── models/                 # Modelos V1 (.joblib)
│   ├── models_v2/              # Modelos V2
│   ├── models_v3/              # Modelos V3
│   ├── models_v4/              # Modelos V4 (DT + RF + XGBoost + thresholds)
│   ├── evaluation/             # Gráficos e análises V1
│   ├── evaluation_v2/          # Avaliação V2 + comparação V1 vs V2
│   ├── evaluation_v3/          # Avaliação V3 + comparação V1→V3
│   └── evaluation_v4/          # Avaliação V4 + comparação V1→V4
│
├── notebooks/
│   ├── chess_move_classifier.ipynb   # Notebook principal (pipeline completo)
│   ├── v1_posicional.ipynb           # Análise detalhada V1
│   ├── v2_tatica.ipynb               # Diagnóstico V1 + features táticas
│   ├── v3_lookahead.ipynb            # Diagnóstico V2 + features look-ahead
│   └── comparacao.ipynb              # Comparação V1→V4 (todos os modelos)
│
├── src/
│   ├── download_pgn.py         # Download do PGN do Lichess
│   ├── pgn_stream.py           # Streaming de PGN comprimido (.zst)
│   ├── filter_games.py         # Filtragem por rating, tempo, variante
│   ├── label_moves.py          # Rotulagem via Stockfish (delta CP)
│   ├── extract_features.py     # Extração de features (--v2, --v3 flags)
│   ├── train_models.py         # Treino DT + RF + XGBoost com GridSearchCV
│   ├── evaluate_models.py      # Avaliação V1
│   ├── evaluate_v2.py          # Avaliação V2 + comparação
│   ├── evaluate_v3.py          # Avaliação V3 + comparação
│   ├── evaluate_v4.py          # Avaliação V4 + comparação
│   ├── version_config.py       # Configuração de versões (paths, flags)
│   └── notebook_utils.py       # Funções auxiliares para notebooks
│
└── docs/
    ├── 01-visao-geral/         # Visão geral e decisões de projeto (DP-01 a DP-09)
    ├── 02-dados/               # Coleta, filtragem, rotulagem
    ├── 03-features/            # Features posicionais, táticas, look-ahead
    ├── 04-modelagem/           # Modelos, hiperparâmetros, spec V4 XGBoost
    ├── 05-pipeline/            # Estrutura do notebook
    └── 06-riscos-e-limitacoes/ # Diagnósticos V1, V2, V3
```

## Pré-requisitos

- Python 3.10+
- [Stockfish](https://stockfishchess.org/) (necessário apenas para re-executar a rotulagem)

```bash
# macOS
brew install stockfish

# Linux
sudo apt install stockfish
```

## Setup

```bash
# Clonar o repositório
git clone <url-do-repo>
cd Chess

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Instalar dependências
pip install -r requirements.txt
```

## Como usar

### Opção 1: Notebooks (recomendado)

| Notebook | Descrição |
|----------|-----------|
| `notebooks/comparacao.ipynb` | Comparação lado a lado V1→V4 com todos os modelos |
| `notebooks/chess_move_classifier.ipynb` | Pipeline completo autocontido |
| `notebooks/v1_posicional.ipynb` | Análise detalhada da V1 |
| `notebooks/v2_tatica.ipynb` | Diagnóstico V1 + features táticas V2 |
| `notebooks/v3_lookahead.ipynb` | Diagnóstico V2 + features look-ahead V3 |

O notebook `comparacao.ipynb` é o melhor ponto de entrada — mostra a evolução completa do projeto com tabelas, gráficos comparativos e análise de features.

### Opção 2: Scripts individuais

Cada etapa pode ser executada separadamente:

```bash
# 1. Download do PGN (~272 MiB)
python src/download_pgn.py

# 2. Filtragem e amostragem (Elo 1400–1700, blitz/rapid, seed=42)
python src/filter_games.py

# 3. Rotulagem via Stockfish (~174 min com 6 workers)
python src/label_moves.py -w 6

# 4. Extração de features
python src/extract_features.py -w 6           # V1: 33 features
python src/extract_features.py -w 6 --v2      # V2: 52 features (+táticas)
python src/extract_features.py -w 6 --v3      # V3: 67 features (+look-ahead)

# 5. Treino dos modelos
python src/train_models.py                     # V1: DT + RF
python src/train_models.py --v2                # V2: DT + RF
python src/train_models.py --v3                # V3: DT + RF
python src/train_models.py --v4                # V4: DT + RF + XGBoost

# 6. Avaliação
python src/evaluate_models.py                  # V1
python src/evaluate_v2.py                      # V2 + comparação V1 vs V2
python src/evaluate_v3.py                      # V3 + comparação V1→V3
python src/evaluate_v4.py                      # V4 + comparação V1→V4
```

## Pipeline

```
Lichess PGN (jan/2015, ~4.7M partidas)
    │
    ├─ Filtro: Elo 1400–1700, Blitz/Rapid, Normal
    ├─ Amostragem: 10%, seed=42 → 3,000 partidas
    ├─ Lances 8–40 (meio-jogo) → 136,620 lances
    │
    ├─ Stockfish depth 15 → delta (centipawn loss)
    ├─ bom (δ ≥ −50), ruim (δ ≤ −150), descartado (cinzenta)
    │  → 92,197 bom + 17,093 ruim = 109,290 lances
    │
    ├─ Features (evolução iterativa):
    │  V1: 33 features posicionais (7 grupos)
    │  V2: +19 features táticas (5 grupos) = 52 features
    │  V3: +15 features look-ahead (3 grupos) = 67 features
    │
    ├─ Split 70/15/15 estratificado
    │
    ├─ Modelos:
    │  V1–V3: Decision Tree + Random Forest (class_weight="balanced")
    │  V4: + XGBoost (scale_pos_weight=5.39) + threshold tuning
    │  GridSearchCV (5 folds, scoring=F1)
    │
    └─ Avaliação: confusion matrix, feature importance,
       ROC/PR curves, learning curves, análise de erros,
       comparação cross-version
```

## Features extraídas (67)

| Grupo | # | Versão | Exemplo |
|-------|---|--------|---------|
| Material (11) | 11 | V1 | `material_diff`, `white_queens` |
| Mobilidade (3) | 3 | V1 | `legal_moves_player`, `mobility_diff` |
| Segurança do rei (4) | 4 | V1 | `player_castled`, `king_pawn_shield` |
| Estrutura de peões (4) | 4 | V1 | `player_doubled_pawns` |
| Controle do centro (3) | 3 | V1 | `player_center_control` |
| Características do lance (6) | 6 | V1 | `is_capture`, `moved_piece` |
| Contexto (2) | 2 | V1 | `move_number`, `is_white` |
| Peças indefesas (5) | 5 | V2 | `hanging_count_player`, `hanging_value_opponent` |
| Capturas com ganho (4) | 4 | V2 | `good_captures_player` |
| Cravadas/alinhamentos (2) | 2 | V2 | `pins_player`, `pins_opponent` |
| Rei avançado (4) | 4 | V2 | `king_tropism_player`, `king_open_file_opponent` |
| Tensão (4) | 4 | V2 | `contested_squares`, `total_attacks_player` |
| Deltas pré/pós lance (8) | 8 | V3 | `delta_material`, `delta_mobility_player` |
| Resposta do adversário (4) | 4 | V3 | `opponent_best_capture_value`, `opponent_can_check` |
| SEE — Static Exchange Eval (3) | 3 | V3 | `see_of_move`, `worst_see_against_player`, `is_losing_capture` |

## Licença dos dados

Os dados do Lichess são disponibilizados sob licença [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (domínio público).
