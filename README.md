# Chess Move Classifier

Classificador supervisionado de qualidade de lances de xadrez (**bom** vs. **ruim**) para jogadores de nível intermediário (1200–1500 Chess.com / 1400–1700 Lichess).

Projeto acadêmico da disciplina de **Paradigmas de Aprendizagem de Máquina**.

## Resultados

| Modelo | Accuracy | F1 (ruim) | Recall (ruim) | Precision (ruim) | ROC-AUC |
|--------|----------|-----------|---------------|-------------------|---------|
| Decision Tree | 0.6176 | 0.3280 | 0.5967 | 0.2262 | 0.6487 |
| **Random Forest** | **0.6827** | **0.3525** | **0.5523** | **0.2589** | **0.6837** |

- **Dataset:** 109,290 lances de meio-jogo extraídos de 3,000 partidas do Lichess
- **33 features posicionais** (material, mobilidade, segurança do rei, estrutura de peões, controle do centro, características do lance, contexto)
- **Rotulagem:** Stockfish depth 15 — bom (δ ≥ −50 cp), ruim (δ ≤ −150 cp)

## Estrutura do projeto

```
Chess/
├── README.md
├── requirements.txt
├── PROGRESS.md                 # Tracking de progresso do desenvolvimento
│
├── data/
│   ├── raw/                    # PGN bruto do Lichess (~272 MiB)
│   ├── filtered/               # Lances filtrados (136,620 linhas)
│   ├── labeled/                # Lances rotulados pelo Stockfish
│   ├── features/               # Features extraídas (109,290 × 33)
│   ├── models/                 # Modelos treinados (.joblib) + resultados
│   └── evaluation/             # Gráficos e análises de avaliação
│
├── notebooks/
│   └── chess_move_classifier.ipynb   # Notebook final (entrega)
│
├── src/
│   ├── download_pgn.py         # Download do PGN do Lichess
│   ├── pgn_stream.py           # Streaming de PGN comprimido (.zst)
│   ├── filter_games.py         # Filtragem por rating, tempo, variante
│   ├── label_moves.py          # Rotulagem via Stockfish (delta CP)
│   ├── extract_features.py     # Extração das 33 features posicionais
│   ├── train_models.py         # Treino DT + RF com GridSearchCV
│   └── evaluate_models.py      # Avaliação completa (plots, erros, regras)
│
└── docs/                       # Documentação de metodologia e decisões
    ├── 01-visao-geral/
    ├── 02-dados/
    ├── 03-features/
    ├── 04-modelagem/
    └── 05-pipeline/
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

### Opção 1: Notebook (recomendado)

Abrir `notebooks/chess_move_classifier.ipynb` — contém o pipeline completo com dados pré-computados. Todas as visualizações e análises são geradas inline.

Para re-executar o pipeline desde o início, alterar `RERUN_PIPELINE = True` na segunda célula (demora ~3 horas por causa da rotulagem Stockfish).

### Opção 2: Scripts individuais

Cada etapa pode ser executada separadamente:

```bash
# 1. Download do PGN (~272 MiB)
python src/download_pgn.py

# 2. Filtragem e amostragem (Elo 1400–1700, blitz/rapid, seed=42)
python src/filter_games.py

# 3. Rotulagem via Stockfish (~174 min com 6 workers)
python src/label_moves.py -w 6

# 4. Extração de features (~4s com 6 workers)
python src/extract_features.py -w 6

# 5. Treino dos modelos (GridSearchCV)
python src/train_models.py

# 6. Avaliação completa (plots + análise de erros)
python src/evaluate_models.py
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
    ├─ 33 features posicionais (7 grupos)
    │
    ├─ Split 70/15/15 estratificado
    ├─ Decision Tree + Random Forest (class_weight="balanced")
    ├─ GridSearchCV (5 folds, scoring=F1)
    │
    └─ Avaliação: confusion matrix, feature importance,
       ROC/PR curves, learning curves, análise de erros
```

## Features extraídas

| Grupo | Features | Exemplo |
|-------|----------|---------|
| Material (11) | Contagem por tipo/cor + diferença | `material_diff`, `white_queens` |
| Mobilidade (3) | Lances legais + diferença | `legal_moves_player`, `mobility_diff` |
| Segurança do rei (4) | Roque, escudo de peões | `player_castled`, `king_pawn_shield` |
| Estrutura de peões (4) | Dobrados, isolados, passados | `player_doubled_pawns` |
| Controle do centro (3) | Ataques e ocupação de d4/d5/e4/e5 | `player_center_control` |
| Características do lance (6) | Captura, xeque, promoção, peça | `is_capture`, `moved_piece` |
| Contexto (2) | Fase do jogo, cor | `move_number`, `is_white` |

## Licença dos dados

Os dados do Lichess são disponibilizados sob licença [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (domínio público).
