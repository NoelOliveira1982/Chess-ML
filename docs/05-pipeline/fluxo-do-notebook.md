# Fluxo do Notebook

## Estrutura geral

O notebook final (Jupyter / Colab) segue 8 seções sequenciais. Cada seção alterna **células Markdown** (explicação, justificativa) com **células de código** (implementação). Toda decisão relevante deve ter justificativa inline.

---

## Seção 1 — Introdução

**Conteúdo Markdown:**
- Descrição intuitiva do problema (jogadores 1200–1500, lances "suspeitos").
- Objetivo do classificador (bom/ruim).
- Público-alvo conceitual e real (banca da disciplina).
- Resumo da abordagem (dados do Lichess, rotulagem por Stockfish, classificação supervisionada).

**Código:** nenhum (ou apenas imports gerais e configuração de seeds).

```python
import random
import numpy as np
import pandas as pd
import chess
import chess.pgn
import chess.engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

---

## Seção 2 — Coleta e descrição dos dados

**Conteúdo Markdown:**
- Fonte: Lichess open database, licença CC0.
- Qual ficheiro PGN foi usado (mês, ano, tamanho).
- Filtros aplicados (rating, tempo, variante).

**Código:**
- Função de streaming do PGN (ou carga de CSV pré-processado).
- Contagem de partidas antes e depois dos filtros.
- Estatísticas iniciais: distribuição de ratings, controles de tempo, resultados.
- Histograma de `WhiteElo` e `BlackElo` no dataset filtrado.

**Referências:** [fonte-e-coleta.md](../02-dados/fonte-e-coleta.md), [filtragem-e-amostragem.md](../02-dados/filtragem-e-amostragem.md)

---

## Seção 3 — Rotulagem

**Conteúdo Markdown:**
- Explicação do conceito de centipawn loss.
- Definição dos limiares (bom >= -50cp, ruim <= -150cp, zona cinzenta descartada).
- Justificativa dos limiares para a faixa 1200–1500.

**Código:**
- Função de rotulagem usando Stockfish.
- Execução sobre o dataset filtrado (ou carga de resultados pré-computados).
- Histograma da distribuição de `delta_cp`.
- Contagem por classe (bom / ruim / descartado).
- Gráfico de barras com proporção final.

**Referências:** [rotulagem.md](../02-dados/rotulagem.md)

---

## Seção 4 — Engenharia de features

**Conteúdo Markdown:**
- Tabela-resumo de todas as features (nome, tipo, descrição).
- Justificativa das features escolhidas e por que one-hot de casas foi evitado.

**Código:**
- Funções de extração de features (material, mobilidade, segurança do rei, etc.).
- Aplicação sobre o dataset rotulado → DataFrame final.
- `.describe()` e correlação entre features (`sns.heatmap`).
- Verificação de valores ausentes ou outliers.

**Referências:** [engenharia-de-features.md](../03-features/engenharia-de-features.md)

---

## Seção 5 — Treino dos modelos

**Conteúdo Markdown:**
- Explicação do split treino/validação/teste.
- Justificativa de stratificação e tratamento de desbalanceamento.
- Breve descrição de cada modelo e por que foi escolhido.

**Código:**
- Split com `train_test_split` (estratificado).
- `GridSearchCV` para Árvore de Decisão.
- Treino do modelo secundário (RF / KNN / MLP).
- Print dos melhores hiperparâmetros.

**Referências:** [modelos-e-hiperparametros.md](../04-modelagem/modelos-e-hiperparametros.md)

---

## Seção 6 — Avaliação

**Conteúdo Markdown:**
- Métricas escolhidas e por que (com ênfase em recall da classe "ruim").
- Discussão: o que o modelo consegue e o que não consegue.

**Código:**
- `classification_report` para cada modelo.
- Matriz de confusão (gráfico).
- Tabela comparativa entre modelos.
- Feature importance (gráfico de barras horizontal).
- Curva de aprendizado (opcional).

**Referências:** [avaliacao-e-metricas.md](../04-modelagem/avaliacao-e-metricas.md)

---

## Seção 7 — Exemplos interpretados

**Conteúdo Markdown:**
- Seleção de 3–5 casos de acerto e 3–5 de erro.
- Para cada caso: FEN, lance, delta real, predição, features dominantes, explicação em linguagem de xadrez.

**Código:**
- Função que pega um exemplo do teste e mostra features + predição + regras da árvore.
- Visualização da árvore (parcial) com `plot_tree` ou `export_text`.
- Tradução das regras para conceitos de xadrez.

---

## Seção 8 — Conclusões e trabalhos futuros

**Conteúdo Markdown:**
- Resumo dos resultados (qual modelo foi melhor, qual métrica é mais relevante).
- Limitações:
  - Dataset limitado a uma faixa de rating e controle de tempo.
  - Features simples — não capturam padrões táticos complexos (cravadas, raio-x, etc.).
  - Dependência da qualidade da avaliação do Stockfish no depth escolhido.
- Trabalhos futuros:
  - Features mais sofisticadas (ameaças, cravadas, peças indefesas).
  - Foco em fases específicas (finais vs. meio-jogo).
  - Multiclasse (brilhante / bom / imprecisão / erro / blunder).
  - Acoplar o classificador a um agente que sugere lances.

**Código:** nenhum (ou save do modelo com `joblib`).

---

## Dica prática

Dividir o trabalho em dois momentos:

1. **Notebook de pré-processamento** (`01_data_prep.ipynb`): coleta, filtragem, rotulagem, extração de features → salva CSV.
2. **Notebook principal** (`02_modeling.ipynb`): carrega CSV, treina, avalia, interpreta.

Isso evita re-rodar o Stockfish toda vez que mudar algo na modelagem.
