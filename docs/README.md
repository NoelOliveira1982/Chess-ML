# Classificador de Qualidade de Lances de Xadrez

Projeto de Machine Learning para a disciplina de Paradigmas de Aprendizagem de Máquina.

**Objetivo:** treinar um classificador supervisionado que, dada uma posição de xadrez e um lance candidato, classifique o lance como **bom** ou **ruim** para jogadores na faixa de rating 1200–1500.

---

## Índice da documentação

| Seção | Conteúdo |
|-------|----------|
| [01 — Visão geral](01-visao-geral/objetivo-e-escopo.md) | Problema, objetivo principal e secundários, escopo |
| [01 — Decisões de projeto](01-visao-geral/decisoes-de-projeto.md) | Log de decisões tomadas com justificativas |
| [02 — Fonte e coleta de dados](02-dados/fonte-e-coleta.md) | Lichess open DB, métodos de download, streaming de PGN |
| [02 — Filtragem e amostragem](02-dados/filtragem-e-amostragem.md) | Critérios de rating, controle de tempo, fase do jogo |
| [02 — Rotulagem](02-dados/rotulagem.md) | Avaliação Stockfish, limiares de delta-score, zona cinzenta |
| [03 — Engenharia de features](03-features/engenharia-de-features.md) | Lista completa de features e como extraí-las |
| [04 — Modelos e hiperparâmetros](04-modelagem/modelos-e-hiperparametros.md) | Decision Tree, modelo secundário, tuning |
| [04 — Avaliação e métricas](04-modelagem/avaliacao-e-metricas.md) | Métricas, split, interpretabilidade, análise de erros |
| [05 — Fluxo do notebook](05-pipeline/fluxo-do-notebook.md) | Estrutura das seções do notebook final |
| [06 — Riscos e limitações](06-riscos-e-limitacoes/riscos.md) | Riscos, mitigações, trabalho futuro |

---

## Stack técnica

| Componente | Ferramenta |
|------------|------------|
| Linguagem | Python 3.10+ |
| Ambiente | Jupyter Notebook / Google Colab |
| Xadrez | `python-chess`, Stockfish (via `chess.engine`) |
| Dados | `zstandard`, `pandas`, `numpy` |
| ML | `scikit-learn` |
| Visualização | `matplotlib`, `seaborn` |

---

## Como navegar

Cada pasta numerada corresponde a uma fase do projeto. Os arquivos dentro de cada pasta são curtos e autocontidos. Comece por [01 — Visão geral](01-visao-geral/objetivo-e-escopo.md) para entender o problema, depois siga a numeração.
