---
description: "Etapa 06: Avaliação, métricas e interpretabilidade"
---

## Contexto

Ler `PROGRESS.md`. Pré-requisito: etapa 05 concluída (modelos treinados em `models/`).

## Documentação de referência

- `docs/04-modelagem/avaliacao-e-metricas.md`

## O que fazer

1. **Carregar modelos** de `models/` e dados de teste (não usados no treino).
2. **Métricas por modelo:**
   - `classification_report` com target_names=["bom", "ruim"].
   - `ConfusionMatrixDisplay`.
   - ROC-AUC se aplicável.
3. **Tabela comparativa** entre Decision Tree e modelo secundário.
4. **Feature importances:**
   - Bar plot horizontal das top 15 features (DT e RF se aplicável).
5. **Interpretabilidade da árvore:**
   - `export_text` com max_depth=5.
   - Traduzir top 3–5 regras para linguagem de xadrez.
6. **Análise de erros:**
   - Selecionar 5 falsos positivos e 5 falsos negativos.
   - Para cada: FEN, lance, delta real, predição, features dominantes, explicação.
7. **Curva de aprendizado** (opcional mas recomendado).
8. **Salvar gráficos** em `outputs/`.

## Ao finalizar

- Atualizar `PROGRESS.md`: etapa 06 → concluída, registrar métricas principais.
- Reportar: "Avaliação concluída. DT accuracy=X, F1_ruim=Y. Próxima etapa: 07 — Notebook final."
