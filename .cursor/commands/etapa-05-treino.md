---
description: "Etapa 05: Treino dos modelos"
---

## Contexto

Ler `PROGRESS.md`. Pré-requisito: etapa 04 concluída (`data/features.csv` existe).

## Documentação de referência

- `docs/04-modelagem/modelos-e-hiperparametros.md`

## O que fazer

1. **Carregar dados** de `data/features.csv`. Separar X e y. Converter label para int.
2. **Split:** 70/15/15 com `stratify=y`, `random_state=42`.
3. **Decision Tree:**
   - `GridSearchCV` com grid documentado (max_depth, min_samples_leaf, criterion).
   - `cv=5`, `scoring="f1"`, `class_weight="balanced"`.
   - Registrar `best_params_` e `best_score_`.
4. **Modelo secundário** (escolher um):
   - Random Forest, KNN ou MLP.
   - Se KNN/MLP: Pipeline com StandardScaler.
   - GridSearchCV com grid apropriado.
5. **Salvar modelos** com `joblib.dump()` em `models/`.
6. **Não avaliar no teste ainda** — isso é etapa 06.

## Ao finalizar

- Atualizar `PROGRESS.md`: etapa 05 → concluída, registrar best_params de cada modelo.
- Reportar: "Treino concluído. DT best F1=X, Modelo2 best F1=Y. Próxima etapa: 06 — Avaliação."
