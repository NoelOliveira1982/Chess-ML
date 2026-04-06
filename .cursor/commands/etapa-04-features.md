---
description: "Etapa 04: Engenharia de features"
---

## Contexto

Ler `PROGRESS.md`. Pré-requisito: etapa 03 concluída (`data/labeled_moves.csv` existe).

## Documentação de referência

- `docs/03-features/engenharia-de-features.md`

## O que fazer

1. **Criar `src/features.py`** com funções de extração por grupo:
   - `material_features(board)` → dict com 11 features.
   - `mobility_features(board)` → dict com 3 features.
   - `king_safety_features(board)` → dict com 4 features.
   - `pawn_structure_features(board)` → dict com 4 features.
   - `center_control_features(board)` → dict com 3 features.
   - `move_features(board, move)` → dict com 6 features.
   - `context_features(board, move_number)` → dict com 2 features.
   - `extract_all_features(board, move, move_number)` → dict completo (~33 features).
2. **Criar script `src/build_features.py`** que:
   - Lê `data/labeled_moves.csv`.
   - Reconstrói board a partir de FEN.
   - Extrai features de cada posição.
   - Gera DataFrame com features + label.
   - Salva `data/features.csv`.
3. **Validação:**
   - `.describe()` para verificar ranges.
   - Verificar NaN ou valores inconsistentes.
   - Heatmap de correlação entre features.
   - Imprimir shape final.

## Ao finalizar

- Atualizar `PROGRESS.md`: etapa 04 → concluída, registrar nº de features e shape do dataset.
- Reportar: "Features extraídas. Shape: (N, 33+1). Próxima etapa: 05 — Treino."
