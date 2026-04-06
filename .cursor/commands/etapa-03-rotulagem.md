---
description: "Etapa 03: Rotulagem de lances via Stockfish"
---

## Contexto

Ler `PROGRESS.md`. Pré-requisito: etapa 02 concluída (`data/filtered_moves.csv` existe).

## Documentação de referência

- `docs/02-dados/rotulagem.md`

## O que fazer

1. **Criar `src/labeling.py`** com:
   - Função que recebe FEN + lance, avalia com Stockfish depth 15, retorna delta_cp.
   - Descarte de posições com |eval| > 500cp.
   - Classificação: bom (delta >= -50cp), ruim (delta <= -150cp), zona cinzenta descartada.
2. **Criar script `src/label_moves.py`** que:
   - Lê `data/filtered_moves.csv`.
   - Rotula cada lance com Stockfish.
   - Salva `data/labeled_moves.csv` (fen, move_uci, delta_cp, label).
   - Suporta checkpoint: se o script for interrompido, retomar de onde parou.
3. **Performance:** medir tempo por posição, estimar tempo total. Ajustar depth se necessário.
4. **Plotar:**
   - Histograma de delta_cp.
   - Contagem por classe (bom / ruim / descartado).
   - Proporção final.

## Atenção

- Esta etapa pode demorar (25-50 min para 30k posições). Implementar checkpoint obrigatoriamente.
- Sempre fechar engine com `engine.quit()`.

## Ao finalizar

- Atualizar `PROGRESS.md`: etapa 03 → concluída, registrar contagem por classe e tempo total.
- Reportar: "Rotulagem concluída. X bom, Y ruim. Próxima etapa: 04 — Features."
