---
description: "Etapa 02: Filtragem e amostragem de partidas"
---

## Contexto

Ler `PROGRESS.md`. Pré-requisito: etapa 01 concluída (PGN baixado, streaming funcional).

## Documentação de referência

- `docs/02-dados/filtragem-e-amostragem.md`

## O que fazer

1. **Criar `src/filters.py`** com funções de filtro:
   - `rating_ok(game, lo=1400, hi=1700)` — ambos jogadores na faixa.
   - `time_control_ok(game)` — TimeControl em {"180+0","180+2","300+0","300+3","600+0","600+5"}.
   - `termination_ok(game)` — apenas "Normal".
2. **Criar script `src/filter_games.py`** que:
   - Faz streaming do PGN.
   - Aplica filtros.
   - Amostra com `random.seed(42)` e taxa configurável.
   - Extrai lances do meio-jogo (lance 8 a 40).
   - Salva partidas filtradas (FEN de cada posição + lance jogado) em `data/filtered_moves.csv`.
3. **Registrar estatísticas:**
   - Total de partidas lidas.
   - Total após cada filtro.
   - Total de lances extraídos.
   - Histograma de rating dos jogadores filtrados.
4. **Testar** com amostra pequena (primeiras 10k partidas do PGN).

## Ao finalizar

- Atualizar `PROGRESS.md`: etapa 02 → concluída, registrar contagens e artefato `data/filtered_moves.csv`.
- Reportar: "Filtragem concluída. X partidas, Y lances. Próxima etapa: 03 — Rotulagem."
