---
description: "Etapa 01: Coleta de dados (download PGN do Lichess)"
---

## Contexto

Ler `PROGRESS.md`. Pré-requisito: etapa 00 concluída.

## Documentação de referência

- `docs/02-dados/fonte-e-coleta.md`

## O que fazer

1. **Escolher ficheiro PGN** do Lichess open database:
   - Preferir mês antigo e menor (2014 ou 2015, ~1-3 GB comprimido).
   - Documentar qual ficheiro exato em PROGRESS.md.
2. **Criar script de download** em `src/download_pgn.py`:
   - Usar `requests` ou `urllib` para baixar o .pgn.zst.
   - Mostrar progresso do download.
   - Salvar em `data/raw/`.
3. **Criar função de streaming** em `src/pgn_stream.py`:
   - Função `stream_games(filepath)` que usa `zstandard` + `chess.pgn.read_game()`.
   - Yield de um jogo por vez, memória constante.
4. **Testar:** fazer streaming das primeiras 100 partidas, imprimir headers (WhiteElo, BlackElo, TimeControl).

## Ao finalizar

- Atualizar `PROGRESS.md`: etapa 01 → concluída, registrar nome do PGN e scripts criados.
- Reportar: "Coleta configurada. Próxima etapa: 02 — Filtragem e amostragem."
