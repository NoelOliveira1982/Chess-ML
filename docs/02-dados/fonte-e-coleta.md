# Fonte e Coleta de Dados

## Fonte escolhida

**Lichess open database** — <https://database.lichess.org/>

- Licença: **CC0** (domínio público, uso livre para fins acadêmicos e comerciais).
- Formato: ficheiros `.pgn.zst` (PGN comprimido com Zstandard), um por mês.
- Cobertura: todas as partidas rated de xadrez standard desde 2013.

## Opção A — Download de um ficheiro mensal antigo (recomendado)

Ficheiros de meses antigos (2013–2015) são **muito menores** (~1–5 GB comprimidos, vs. ~20 GB dos meses recentes). Para o escopo do projeto, um único mês antigo já fornece centenas de milhares de partidas na faixa de rating alvo.

### Passo a passo

1. Acessar <https://database.lichess.org/standard/list.txt> para ver a lista de ficheiros disponíveis.
2. Escolher um ficheiro antigo (ex.: `lichess_db_standard_rated_2015-01.pgn.zst`).
3. Download via terminal:

```bash
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2015-01.pgn.zst
```

4. **Não descomprimir o ficheiro inteiro.** Usar streaming em Python (ver seção abaixo).

### Processamento em streaming (sem carregar tudo na RAM)

```python
import zstandard as zstd
import chess.pgn
import io

def stream_games(filepath):
    """Itera sobre jogos de um .pgn.zst sem carregar tudo na memória."""
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            yield game
```

**Consumo de memória:** constante, independente do tamanho do ficheiro. Viável em 16 GB de RAM.

## Opção B — API do Lichess (para amostras menores)

A API REST do Lichess permite exportar partidas de um usuário específico ou consultar o Explorer de aberturas. Não tem filtro nativo por faixa de rating no endpoint de download, então seria necessário:

1. Identificar usuários com rating na faixa alvo.
2. Exportar suas partidas via `GET /api/games/user/{username}`.

Essa abordagem é **menos prática** para obter volume, mas pode complementar ou servir para validação pontual.

### Exemplo de chamada

```bash
curl -H "Accept: application/x-chess-pgn" \
  "https://lichess.org/api/games/user/USERNAME?max=100&rated=true&perfType=blitz"
```

## Opção C — Lichess Explorer API (estatísticas de abertura)

O endpoint `https://explorer.lichess.ovh/lichess` permite consultar estatísticas de aberturas filtradas por rating. Útil como dado complementar, mas **não retorna partidas individuais com lances** — apenas agregados (vitória/empate/derrota por abertura).

## Decisão

Usar **Opção A** como método principal. Documentar versão exata do ficheiro usado (nome, mês, tamanho) para reprodutibilidade.

## Dependências necessárias

```bash
pip install python-chess zstandard pandas numpy
```

## Campos disponíveis nos headers PGN do Lichess

| Campo | Exemplo | Uso no projeto |
|-------|---------|----------------|
| `WhiteElo` | 1523 | Filtragem por rating |
| `BlackElo` | 1487 | Filtragem por rating |
| `TimeControl` | 300+3 | Filtragem por controle de tempo |
| `ECO` | B20 | Classificação de abertura |
| `Opening` | Sicilian Defense | Nome da abertura |
| `Result` | 1-0 | Metadado complementar |
| `Termination` | Normal | Filtrar por tipo de fim de jogo |
| `UTCDate` | 2015.01.15 | Referência temporal |
