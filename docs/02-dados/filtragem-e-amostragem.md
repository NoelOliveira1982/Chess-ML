# Filtragem e Amostragem

## Visão geral do pipeline de filtragem

```
PGN bruto (~milhões de partidas)
  │
  ├─ Filtro de rating ──────────── ambos jogadores entre 1400–1700 (Lichess)
  ├─ Filtro de tempo ───────────── blitz/rapid (3–10 min)
  ├─ Filtro de término ─────────── apenas partidas terminadas normalmente
  │
  ▼
Partidas filtradas (~milhares)
  │
  ├─ Extração de lances ────────── lance 8 a 40 (meio-jogo)
  ├─ Remoção de posições decididas  eval > |5.0| peões
  │
  ▼
Lances candidatos (~dezenas de milhares)
  │
  ├─ Rotulagem via Stockfish ───── bom / ruim / descartado
  │
  ▼
Dataset final (~5k–30k lances rotulados)
```

## Critérios de filtragem

### Rating

- **Ambos os jogadores** (`WhiteElo` e `BlackElo`) devem estar entre **1400 e 1700** no Lichess.
- Justificativa: partidas onde um jogador é muito mais forte distorcem a qualidade dos lances (ver [DP-03](../01-visao-geral/decisoes-de-projeto.md)).

```python
def rating_ok(game, lo=1400, hi=1700):
    try:
        w = int(game.headers["WhiteElo"])
        b = int(game.headers["BlackElo"])
        return lo <= w <= hi and lo <= b <= hi
    except (KeyError, ValueError):
        return False
```

### Controle de tempo

Aceitar partidas cujo `TimeControl` corresponda a **blitz** (3–5 min) ou **rapid curto** (10 min):

| TimeControl | Formato |
|-------------|---------|
| `180+0` | Blitz 3 min |
| `180+2` | Blitz 3+2 |
| `300+0` | Blitz 5 min |
| `300+3` | Blitz 5+3 |
| `600+0` | Rapid 10 min |
| `600+5` | Rapid 10+5 |

```python
ALLOWED_TC = {"180+0", "180+2", "300+0", "300+3", "600+0", "600+5"}

def time_control_ok(game):
    return game.headers.get("TimeControl", "") in ALLOWED_TC
```

### Término da partida

Aceitar apenas `Termination: Normal` (checkmate ou resignação). Excluir partidas decididas por tempo, abandono ou outros motivos — nesses casos a qualidade dos últimos lances é discutível.

### Variante

Aceitar apenas `Standard` (sem Chess960, Crazyhouse, etc.).

## Extração de lances: fase do jogo

Do total de lances de cada partida, extrair apenas o **meio-jogo**:

- **Início:** lance 8 (após a abertura típica de 7–8 lances).
- **Fim:** lance 40 (antes de finais muito simplificados).

Justificativa:
- Lances de abertura são "teóricos" e não refletem a capacidade do jogador de avaliar a posição.
- Finais tardios tendem a ter poucas peças e avaliações extremas, enviesando os rótulos.

## Remoção de posições já decididas

Antes de rotular, descartar posições onde a avaliação do Stockfish já mostra vantagem esmagadora:

- Excluir se `|eval| > 500` centipawns (5 peões) antes do lance.
- Justificativa: nesses casos, quase qualquer lance é "bom" para o lado vencedor e "ruim" para o perdedor — o rótulo não carrega informação útil.

## Amostragem

### Tamanho-alvo

- **1000–3000 partidas** filtradas.
- De cada partida, ~15–25 lances de meio-jogo (lance 8–40, alternando brancas e pretas).
- Total estimado: **15k–60k lances brutos** → após rotulagem e remoção da zona cinzenta → **~5k–30k lances no dataset final**.

### Estratégia de amostragem

- Se o ficheiro mensal tem muitas partidas na faixa, fazer **amostragem aleatória** das primeiras N partidas que passam nos filtros (`random.random() < p` durante o streaming).
- Registrar a seed do random para reprodutibilidade.

```python
import random

random.seed(42)
SAMPLE_RATE = 0.05  # ajustar conforme volume do ficheiro

def should_sample():
    return random.random() < SAMPLE_RATE
```

## Checklist de reprodutibilidade

- [ ] Nome exato do ficheiro PGN usado (mês/ano).
- [ ] Seed do random.
- [ ] Valores exatos dos filtros (rating, tempo, fase).
- [ ] Número de partidas antes e depois de cada filtro.
- [ ] Distribuição de ratings no dataset filtrado (histograma).
