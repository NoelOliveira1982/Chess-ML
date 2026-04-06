# Rotulagem dos Lances

## Método: delta-score via Stockfish

Para cada lance do dataset, o rótulo é gerado comparando a **avaliação do Stockfish antes e depois do lance jogado**.

### Definição formal

```
eval_antes  = avaliação do Stockfish na posição ANTES do lance (do ponto de vista do jogador)
eval_depois = avaliação do Stockfish na posição DEPOIS do lance (do ponto de vista do jogador)

delta = eval_depois - eval_antes
```

- `delta ≈ 0`: lance manteve a avaliação → lance razoável.
- `delta << 0`: lance piorou muito a posição → lance ruim.
- `delta > 0`: lance melhorou a posição (adversário errou antes, ou o lance explora algo) → lance bom.

### Atenção à perspectiva

O Stockfish sempre avalia do ponto de vista das brancas. Para normalizar:

```python
def normalize_score(score, is_white_turn):
    """Converte score para perspectiva do jogador da vez."""
    cp = score.relative.score(mate_score=10000)
    return cp
```

O método `score.relative` do `python-chess` já faz essa normalização automaticamente.

## Limiares de classificação

| Classe | Condição (centipawns) | Significado |
|--------|-----------------------|-------------|
| **Bom** | `delta >= -50` | Lance não piora significativamente (≤ 0.5 peão de perda) |
| **Descartado** | `-150 < delta < -50` | Zona cinzenta — lance ambíguo |
| **Ruim** | `delta <= -150` | Lance piora a posição em ≥ 1.5 peão |

### Justificativa dos limiares

- **-50 cp:** na faixa 1200–1500, uma perda de até meio peão é considerada "imprecisão leve" — jogadores desse nível frequentemente oscilam nessa margem. Classificar esses lances como "ruim" seria excessivamente severo.
- **-150 cp:** uma perda de 1.5 peão ou mais corresponde, na prática, a um erro grave (peça pendurada, tática não vista, posição desmoronando). Para esse nível de jogo, é um limiar claro.
- **Zona cinzenta (-50 a -150):** lances que perdem entre 0.5 e 1.5 peão são imprecisões intermediárias. Classificá-los como "bom" ou "ruim" seria ruidoso. Descartá-los **melhora a qualidade dos rótulos** ao custo de reduzir levemente o dataset (ver [DP-05](../01-visao-geral/decisoes-de-projeto.md)).

### Esses limiares podem ser ajustados

A escolha é documentada e pode ser revisada após inspeção da distribuição. Se a classe "ruim" ficar com poucos exemplos, relaxar para -100 cp. Se "bom" dominar excessivamente, apertar para -30 cp. Documentar a distribuição antes e depois.

## Implementação com python-chess + Stockfish

### Instalação do Stockfish

```bash
# macOS (Apple Silicon)
brew install stockfish

# Verificar instalação
stockfish --help
```

### Código de rotulagem

```python
import chess
import chess.engine
import chess.pgn

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # ajustar conforme sistema
DEPTH = 15

def label_moves(game):
    """Gera (fen, move_uci, delta, label) para cada lance do meio-jogo."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    board = game.board()
    results = []

    for move_number, node in enumerate(game.mainline()):
        move = node.move
        if move_number < 7 or move_number > 39:
            board.push(move)
            continue

        # Avaliação antes do lance
        info_before = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
        score_before = info_before["score"].relative.score(mate_score=10000)

        # Jogar o lance
        board.push(move)

        # Avaliação depois do lance (perspectiva invertida, pois mudou o turno)
        info_after = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
        score_after = info_after["score"].relative.score(mate_score=10000)

        # Delta do ponto de vista de quem jogou
        delta = -score_after - score_before

        # Rotulagem
        if delta >= -50:
            label = "bom"
        elif delta <= -150:
            label = "ruim"
        else:
            label = None  # zona cinzenta, descartar

        if label is not None:
            results.append({
                "fen": board.peek().uci(),  # posição antes (reconstruível)
                "move_uci": move.uci(),
                "delta_cp": delta,
                "label": label,
            })

    engine.quit()
    return results
```

### Performance esperada no M4

| Depth | Posições/segundo (aprox.) | Tempo para 30k posições |
|-------|--------------------------|------------------------|
| 10 | ~30–40 | ~12–17 min |
| 15 | ~10–20 | ~25–50 min |
| 18 | ~5–10 | ~50–100 min |

**Recomendação:** usar depth 15 como padrão. É suficiente para detectar táticas de 3–4 lances e erros posicionais claros.

## Alternativa descartada: rotulagem por resultado da partida

Associar "bom/ruim" com base no resultado final (vitória/derrota) **não é adequado**:

- O resultado depende de todos os lances subsequentes, não apenas do lance avaliado.
- Um lance excelente pode ocorrer numa partida perdida (e vice-versa).
- Introduziria **ruído massivo** nos rótulos.

## Alternativa descartada: rotulagem por taxa de vitória estatística

Medir "bom/ruim" pela taxa de vitória de quem joga aquele lance naquela posição exigiria:

- Milhares de partidas **na mesma posição exata** para ter significância estatística.
- Volume de dados impraticável para o escopo do projeto.

## Checklist

- [ ] Stockfish instalado e caminho correto configurado.
- [ ] Depth definido e documentado.
- [ ] Distribuição de delta_cp plotada (histograma).
- [ ] Contagem de lances por classe (bom / ruim / descartado).
- [ ] Se desbalanceado, documentar proporção e estratégia de mitigação.
