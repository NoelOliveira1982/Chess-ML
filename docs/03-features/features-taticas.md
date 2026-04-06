# Features Táticas — Upgrade V1 → V2

## Motivação

O diagnóstico da V1 ([diagnostico-v1.md](../06-riscos-e-limitacoes/diagnostico-v1.md)) mostrou que as 33 features posicionais têm poder preditivo negligível (melhor Cohen's d = 0.24, melhor correlação = 0.084). A causa: as features descrevem o **cenário** da posição mas não as **ameaças táticas** que causam 80%+ dos erros na faixa 1400–1700 Lichess.

Este documento especifica as features táticas a adicionar, a implementação, e o pipeline de upgrade.

---

## Grupo 8 — Ameaças e peças indefesas

Features que detectam peças **atacadas sem defesa adequada** — a causa nº 1 de erros nesta faixa de rating.

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `hanging_pieces_player` | int | Nº de peças do jogador atacadas pelo adversário e não defendidas por peças do jogador |
| `hanging_pieces_opponent` | int | Nº de peças do adversário atacadas pelo jogador e não defendidas |
| `hanging_value_player` | int | Soma do valor material das peças penduradas do jogador |
| `hanging_value_opponent` | int | Soma do valor material das peças penduradas do adversário |
| `min_attacker_vs_piece_player` | int | Para a peça do jogador mais vulnerável: valor do atacante mais barato − valor da peça (negativo = troca desfavorável possível) |

### Implementação

```python
def _hanging_features(board: chess.Board) -> dict:
    """Detect undefended pieces for both sides."""
    turn = board.turn
    opp = not turn

    def hanging(board, color, attacker_color):
        count = 0
        value = 0
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(pt, color):
                attackers = board.attackers(attacker_color, sq)
                defenders = board.attackers(color, sq)
                if attackers and not defenders:
                    count += 1
                    value += PIECE_VALUES.get(pt, 0)
        return count, value

    h_player, hv_player = hanging(board, turn, opp)
    h_opp, hv_opp = hanging(board, opp, turn)

    return {
        "hanging_pieces_player": h_player,
        "hanging_pieces_opponent": h_opp,
        "hanging_value_player": hv_player,
        "hanging_value_opponent": hv_opp,
    }
```

### Custo computacional

- Usa `board.attackers()` — O(1) por casa graças aos bitboards internos do `python-chess`.
- Total: ~64 chamadas por posição → desprezível.

---

## Grupo 9 — Ameaças de captura com ganho

Features que detectam se o adversário pode capturar peças do jogador com **ganho material** (atacante vale menos que o alvo).

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `threats_against_player` | int | Nº de peças do jogador atacadas por peça de menor valor |
| `threats_against_opponent` | int | Nº de peças do adversário atacadas por peça de menor valor do jogador |
| `max_threat_value_player` | int | Maior ganho material possível numa captura contra o jogador |
| `max_threat_value_opponent` | int | Maior ganho material possível numa captura pelo jogador |

### Implementação

```python
def _threat_features(board: chess.Board) -> dict:
    """Detect captures where attacker is worth less than the target."""
    turn = board.turn
    opp = not turn

    def threats(board, target_color, attacker_color):
        count = 0
        max_gain = 0
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(pt, target_color):
                target_val = PIECE_VALUES[pt]
                for att_sq in board.attackers(attacker_color, sq):
                    att_piece = board.piece_at(att_sq)
                    if att_piece:
                        att_val = PIECE_VALUES.get(att_piece.piece_type, 0)
                        if att_val < target_val:
                            count += 1
                            gain = target_val - att_val
                            max_gain = max(max_gain, gain)
                            break  # one threat per target is enough
        return count, max_gain

    t_player, mt_player = threats(board, turn, opp)
    t_opp, mt_opp = threats(board, opp, turn)

    return {
        "threats_against_player": t_player,
        "threats_against_opponent": t_opp,
        "max_threat_value_player": mt_player,
        "max_threat_value_opponent": mt_opp,
    }
```

---

## Grupo 10 — Cravadas (pins)

Uma peça está **cravada** quando não pode mover sem expor uma peça de maior valor atrás dela (na mesma linha/diagonal) a um ataque. A detecção foca em cravadas por bispos, torres e damas.

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `pinned_pieces_player` | int | Nº de peças do jogador absolutamente cravadas (contra o rei) |
| `pinned_pieces_opponent` | int | Nº de peças do adversário absolutamente cravadas |

### Implementação

```python
def _pin_features(board: chess.Board) -> dict:
    """Count absolutely pinned pieces (pinned to king)."""
    turn = board.turn
    opp = not turn

    def count_pinned(board, color):
        count = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_pinned(color, sq):
                    count += 1
        return count

    return {
        "pinned_pieces_player": count_pinned(board, turn),
        "pinned_pieces_opponent": count_pinned(board, opp),
    }
```

### Custo computacional

- `board.is_pinned()` é O(1) no `python-chess` (usa bitboards pré-computados).
- Total: 32 chamadas no máximo (peças no tabuleiro).

---

## Grupo 11 — Segurança do rei (avançada)

Complementa o grupo 3 da V1 com medidas mais granulares de vulnerabilidade do rei.

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `king_attackers_player` | int | Nº de peças adversárias que atacam casas ao redor do rei do jogador |
| `king_attackers_opponent` | int | Nº de peças do jogador que atacam casas ao redor do rei adversário |
| `king_open_files_player` | int | Nº de colunas abertas (sem peões) adjacentes ao rei do jogador |
| `king_escape_squares_player` | int | Nº de casas para onde o rei do jogador pode mover legalmente |

### Implementação

```python
def _king_safety_v2(board: chess.Board) -> dict:
    turn = board.turn
    opp = not turn

    def king_zone_attackers(board, king_color, attacker_color):
        king_sq = board.king(king_color)
        if king_sq is None:
            return 0
        zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]) | chess.SquareSet.from_square(king_sq)
        attacker_sqs = set()
        for sq in zone:
            for att in board.attackers(attacker_color, sq):
                attacker_sqs.add(att)
        return len(attacker_sqs)

    def king_open_files(board, color):
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        king_file = chess.square_file(king_sq)
        count = 0
        for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
            has_pawn = False
            for r in range(8):
                sq = chess.square(f, r)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN:
                    has_pawn = True
                    break
            if not has_pawn:
                count += 1
        return count

    def king_escape_squares(board, color):
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        count = 0
        for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
            piece = board.piece_at(sq)
            if piece and piece.color == color:
                continue
            if not board.is_attacked_by(not color, sq):
                count += 1
        return count

    return {
        "king_attackers_player": king_zone_attackers(board, turn, opp),
        "king_attackers_opponent": king_zone_attackers(board, opp, turn),
        "king_open_files_player": king_open_files(board, turn),
        "king_escape_squares_player": king_escape_squares(board, turn),
    }
```

---

## Grupo 12 — Tensão e complexidade

Features que medem o grau de "perigo" da posição — posições mais tensas têm mais lances críticos e maior probabilidade de erro.

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `total_attacks_player` | int | Nº total de casas atacadas pelo jogador |
| `total_attacks_opponent` | int | Nº total de casas atacadas pelo adversário |
| `contested_squares` | int | Nº de casas atacadas por ambos os lados |
| `undefended_pieces_player` | int | Nº de peças do jogador (excl. peões) sem nenhum defensor |

### Implementação

```python
def _tension_features(board: chess.Board) -> dict:
    turn = board.turn
    opp = not turn

    attacked_by_player = chess.SquareSet()
    attacked_by_opp = chess.SquareSet()
    for sq in chess.SQUARES:
        if board.is_attacked_by(turn, sq):
            attacked_by_player.add(sq)
        if board.is_attacked_by(opp, sq):
            attacked_by_opp.add(sq)

    contested = attacked_by_player & attacked_by_opp

    undefended = 0
    for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(pt, turn):
            if not board.attackers(turn, sq):
                undefended += 1

    return {
        "total_attacks_player": len(attacked_by_player),
        "total_attacks_opponent": len(attacked_by_opp),
        "contested_squares": len(contested),
        "undefended_pieces_player": undefended,
    }
```

---

## Resumo: features V2

| Grupo | Qtd. | Tipo | Cohen's d esperado |
|-------|------|------|-------------------|
| **G8 — Peças indefesas** | 5 | Ameaças diretas | Alto (> 0.5) |
| **G9 — Capturas com ganho** | 4 | Ameaças de troca | Médio–Alto |
| **G10 — Cravadas** | 2 | Tática específica | Médio |
| **G11 — Rei (avançado)** | 4 | Segurança detalhada | Médio |
| **G12 — Tensão** | 4 | Complexidade geral | Pequeno–Médio |
| **Total V2** | **19** | — | — |
| **Total (V1 + V2)** | **52** | — | — |

### Impacto esperado

Com features que capturam directamente as causas dos erros (peças penduradas, ameaças de captura, cravadas), espera-se:

| Cenário | F1-ruim estimado | Justificativa |
|---------|------------------|---------------|
| V1 (actual) | 0.35 | Features posicionais apenas |
| V2 conservador | 0.45–0.50 | Features táticas com sinal mais forte |
| V2 optimista | 0.55–0.65 | Se features táticas tiverem Cohen's d > 0.5 |

---

## Pipeline de upgrade

### O que NÃO muda

- Dados brutos (PGN), filtragem e rotulagem — reutilizamos `data/labeled/moves_labeled.csv`
- Modelos (DT + RF) e hiperparâmetros grid — reutilizamos a mesma configuração
- Métricas e avaliação — mesmo framework

### O que muda

1. **`src/extract_features.py`** — adicionar os 5 novos grupos de features
2. **`data/features/features_v2.csv`** — novo CSV com 52 features
3. **Re-treino** dos modelos com as novas features
4. **Re-avaliação** comparando V1 vs V2
5. **Notebook** atualizado com a análise comparativa

### Etapas do dev-loop

| Etapa | Descrição | Artefato |
|-------|-----------|----------|
| 08 — Features táticas | Implementar grupos 8–12, gerar `features_v2.csv` | `data/features/features_v2.csv` |
| 09 — Re-treino V2 | Treinar DT + RF com 52 features | `data/models_v2/` |
| 10 — Avaliação V1 vs V2 | Comparar métricas, gerar gráficos comparativos | `data/evaluation_v2/` |
| 11 — Notebook V2 | Atualizar notebook com seção de diagnóstico + resultados V2 | `notebooks/chess_move_classifier.ipynb` |

---

## Checklist de implementação

- [ ] `hanging_pieces_player` e `hanging_pieces_opponent` testados com posições conhecidas
- [ ] `threats_against_player` e `threats_against_opponent` validados
- [ ] `pinned_pieces_player` e `pinned_pieces_opponent` — confirmar que `board.is_pinned()` funciona como esperado
- [ ] Features de rei — `BB_KING_ATTACKS` existe no `python-chess` (verificar versão)
- [ ] Nenhuma feature retorna NaN ou valores inesperados
- [ ] Extração de 109,290 linhas com 52 features completa sem erros
- [ ] Correlações das novas features com label são superiores às da V1 (> 0.10)

---

## Referências

- [diagnostico-v1.md](../06-riscos-e-limitacoes/diagnostico-v1.md) — diagnóstico completo da V1
- [engenharia-de-features.md](engenharia-de-features.md) — features V1 (grupos 1–7)
- [riscos.md](../06-riscos-e-limitacoes/riscos.md) — R3 previa esta limitação
