# Features Look-Ahead — Upgrade V2 → V3

## Motivação

O diagnóstico da V2 ([diagnostico-v2.md](../06-riscos-e-limitacoes/diagnostico-v2.md)) mostrou que as 52 features (posicionais + táticas) descrevem apenas o **estado do tabuleiro antes do lance**. A qualidade de um lance depende do seu **impacto** — o que muda no tabuleiro depois de jogar. Para capturar isso, a V3 introduz features de **look-ahead**: avaliar a posição resultante e calcular deltas.

### Referência académica

- ResearchSquare 2025: features de delta + complexidade → F1 = 0.75, AUC = 0.82
- Stockfish 18 (SFNNv10): "Threat Inputs" (pares atacante-alvo dinâmicos) → +46 Elo
- Maia Chess (Microsoft): predição de lances humanos com features de posição antes e depois

---

## Impacto no pipeline pesado

**NÃO é necessário re-rodar a rotulagem Stockfish (etapa 03, ~174 min).**

Os dados existentes (`data/labeled/moves_labeled.csv`) contêm o `fen_before` e o `move_uci` — tudo o que é necessário para calcular a posição depois do lance via `board.push(move)`. A V3 modifica apenas `extract_features.py`, sem tocar nos dados brutos nem na rotulagem.

### Custo computacional

| Operação | V2 | V3 estimado |
|---|---|---|
| Extração de features | ~9s (6 workers) | ~20-25s (6 workers) |
| Treino dos modelos | ~30s | ~35s (mais features) |
| Pipeline total | ~40s | ~60s |

O aumento é trivial — `board.push(move)` é O(1) no `python-chess`.

---

## Grupo 13 — Features delta (antes vs depois do lance)

A ideia central: para cada feature existente que faz sentido comparar, calcular o valor **depois do lance** e o **delta** (depois − antes).

| Feature | Tipo | Descrição | Cohen's d esperado |
|---------|------|-----------|-------------------|
| `delta_hanging_player` | int | Mudança em peças indefesas do jogador após o lance | Alto (> 0.5) |
| `delta_hanging_opponent` | int | Mudança em peças indefesas do adversário | Médio–Alto |
| `delta_hanging_value_player` | int | Mudança no valor material indefeso do jogador | Alto |
| `delta_threats_against_player` | int | Novas ameaças de captura com ganho contra o jogador | Alto |
| `delta_mobility_player` | int | Mudança na mobilidade do jogador (legal moves) | Médio |
| `delta_mobility_opponent` | int | Mudança na mobilidade do adversário | Médio–Alto |
| `delta_contested_squares` | int | Mudança no nº de casas disputadas | Médio |
| `delta_king_attackers_player` | int | Mudança no nº de peças atacando a zona do rei do jogador | Médio |

### Implementação

```python
def _delta_features(board_before: chess.Board, move: chess.Move) -> dict:
    """Compute feature deltas between position before and after the move."""
    board_after = board_before.copy()
    board_after.push(move)

    # Hanging pieces before
    hang_before = _hanging_features(board_before)
    # Hanging pieces after (sides are swapped because turn changed)
    hang_after = _hanging_features(board_after)

    # Mobility before
    mob_before = _mobility_features(board_before)
    # Mobility after (from opponent's perspective, who is now to move)
    mob_after = _mobility_features(board_after)

    # Tension before/after
    tension_before = _tension_features(board_before)
    tension_after = _tension_features(board_after)

    # King safety before/after
    king_before = _king_safety_v2(board_before)
    king_after = _king_safety_v2(board_after)

    return {
        # After push, "player" in the new position = opponent in the old one.
        # We want: did MY pieces become more/less hanging?
        "delta_hanging_player": (
            hang_after["hanging_pieces_opponent"]  # my pieces, from new side's view
            - hang_before["hanging_pieces_player"]
        ),
        "delta_hanging_opponent": (
            hang_after["hanging_pieces_player"]  # their pieces
            - hang_before["hanging_pieces_opponent"]
        ),
        "delta_hanging_value_player": (
            hang_after["hanging_value_opponent"]
            - hang_before["hanging_value_player"]
        ),
        "delta_threats_against_player": (
            _threat_features(board_after)["threats_against_opponent"]
            - _threat_features(board_before)["threats_against_player"]
        ),
        # After push, opponent's legal moves = board_after.legal_moves
        "delta_mobility_player": (
            mob_after["legal_moves_opponent"]  # my moves from new perspective
            - mob_before["legal_moves_player"]
        ),
        "delta_mobility_opponent": (
            mob_after["legal_moves_player"]  # their moves
            - mob_before["legal_moves_opponent"]
        ),
        "delta_contested_squares": (
            tension_after["contested_squares"]
            - tension_before["contested_squares"]
        ),
        "delta_king_attackers_player": (
            king_after["king_attackers_opponent"]  # attackers on my king
            - king_before["king_attackers_player"]
        ),
    }
```

### Atenção ao swap de perspectiva

Depois de `board.push(move)`, o turno muda. Portanto:
- O que era "player" passa a ser "opponent" e vice-versa
- `board_after["hanging_pieces_opponent"]` = peças do jogador original que estão penduradas
- `board_after["legal_moves_player"]` = mobilidade do adversário original

Esta lógica de swap é **crítica** para os deltas serem corretos.

---

## Grupo 14 — Resposta do adversário (look-ahead de 1 ply)

Features que medem **o que o adversário pode fazer imediatamente** após o lance jogado.

| Feature | Tipo | Descrição | Cohen's d esperado |
|---------|------|-----------|-------------------|
| `opponent_best_capture_value` | int | Valor da melhor captura disponível para o adversário após o lance | Alto (> 0.5) |
| `opponent_can_check` | bool(int) | O adversário pode dar xeque na resposta? | Médio |
| `opponent_has_fork_capture` | bool(int) | Adversário tem captura que ataca ≥2 peças simultaneamente? | Médio |
| `opponent_num_good_captures` | int | Nº de capturas com ganho material (SEE > 0) para o adversário | Alto |
| `created_hanging_self` | bool(int) | O lance deixou a peça movida sem defesa? | Alto |

### Implementação

```python
def _opponent_response_features(board_before: chess.Board, move: chess.Move) -> dict:
    """Evaluate what the opponent can do after our move."""
    board_after = board_before.copy()
    board_after.push(move)

    # Best capture available to opponent
    best_capture_val = 0
    num_good_captures = 0
    can_check = False

    for opp_move in board_after.legal_moves:
        if board_after.gives_check(opp_move):
            can_check = True

        if board_after.is_capture(opp_move):
            captured_sq = opp_move.to_square
            captured_piece = board_after.piece_at(captured_sq)

            # En passant
            if captured_piece is None and board_after.is_en_passant(opp_move):
                capture_val = PIECE_VALUES[chess.PAWN]
            elif captured_piece:
                capture_val = PIECE_VALUES.get(captured_piece.piece_type, 0)
            else:
                capture_val = 0

            # Simple SEE: is the attacker cheaper than the target?
            attacker = board_after.piece_at(opp_move.from_square)
            attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            net_gain = capture_val - attacker_val

            if net_gain > 0 or not board_before.attackers(board_before.turn, opp_move.to_square):
                # Winning capture or undefended target
                num_good_captures += 1

            best_capture_val = max(best_capture_val, capture_val)

    # Did the move leave the moved piece undefended?
    to_sq = move.to_square
    moved_piece = board_after.piece_at(to_sq)
    created_hanging = 0
    if moved_piece and moved_piece.piece_type != chess.PAWN:
        defenders = board_after.attackers(board_before.turn, to_sq)
        attackers = board_after.attackers(not board_before.turn, to_sq)
        if attackers and not defenders:
            created_hanging = 1

    return {
        "opponent_best_capture_value": best_capture_val,
        "opponent_can_check": int(can_check),
        "opponent_num_good_captures": num_good_captures,
        "created_hanging_self": created_hanging,
    }
```

### Custo computacional

- Iterar sobre `board_after.legal_moves`: ~30-40 lances por posição
- Cada lance: `is_capture()`, `gives_check()`, `piece_at()` — todos O(1)
- Total por posição: ~200 operações → desprezível

---

## Grupo 15 — Static Exchange Evaluation (SEE) simplificada

A SEE avalia se uma sequência de capturas num quadrado resulta em ganho ou perda material. É a base de **toda** engine de xadrez para decidir se uma captura vale a pena.

| Feature | Tipo | Descrição | Cohen's d esperado |
|---------|------|-----------|-------------------|
| `see_of_move` | int | Resultado SEE do lance jogado (se captura). 0 se não é captura | Alto |
| `worst_see_against_player` | int | Pior SEE possível contra uma peça do jogador (após o lance) | Alto |
| `is_losing_capture` | bool(int) | O lance é uma captura com SEE < 0? | Alto |

### Implementação (SEE simplificada)

```python
def _simple_see(board: chess.Board, square: chess.Square) -> int:
    """Simplified Static Exchange Evaluation on a square.

    Simulates alternating captures on the square, starting with the
    side that does NOT own the piece currently on the square (the
    attacker). Returns the net material gain/loss for the initial
    attacker.
    """
    piece = board.piece_at(square)
    if piece is None:
        return 0

    # Build ordered list of attackers for both sides
    def get_attackers_sorted(board, color, square):
        attackers = []
        for sq in board.attackers(color, square):
            p = board.piece_at(sq)
            if p:
                attackers.append((PIECE_VALUES.get(p.piece_type, 0), sq))
        attackers.sort()  # cheapest first
        return attackers

    target_val = PIECE_VALUES.get(piece.piece_type, 0)
    defender_color = piece.color
    attacker_color = not defender_color

    attackers = get_attackers_sorted(board, attacker_color, square)
    defenders = get_attackers_sorted(board, defender_color, square)

    if not attackers:
        return 0

    # Simulate swap
    gains = [target_val]  # gain from initial capture
    current_val = target_val
    att_idx = 0
    def_idx = 0

    # Attacker captures first
    if att_idx < len(attackers):
        current_val = attackers[att_idx][0]
        att_idx += 1
    else:
        return gains[0]

    # Alternating: defender recaptures, then attacker, etc.
    while True:
        # Defender recaptures
        if def_idx < len(defenders):
            gains.append(current_val)
            current_val = defenders[def_idx][0]
            def_idx += 1
        else:
            break

        # Attacker recaptures
        if att_idx < len(attackers):
            gains.append(current_val)
            current_val = attackers[att_idx][0]
            att_idx += 1
        else:
            break

    # Negamax back through the gains list
    while len(gains) > 1:
        gains[-2] = max(gains[-2], -gains[-1] + gains[-2])
        gains.pop()

    return gains[0]


def _see_features(board_before: chess.Board, move: chess.Move) -> dict:
    """SEE-based features for the played move."""
    see_val = 0
    is_losing = 0

    if board_before.is_capture(move):
        board_temp = board_before.copy()
        board_temp.push(move)
        see_val = _simple_see(board_before, move.to_square)
        if see_val < 0:
            is_losing = 1

    # Worst SEE against any of our pieces after the move
    board_after = board_before.copy()
    board_after.push(move)

    worst_see = 0
    player_color = board_before.turn
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board_after.pieces(pt, player_color):
            if board_after.attackers(not player_color, sq):
                see = _simple_see(board_after, sq)
                worst_see = min(worst_see, -see)  # negate: loss for us

    return {
        "see_of_move": see_val,
        "worst_see_against_player": worst_see,
        "is_losing_capture": is_losing,
    }
```

### Custo computacional

- SEE por quadrado: O(número de atacantes × defensores) — tipicamente 2-4 peças → negligível
- `worst_see_against_player`: avaliar ~10-15 peças × SEE cada → ~50 operações
- Total estimado: ~100-200 operações por posição → desprezível

---

## Resumo: features V3

| Grupo | Qtd | Tipo | Cohen's d esperado |
|-------|-----|------|-------------------|
| **G13 — Deltas (antes vs depois)** | 8 | Impacto do lance | Alto (> 0.5) |
| **G14 — Resposta do adversário** | 4 | Look-ahead 1 ply | Alto |
| **G15 — SEE** | 3 | Avaliação de trocas | Alto |
| **Total V3 (novas)** | **15** | — | — |
| **Total (V1 + V2 + V3)** | **67** | — | — |

### Impacto esperado

| Cenário | F1-ruim estimado | Justificativa |
|---------|------------------|---------------|
| V2 (atual) | 0.37 | Features pré-lance apenas |
| V3 conservador | 0.45–0.50 | Deltas capturam impacto do lance |
| V3 optimista | 0.50–0.60 | Deltas + SEE + resposta do adversário |

A meta acadêmica de **F1 ≥ 0.50** é realista com estas features.

---

## Pipeline de upgrade V2 → V3

### O que NÃO muda

- **Dados brutos** (PGN) — reutilizamos
- **Filtragem e amostragem** — reutilizamos `data/filtered/moves_filtered.csv`
- **Rotulagem Stockfish** — reutilizamos `data/labeled/moves_labeled.csv` (NÃO precisa re-rodar os ~174 min)
- **Modelos** (DT + RF) — mesmos algoritmos e grid de hiperparâmetros
- **Métricas** — mesmo framework de avaliação

### O que muda

1. **`src/extract_features.py`** — adicionar flag `--v3`, grupos G13–G15
2. **`data/features/features_v3.csv`** — novo CSV com 67 features
3. **Re-treino** dos modelos com as novas features
4. **Re-avaliação** comparando V2 vs V3
5. **Notebook** atualizado com análise comparativa V1 → V2 → V3

### Etapas do dev-loop

| Etapa | Descrição | Artefato |
|-------|-----------|----------|
| 12 — Features look-ahead | Implementar G13–G15, gerar `features_v3.csv` | `data/features/features_v3.csv` |
| 13 — Re-treino V3 | Treinar DT + RF com 67 features | `data/models_v3/` |
| 14 — Avaliação V2 vs V3 | Comparar métricas, gerar gráficos V1→V2→V3 | `data/evaluation_v3/` |
| 15 — Notebook V3 | Atualizar notebook com seção V3 | `notebooks/chess_move_classifier.ipynb` |

---

## Checklist de implementação

- [ ] Swap de perspectiva correto nos deltas (após `board.push()`, o turno muda)
- [ ] `delta_hanging_player`: positivo = mais peças indefesas (mau), negativo = menos (bom)
- [ ] `opponent_best_capture_value`: testar com posições onde adversário pode comer peça grátis
- [ ] `created_hanging_self`: confirmar detecção quando peça se move para casa sem defensores
- [ ] `_simple_see()`: validar com trocas conhecidas (ex.: BxN defendido por P → SEE = 3-1 = 2 para defensor)
- [ ] `worst_see_against_player`: confirmar que retorna valor negativo quando perdemos material
- [ ] Nenhuma feature retorna NaN ou valores inesperados
- [ ] Extração de 109,290 linhas com 67 features completa sem erros
- [ ] Correlações das novas features com label são superiores às da V2 (> 0.15)

---

## Referências

- [diagnostico-v2.md](../06-riscos-e-limitacoes/diagnostico-v2.md) — diagnóstico completo da V2
- [features-taticas.md](features-taticas.md) — features V2 (grupos 8–12)
- [engenharia-de-features.md](engenharia-de-features.md) — features V1 (grupos 1–7)
- ResearchSquare 2025 — "Complexity scoring system" com F1 = 0.75, AUC = 0.82
- Stockfish 18 SFNNv10 — "Threat Inputs" (+46 Elo)
- [SEE — Chessprogramming wiki](https://www.chessprogramming.org/Static_Exchange_Evaluation)
