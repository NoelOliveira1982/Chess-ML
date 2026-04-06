# Engenharia de Features

## Visão geral

Cada linha do dataset final representa **um lance**. As features descrevem o **estado do tabuleiro antes do lance** e **características do lance em si**. Todas são extraídas programaticamente usando `python-chess`.

A tabela final terá formato:

```
| feature_1 | feature_2 | ... | feature_N | label |
|-----------|-----------|-----|-----------|-------|
|   ...     |   ...     |     |   ...     |  bom  |
```

---

## Grupo 1 — Material

Features que contam o material de cada lado e a diferença.

| Feature | Tipo | Descrição | Extração com python-chess |
|---------|------|-----------|---------------------------|
| `white_pawns` | int | Nº de peões brancos | `len(board.pieces(chess.PAWN, chess.WHITE))` |
| `white_knights` | int | Nº de cavalos brancos | `len(board.pieces(chess.KNIGHT, chess.WHITE))` |
| `white_bishops` | int | Nº de bispos brancos | `len(board.pieces(chess.BISHOP, chess.WHITE))` |
| `white_rooks` | int | Nº de torres brancas | `len(board.pieces(chess.ROOK, chess.WHITE))` |
| `white_queens` | int | Nº de damas brancas | `len(board.pieces(chess.QUEEN, chess.WHITE))` |
| `black_pawns` | int | Nº de peões pretos | `len(board.pieces(chess.PAWN, chess.BLACK))` |
| `black_knights` | int | Nº de cavalos pretos | `len(board.pieces(chess.KNIGHT, chess.BLACK))` |
| `black_bishops` | int | Nº de bispos pretos | `len(board.pieces(chess.BISHOP, chess.BLACK))` |
| `black_rooks` | int | Nº de torres pretas | `len(board.pieces(chess.ROOK, chess.BLACK))` |
| `black_queens` | int | Nº de damas pretas | `len(board.pieces(chess.QUEEN, chess.BLACK))` |
| `material_diff` | float | Diferença material ponderada (perspectiva do jogador da vez) | Calculada com pesos padrão |

### Pesos de material padrão

```python
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

def material_score(board, color):
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
    return score

def material_diff(board):
    """Positivo = vantagem do jogador da vez."""
    turn = board.turn
    return material_score(board, turn) - material_score(board, not turn)
```

---

## Grupo 2 — Mobilidade

| Feature | Tipo | Descrição | Extração |
|---------|------|-----------|----------|
| `legal_moves_player` | int | Nº de lances legais do jogador da vez | `len(list(board.legal_moves))` |
| `legal_moves_opponent` | int | Nº de lances legais do adversário | Trocar turno, contar, voltar |
| `mobility_diff` | int | Diferença de mobilidade | `player - opponent` |

```python
def mobility(board):
    player_moves = len(list(board.legal_moves))
    board.push(chess.Move.null())
    opponent_moves = len(list(board.legal_moves))
    board.pop()
    return player_moves, opponent_moves
```

---

## Grupo 3 — Segurança do rei

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `player_castled` | bool | Jogador da vez já fez roque |
| `opponent_castled` | bool | Adversário já fez roque |
| `player_can_castle` | bool | Jogador ainda tem direito a roque |
| `king_pawn_shield` | int | Nº de peões na frente do rei do jogador (0–3) |

### Detecção de roque realizado

Roque realizado não é diretamente acessível pelo `board` — precisa ser inferido pelo histórico de lances ou pela posição do rei:

```python
def has_castled(board, color):
    """Heurística: rei fora da casa inicial indica que rocou (ou moveu)."""
    king_sq = board.king(color)
    initial = chess.E1 if color == chess.WHITE else chess.E8
    return king_sq != initial

def king_pawn_shield(board, color):
    """Conta peões do próprio lado adjacentes ao rei (na frente)."""
    king_sq = board.king(color)
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    shield_rank = king_rank + (1 if color == chess.WHITE else -1)

    if not (0 <= shield_rank <= 7):
        return 0

    count = 0
    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        sq = chess.square(f, shield_rank)
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            count += 1
    return count
```

---

## Grupo 4 — Estrutura de peões

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `player_doubled_pawns` | int | Nº de peões dobrados do jogador |
| `player_isolated_pawns` | int | Nº de peões isolados do jogador |
| `player_passed_pawns` | int | Nº de peões passados do jogador |
| `opponent_passed_pawns` | int | Nº de peões passados do adversário |

```python
def doubled_pawns(board, color):
    """Peões na mesma coluna."""
    count = 0
    for file in range(8):
        pawns_in_file = 0
        for rank in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                pawns_in_file += 1
        if pawns_in_file > 1:
            count += pawns_in_file - 1
    return count

def isolated_pawns(board, color):
    """Peões sem peões amigos em colunas adjacentes."""
    pawn_files = set()
    for sq in board.pieces(chess.PAWN, color):
        pawn_files.add(chess.square_file(sq))

    count = 0
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        has_neighbor = (f - 1 in pawn_files) or (f + 1 in pawn_files)
        if not has_neighbor:
            count += 1
    return count
```

---

## Grupo 5 — Controle do centro

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `player_center_control` | int | Nº de casas centrais atacadas pelo jogador |
| `opponent_center_control` | int | Nº de casas centrais atacadas pelo adversário |
| `player_center_occupation` | int | Nº de peças do jogador nas 4 casas centrais |

Casas centrais: `d4`, `d5`, `e4`, `e5`.

```python
CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

def center_control(board, color):
    count = 0
    for sq in CENTER_SQUARES:
        if board.is_attacked_by(color, sq):
            count += 1
    return count

def center_occupation(board, color):
    count = 0
    for sq in CENTER_SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            count += 1
    return count
```

---

## Grupo 6 — Características do lance

Features extraídas do lance jogado, não da posição.

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `is_capture` | bool | O lance captura uma peça? |
| `is_check` | bool | O lance dá xeque? |
| `is_promotion` | bool | O lance é uma promoção de peão? |
| `moved_piece` | int | Tipo da peça movida (1=peão, 2=cavalo, ..., 6=rei) |
| `move_to_center` | bool | Casa de destino é uma das 4 centrais? |
| `move_to_extended_center` | bool | Casa de destino está no centro expandido (c3–f6)? |

```python
def move_features(board, move):
    return {
        "is_capture": board.is_capture(move),
        "is_check": board.gives_check(move),
        "is_promotion": move.promotion is not None,
        "moved_piece": board.piece_at(move.from_square).piece_type,
        "move_to_center": move.to_square in CENTER_SQUARES,
    }
```

---

## Grupo 7 — Contexto da partida

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `move_number` | int | Número do lance na partida (ply // 2) |
| `is_white` | bool | O jogador da vez é branco? |

---

## Resumo: total de features

| Grupo | Quantidade |
|-------|------------|
| Material | 11 |
| Mobilidade | 3 |
| Segurança do rei | 4 |
| Estrutura de peões | 4 |
| Controle do centro | 3 |
| Características do lance | 6 |
| Contexto | 2 |
| **Total** | **~33** |

## Observações

- Nenhuma feature usa one-hot de casas individuais (ver [DP-06](../01-visao-geral/decisoes-de-projeto.md)).
- Todas as features são numéricas ou booleanas (convertidas para 0/1) — compatíveis diretamente com scikit-learn.
- Features podem ser adicionadas ou removidas após análise exploratória; documentar quaisquer alterações.
