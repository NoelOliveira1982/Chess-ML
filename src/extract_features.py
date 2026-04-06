"""
Extract chess features from labeled moves for supervised classification.

Reads moves_labeled.csv (fen_before, move_uci, label) and computes ~33
features per move across 7 groups: material, mobility, king safety,
pawn structure, center control, move characteristics, and game context.

Output: data/features/features.csv
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import pandas as pd

INPUT_CSV = Path("data/labeled/moves_labeled.csv")
OUTPUT_DIR = Path("data/features")
OUTPUT_CSV = OUTPUT_DIR / "features.csv"
OUTPUT_CSV_V2 = OUTPUT_DIR / "features_v2.csv"
OUTPUT_CSV_V3 = OUTPUT_DIR / "features_v3.csv"

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

EXTENDED_CENTER = set()
for _f in range(2, 6):
    for _r in range(2, 6):
        EXTENDED_CENTER.add(chess.square(_f, _r))


# ── Group 1: Material ──────────────────────────────────────────────

def _material_score(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(pt, color)) * val
        for pt, val in PIECE_VALUES.items()
    )


def _material_features(board: chess.Board) -> dict:
    feats: dict = {}
    for name, color in [("white", chess.WHITE), ("black", chess.BLACK)]:
        feats[f"{name}_pawns"] = len(board.pieces(chess.PAWN, color))
        feats[f"{name}_knights"] = len(board.pieces(chess.KNIGHT, color))
        feats[f"{name}_bishops"] = len(board.pieces(chess.BISHOP, color))
        feats[f"{name}_rooks"] = len(board.pieces(chess.ROOK, color))
        feats[f"{name}_queens"] = len(board.pieces(chess.QUEEN, color))
    turn = board.turn
    feats["material_diff"] = _material_score(board, turn) - _material_score(board, not turn)
    return feats


# ── Group 2: Mobility ──────────────────────────────────────────────

def _mobility_features(board: chess.Board) -> dict:
    player_moves = board.legal_moves.count()

    if board.is_check():
        opponent_moves = 0
    else:
        board.push(chess.Move.null())
        opponent_moves = board.legal_moves.count()
        board.pop()

    return {
        "legal_moves_player": player_moves,
        "legal_moves_opponent": opponent_moves,
        "mobility_diff": player_moves - opponent_moves,
    }


# ── Group 3: King safety ──────────────────────────────────────────

def _has_castled(board: chess.Board, color: chess.Color) -> bool:
    king_sq = board.king(color)
    if king_sq is None:
        return False
    initial = chess.E1 if color == chess.WHITE else chess.E8
    return king_sq != initial


def _king_pawn_shield(board: chess.Board, color: chess.Color) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
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


def _king_safety_features(board: chess.Board) -> dict:
    turn = board.turn
    opp = not turn
    return {
        "player_castled": int(_has_castled(board, turn)),
        "opponent_castled": int(_has_castled(board, opp)),
        "player_can_castle": int(board.has_castling_rights(turn)),
        "king_pawn_shield": _king_pawn_shield(board, turn),
    }


# ── Group 4: Pawn structure ───────────────────────────────────────

def _doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    count = 0
    for file_idx in range(8):
        pawns_in_file = 0
        for rank_idx in range(8):
            sq = chess.square(file_idx, rank_idx)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                pawns_in_file += 1
        if pawns_in_file > 1:
            count += pawns_in_file - 1
    return count


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    pawn_files = set()
    for sq in board.pieces(chess.PAWN, color):
        pawn_files.add(chess.square_file(sq))

    count = 0
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        if (f - 1 not in pawn_files) and (f + 1 not in pawn_files):
            count += 1
    return count


def _passed_pawns(board: chess.Board, color: chess.Color) -> int:
    opp = not color
    count = 0
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True

        ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)
        for adj_f in range(max(0, f - 1), min(7, f + 1) + 1):
            for ahead_r in ahead:
                check_sq = chess.square(adj_f, ahead_r)
                piece = board.piece_at(check_sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == opp:
                    is_passed = False
                    break
            if not is_passed:
                break
        if is_passed:
            count += 1
    return count


def _pawn_structure_features(board: chess.Board) -> dict:
    turn = board.turn
    opp = not turn
    return {
        "player_doubled_pawns": _doubled_pawns(board, turn),
        "player_isolated_pawns": _isolated_pawns(board, turn),
        "player_passed_pawns": _passed_pawns(board, turn),
        "opponent_passed_pawns": _passed_pawns(board, opp),
    }


# ── Group 5: Center control ───────────────────────────────────────

def _center_control(board: chess.Board, color: chess.Color) -> int:
    return sum(1 for sq in CENTER_SQUARES if board.is_attacked_by(color, sq))


def _center_occupation(board: chess.Board, color: chess.Color) -> int:
    count = 0
    for sq in CENTER_SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            count += 1
    return count


def _center_features(board: chess.Board) -> dict:
    turn = board.turn
    opp = not turn
    return {
        "player_center_control": _center_control(board, turn),
        "opponent_center_control": _center_control(board, opp),
        "player_center_occupation": _center_occupation(board, turn),
    }


# ── Group 6: Move characteristics ─────────────────────────────────

def _move_features(board: chess.Board, move: chess.Move) -> dict:
    piece = board.piece_at(move.from_square)
    return {
        "is_capture": int(board.is_capture(move)),
        "is_check": int(board.gives_check(move)),
        "is_promotion": int(move.promotion is not None),
        "moved_piece": piece.piece_type if piece else 0,
        "move_to_center": int(move.to_square in CENTER_SQUARES),
        "move_to_extended_center": int(move.to_square in EXTENDED_CENTER),
    }


# ── Group 7: Game context ─────────────────────────────────────────

def _context_features(row: dict) -> dict:
    return {
        "move_number": int(row["move_number"]),
        "is_white": int(row["color"] == "white"),
    }


# ── Group 8: Hanging pieces (V2) ─────────────────────────────────

def _hanging_features(board: chess.Board) -> dict:
    """Detect undefended pieces for both sides."""
    turn = board.turn
    opp = not turn

    def hanging(board, color, attacker_color):
        count = 0
        value = 0
        min_attacker_vs_piece = 999
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(pt, color):
                attackers = board.attackers(attacker_color, sq)
                defenders = board.attackers(color, sq)
                if attackers and not defenders:
                    count += 1
                    piece_val = PIECE_VALUES.get(pt, 0)
                    value += piece_val
                    for att_sq in attackers:
                        att_piece = board.piece_at(att_sq)
                        if att_piece:
                            att_val = PIECE_VALUES.get(att_piece.piece_type, 0)
                            diff = att_val - piece_val
                            min_attacker_vs_piece = min(min_attacker_vs_piece, diff)
        if min_attacker_vs_piece == 999:
            min_attacker_vs_piece = 0
        return count, value, min_attacker_vs_piece

    h_player, hv_player, mavp = hanging(board, turn, opp)
    h_opp, hv_opp, _ = hanging(board, opp, turn)

    return {
        "hanging_pieces_player": h_player,
        "hanging_pieces_opponent": h_opp,
        "hanging_value_player": hv_player,
        "hanging_value_opponent": hv_opp,
        "min_attacker_vs_piece_player": mavp,
    }


# ── Group 9: Capture threats (V2) ────────────────────────────────

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
                            break
        return count, max_gain

    t_player, mt_player = threats(board, turn, opp)
    t_opp, mt_opp = threats(board, opp, turn)

    return {
        "threats_against_player": t_player,
        "threats_against_opponent": t_opp,
        "max_threat_value_player": mt_player,
        "max_threat_value_opponent": mt_opp,
    }


# ── Group 10: Pins (V2) ──────────────────────────────────────────

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


# ── Group 11: King safety V2 ─────────────────────────────────────

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


# ── Group 12: Tension & complexity (V2) ──────────────────────────

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


# ── Group 13: Delta features — before vs after move (V3) ─────────

def _delta_features(board_before: chess.Board, move: chess.Move) -> dict:
    """Compute feature deltas between position before and after the move.

    After board.push(move), the turn flips. So what was 'player' becomes
    'opponent' and vice-versa in the *_features helpers.
    """
    board_after = board_before.copy()
    board_after.push(move)

    hang_before = _hanging_features(board_before)
    hang_after = _hanging_features(board_after)

    mob_before = _mobility_features(board_before)
    mob_after = _mobility_features(board_after)

    tension_before = _tension_features(board_before)
    tension_after = _tension_features(board_after)

    king_before = _king_safety_v2(board_before)
    king_after = _king_safety_v2(board_after)

    threat_before = _threat_features(board_before)
    threat_after = _threat_features(board_after)

    return {
        "delta_hanging_player": (
            hang_after["hanging_pieces_opponent"]
            - hang_before["hanging_pieces_player"]
        ),
        "delta_hanging_opponent": (
            hang_after["hanging_pieces_player"]
            - hang_before["hanging_pieces_opponent"]
        ),
        "delta_hanging_value_player": (
            hang_after["hanging_value_opponent"]
            - hang_before["hanging_value_player"]
        ),
        "delta_threats_against_player": (
            threat_after["threats_against_opponent"]
            - threat_before["threats_against_player"]
        ),
        "delta_mobility_player": (
            mob_after["legal_moves_opponent"]
            - mob_before["legal_moves_player"]
        ),
        "delta_mobility_opponent": (
            mob_after["legal_moves_player"]
            - mob_before["legal_moves_opponent"]
        ),
        "delta_contested_squares": (
            tension_after["contested_squares"]
            - tension_before["contested_squares"]
        ),
        "delta_king_attackers_player": (
            king_after["king_attackers_opponent"]
            - king_before["king_attackers_player"]
        ),
    }


# ── Group 14: Opponent response — 1-ply look-ahead (V3) ─────────

def _opponent_response_features(board_before: chess.Board, move: chess.Move) -> dict:
    """Evaluate what the opponent can do immediately after our move."""
    board_after = board_before.copy()
    board_after.push(move)

    best_capture_val = 0
    num_good_captures = 0
    can_check = False

    for opp_move in board_after.legal_moves:
        if board_after.gives_check(opp_move):
            can_check = True

        if board_after.is_capture(opp_move):
            captured_sq = opp_move.to_square
            captured_piece = board_after.piece_at(captured_sq)

            if captured_piece is None and board_after.is_en_passant(opp_move):
                capture_val = PIECE_VALUES[chess.PAWN]
            elif captured_piece:
                capture_val = PIECE_VALUES.get(captured_piece.piece_type, 0)
            else:
                capture_val = 0

            attacker = board_after.piece_at(opp_move.from_square)
            attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            net_gain = capture_val - attacker_val

            if net_gain > 0 or not board_before.attackers(board_before.turn, opp_move.to_square):
                num_good_captures += 1

            best_capture_val = max(best_capture_val, capture_val)

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


# ── Group 15: Static Exchange Evaluation (V3) ────────────────────

def _simple_see(board: chess.Board, square: chess.Square) -> int:
    """Simplified Static Exchange Evaluation on a square.

    Simulates alternating captures on the square, cheapest attacker first.
    Returns the net material gain for the initial attacker (the side that
    does NOT own the piece currently on the square).

    Uses the standard swap/negamax algorithm:
      gain[0] = target_value
      gain[d] = value_of_piece_just_placed - gain[d-1]
    Then: gain[d-1] = -max(-gain[d-1], gain[d])  working backwards.
    """
    piece = board.piece_at(square)
    if piece is None:
        return 0

    def get_attackers_sorted(board, color, sq):
        result = []
        for s in board.attackers(color, sq):
            p = board.piece_at(s)
            if p:
                result.append((PIECE_VALUES.get(p.piece_type, 0), s))
        result.sort()
        return result

    target_val = PIECE_VALUES.get(piece.piece_type, 0)
    defender_color = piece.color
    attacker_color = not defender_color

    att_list = get_attackers_sorted(board, attacker_color, square)
    def_list = get_attackers_sorted(board, defender_color, square)

    if not att_list:
        return 0

    gain = [target_val]
    piece_on_sq_val = att_list[0][0]
    att_idx = 1
    def_idx = 0

    while True:
        if def_idx < len(def_list):
            gain.append(piece_on_sq_val - gain[-1])
            piece_on_sq_val = def_list[def_idx][0]
            def_idx += 1
        else:
            break

        if att_idx < len(att_list):
            gain.append(piece_on_sq_val - gain[-1])
            piece_on_sq_val = att_list[att_idx][0]
            att_idx += 1
        else:
            break

    d = len(gain) - 1
    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1

    return gain[0]


def _see_features(board_before: chess.Board, move: chess.Move) -> dict:
    """SEE-based features for the played move."""
    see_val = 0
    is_losing = 0

    if board_before.is_capture(move):
        see_val = _simple_see(board_before, move.to_square)
        if see_val < 0:
            is_losing = 1

    board_after = board_before.copy()
    board_after.push(move)

    worst_see = 0
    player_color = board_before.turn
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board_after.pieces(pt, player_color):
            if board_after.attackers(not player_color, sq):
                see = _simple_see(board_after, sq)
                worst_see = min(worst_see, -see)

    return {
        "see_of_move": see_val,
        "worst_see_against_player": worst_see,
        "is_losing_capture": is_losing,
    }


# ── Row-level extraction ──────────────────────────────────────────

_V2_ENABLED = False
_V3_ENABLED = False


def _extract_row(row: dict) -> dict:
    board = chess.Board(row["fen_before"])
    move = chess.Move.from_uci(row["move_uci"])

    feats: dict = {}
    feats.update(_material_features(board))
    feats.update(_mobility_features(board))
    feats.update(_king_safety_features(board))
    feats.update(_pawn_structure_features(board))
    feats.update(_center_features(board))
    feats.update(_move_features(board, move))
    feats.update(_context_features(row))

    if _V2_ENABLED or _V3_ENABLED:
        feats.update(_hanging_features(board))
        feats.update(_threat_features(board))
        feats.update(_pin_features(board))
        feats.update(_king_safety_v2(board))
        feats.update(_tension_features(board))

    if _V3_ENABLED:
        feats.update(_delta_features(board, move))
        feats.update(_opponent_response_features(board, move))
        feats.update(_see_features(board, move))

    feats["label"] = row["label"]
    return feats


def _extract_batch(rows: list[dict]) -> list[dict]:
    return [_extract_row(r) for r in rows]


def _init_worker(v2: bool, v3: bool = False) -> None:
    global _V2_ENABLED, _V3_ENABLED
    _V2_ENABLED = v2
    _V3_ENABLED = v3


# ── Main pipeline ─────────────────────────────────────────────────

def run(num_workers: int, batch_size: int = 1000, v2: bool = False, v3: bool = False) -> None:
    global _V2_ENABLED, _V3_ENABLED
    _V2_ENABLED = v2 or v3
    _V3_ENABLED = v3

    if v3:
        output_csv = OUTPUT_CSV_V3
    elif v2:
        output_csv = OUTPUT_CSV_V2
    else:
        output_csv = OUTPUT_CSV

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if v3:
        tag = "V3 (67 features)"
    elif v2:
        tag = "V2 (52 features)"
    else:
        tag = "V1 (33 features)"

    print(f"[{tag}] Loading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows")

    rows = df.to_dict("records")
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    print(f"Extracting features with {num_workers} workers, {len(batches)} batches …")
    t0 = time.time()

    all_features: list[dict] = []
    done = 0

    with Pool(num_workers, initializer=_init_worker, initargs=(v2 or v3, v3)) as pool:
        for batch_result in pool.imap(_extract_batch, batches):
            all_features.extend(batch_result)
            done += 1
            if done % 10 == 0 or done == len(batches):
                elapsed = time.time() - t0
                pct = done / len(batches) * 100
                print(
                    f"\r  [{done}/{len(batches)}] {pct:.1f}%  |  "
                    f"{len(all_features):,} rows  |  {elapsed:.0f}s",
                    end="", flush=True,
                )

    print()

    df_out = pd.DataFrame(all_features)
    df_out.to_csv(output_csv, index=False)

    elapsed = time.time() - t0
    feature_cols = [c for c in df_out.columns if c != "label"]
    print(f"\nFeatures → {output_csv}  ({len(df_out):,} rows, {len(feature_cols)} features)")
    print(f"Time: {elapsed:.1f}s")

    print(f"\n=== Feature Summary ===")
    for col in feature_cols:
        lo, hi = df_out[col].min(), df_out[col].max()
        print(f"  {col:40s}  {str(df_out[col].dtype):7s}  [{lo}, {hi}]")

    print(f"\n=== Label Distribution ===")
    print(df_out["label"].value_counts().to_string())


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract chess features from labeled moves.")
    parser.add_argument(
        "-w", "--workers", type=int, default=6,
        help="Number of parallel workers (default: 6)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1000,
        help="Rows per batch (default: 1000)",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="Include tactical features (groups 8-12) and output to features_v2.csv",
    )
    parser.add_argument(
        "--v3", action="store_true",
        help="Include look-ahead features (groups 8-15) and output to features_v3.csv",
    )
    args = parser.parse_args()
    run(num_workers=args.workers, batch_size=args.batch_size, v2=args.v2, v3=args.v3)


if __name__ == "__main__":
    main()
