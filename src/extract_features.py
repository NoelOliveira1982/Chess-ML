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


# ── Row-level extraction ──────────────────────────────────────────

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
    feats["label"] = row["label"]
    return feats


def _extract_batch(rows: list[dict]) -> list[dict]:
    return [_extract_row(r) for r in rows]


# ── Main pipeline ─────────────────────────────────────────────────

def run(num_workers: int, batch_size: int = 1000) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows")

    rows = df.to_dict("records")
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    print(f"Extracting features with {num_workers} workers, {len(batches)} batches …")
    t0 = time.time()

    all_features: list[dict] = []
    done = 0

    with Pool(num_workers) as pool:
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
    df_out.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t0
    feature_cols = [c for c in df_out.columns if c != "label"]
    print(f"\nFeatures → {OUTPUT_CSV}  ({len(df_out):,} rows, {len(feature_cols)} features)")
    print(f"Time: {elapsed:.1f}s")

    print(f"\n=== Feature Summary ===")
    for col in feature_cols:
        lo, hi = df_out[col].min(), df_out[col].max()
        print(f"  {col:30s}  {str(df_out[col].dtype):7s}  [{lo}, {hi}]")

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
    args = parser.parse_args()
    run(num_workers=args.workers, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
