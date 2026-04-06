"""
Filter and sample games from a Lichess .pgn.zst file.

Criteria (from docs/02-dados/filtragem-e-amostragem.md):
  - Both players rated 1400–1700 (Lichess)
  - TimeControl in {180+0, 180+2, 300+0, 300+3, 600+0, 600+5}
  - Termination: Normal
  - Standard variant only
  - Random sampling with configurable seed/rate
  - Extract mid-game moves (full-move 8–40)

Output: CSV with one row per half-move, ready for Stockfish labeling.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn

from pgn_stream import stream_games

ALLOWED_TC = {"180+0", "180+2", "300+0", "300+3", "600+0", "600+5"}

MOVE_LO = 8
MOVE_HI = 40

CSV_COLUMNS = [
    "game_id",
    "game_site",
    "white_elo",
    "black_elo",
    "time_control",
    "result",
    "move_number",
    "color",
    "fen_before",
    "move_uci",
    "move_san",
]


@dataclass
class FilterStats:
    total_scanned: int = 0
    rejected_rating: int = 0
    rejected_tc: int = 0
    rejected_termination: int = 0
    rejected_variant: int = 0
    rejected_sample: int = 0
    accepted: int = 0
    total_moves: int = 0
    elo_values: list[int] = field(default_factory=list)


def rating_ok(game: chess.pgn.Game, lo: int = 1400, hi: int = 1700) -> bool:
    try:
        w = int(game.headers["WhiteElo"])
        b = int(game.headers["BlackElo"])
        return lo <= w <= hi and lo <= b <= hi
    except (KeyError, ValueError):
        return False


def time_control_ok(game: chess.pgn.Game) -> bool:
    return game.headers.get("TimeControl", "") in ALLOWED_TC


def termination_ok(game: chess.pgn.Game) -> bool:
    return game.headers.get("Termination", "") == "Normal"


def variant_ok(game: chess.pgn.Game) -> bool:
    variant = game.headers.get("Variant", "Standard")
    return variant == "Standard"


def extract_midgame_moves(
    game: chess.pgn.Game,
    game_id: int,
    lo: int = MOVE_LO,
    hi: int = MOVE_HI,
) -> list[dict]:
    """Extract half-moves where full-move number is in [lo, hi]."""
    rows = []
    headers = game.headers
    site = headers.get("Site", "")
    white_elo = headers.get("WhiteElo", "")
    black_elo = headers.get("BlackElo", "")
    tc = headers.get("TimeControl", "")
    result = headers.get("Result", "")

    board = game.board()
    for node in game.mainline():
        move = node.move
        full_move = board.fullmove_number
        color = "white" if board.turn == chess.WHITE else "black"

        if full_move > hi:
            break
        if full_move >= lo:
            fen_before = board.fen()
            san = board.san(move)
            rows.append(
                {
                    "game_id": game_id,
                    "game_site": site,
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "time_control": tc,
                    "result": result,
                    "move_number": full_move,
                    "color": color,
                    "fen_before": fen_before,
                    "move_uci": move.uci(),
                    "move_san": san,
                }
            )

        board.push(move)

    return rows


def filter_and_sample(
    pgn_path: Path,
    output_path: Path,
    sample_rate: float,
    seed: int,
    max_games: int | None,
) -> FilterStats:
    random.seed(seed)
    stats = FilterStats()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        t0 = time.time()

        for game in stream_games(pgn_path):
            stats.total_scanned += 1

            if not variant_ok(game):
                stats.rejected_variant += 1
                continue

            if not rating_ok(game):
                stats.rejected_rating += 1
                continue

            if not time_control_ok(game):
                stats.rejected_tc += 1
                continue

            if not termination_ok(game):
                stats.rejected_termination += 1
                continue

            if random.random() >= sample_rate:
                stats.rejected_sample += 1
                continue

            stats.accepted += 1
            stats.elo_values.append(int(game.headers["WhiteElo"]))
            stats.elo_values.append(int(game.headers["BlackElo"]))

            rows = extract_midgame_moves(game, stats.accepted)
            stats.total_moves += len(rows)
            for row in rows:
                writer.writerow(row)

            if stats.total_scanned % 50_000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  scanned {stats.total_scanned:,} | "
                    f"accepted {stats.accepted:,} | "
                    f"moves {stats.total_moves:,} | "
                    f"{elapsed:.0f}s",
                    file=sys.stderr,
                )

            if max_games and stats.accepted >= max_games:
                print(
                    f"  reached max_games={max_games}, stopping.",
                    file=sys.stderr,
                )
                break

    return stats


def print_stats(stats: FilterStats) -> None:
    print("\n=== Filter Statistics ===")
    print(f"Total games scanned:      {stats.total_scanned:,}")
    print(f"Rejected (variant):       {stats.rejected_variant:,}")
    print(f"Rejected (rating):        {stats.rejected_rating:,}")
    print(f"Rejected (time control):  {stats.rejected_tc:,}")
    print(f"Rejected (termination):   {stats.rejected_termination:,}")
    print(f"Rejected (sampling):      {stats.rejected_sample:,}")
    print(f"Accepted games:           {stats.accepted:,}")
    print(f"Total mid-game moves:     {stats.total_moves:,}")
    if stats.accepted > 0:
        avg_moves = stats.total_moves / stats.accepted
        print(f"Avg moves per game:       {avg_moves:.1f}")
    if stats.elo_values:
        print(
            f"Elo range in dataset:     {min(stats.elo_values)}–{max(stats.elo_values)}"
        )
        avg_elo = sum(stats.elo_values) / len(stats.elo_values)
        print(f"Elo mean:                 {avg_elo:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Lichess PGN games and extract mid-game moves."
    )
    parser.add_argument("pgn", type=Path, help="Path to .pgn.zst file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/filtered/moves_filtered.csv"),
        help="Output CSV path (default: data/filtered/moves_filtered.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.10,
        help="Fraction of qualifying games to keep (default: 0.10)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=3000,
        help="Stop after this many accepted games (default: 3000)",
    )
    args = parser.parse_args()

    if not args.pgn.is_file():
        raise SystemExit(f"File not found: {args.pgn}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"PGN source:   {args.pgn}")
    print(f"Output:       {args.output}")
    print(f"Seed:         {args.seed}")
    print(f"Sample rate:  {args.sample_rate}")
    print(f"Max games:    {args.max_games}")
    print()

    stats = filter_and_sample(
        pgn_path=args.pgn,
        output_path=args.output,
        sample_rate=args.sample_rate,
        seed=args.seed,
        max_games=args.max_games,
    )

    print_stats(stats)


if __name__ == "__main__":
    main()
