"""
Stream games from a Lichess .pgn.zst file without loading the whole file in RAM.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Iterator
from pathlib import Path

import chess.pgn
import zstandard as zstd


def stream_games(filepath: str | Path) -> Iterator[chess.pgn.Game]:
    """Iterate over games from a .pgn.zst file; memory usage stays bounded."""
    filepath = Path(filepath)
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            yield game


def print_headers_sample(filepath: Path, limit: int) -> None:
    """Read up to `limit` games and print selected PGN headers (stdout)."""
    n = 0
    for game in stream_games(filepath):
        if n >= limit:
            break
        headers = game.headers
        we = headers.get("WhiteElo", "?")
        be = headers.get("BlackElo", "?")
        tc = headers.get("TimeControl", "?")
        print(f"{n + 1:4d}  WhiteElo={we}  BlackElo={be}  TimeControl={tc}")
        n += 1
    if n == 0:
        print("No games read (empty file or invalid PGN?).", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nPrinted headers for {n} game(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream PGN games from a .pgn.zst file and print header sample."
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to .pgn.zst",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=100,
        help="Number of games to read for the header sample (default: 100).",
    )
    args = parser.parse_args()
    if not args.filepath.is_file():
        raise SystemExit(f"File not found: {args.filepath}")
    print_headers_sample(args.filepath, args.limit)


if __name__ == "__main__":
    main()
