"""
Label chess moves using Stockfish evaluation delta.

For each move, computes:
    score_before = Stockfish eval of position before the move (player's perspective)
    score_after  = Stockfish eval of position after  the move (opponent's perspective)
    delta_cp     = -score_after - score_before  (quality from player's perspective)

Labels:
    bom       — delta >= -50 cp   (acceptable move)
    ruim      — delta <= -150 cp  (clear mistake)
    descartado — in between        (grey zone, excluded from final dataset)

Optimisations:
    - Eval caching: within a game, eval_after of move N = eval_before of move N+1
      (same position, same side to move), cutting total evaluations nearly in half.
    - Multiprocessing: multiple Stockfish engines run in parallel.
    - Checkpointing: progress is saved every CHECKPOINT_EVERY games.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool, current_process
from pathlib import Path

import chess
import chess.engine
import pandas as pd

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
HASH_MB = 64
THREADS_PER_ENGINE = 1

THRESHOLD_GOOD = -50
THRESHOLD_BAD = -150

_cfg = {"depth": 15}

INPUT_CSV = Path("data/filtered/moves_filtered.csv")
OUTPUT_DIR = Path("data/labeled")
OUTPUT_ALL = OUTPUT_DIR / "moves_all_scored.csv"
OUTPUT_LABELED = OUTPUT_DIR / "moves_labeled.csv"
CHECKPOINT_JSON = OUTPUT_DIR / "checkpoint.json"
CHECKPOINT_PARTIAL = OUTPUT_DIR / "partial_results.csv"
CHECKPOINT_EVERY = 100

CSV_OUT_COLUMNS = [
    "game_id", "game_site", "white_elo", "black_elo",
    "time_control", "result", "move_number", "color",
    "fen_before", "move_uci", "move_san",
    "score_before", "score_after", "delta_cp", "label",
]

# ── Worker engine (one per process) ────────────────────────────────

_engine: chess.engine.SimpleEngine | None = None


def _init_engine():
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    _engine.configure({"Hash": HASH_MB, "Threads": THREADS_PER_ENGINE})


def _eval(board: chess.Board) -> int:
    info = _engine.analyse(board, chess.engine.Limit(depth=_cfg["depth"]))  # type: ignore[union-attr]
    return info["score"].relative.score(mate_score=10000)


def _label_game(game_rows: list[dict]) -> list[dict]:
    """Score and label every move in one game. Uses eval caching."""
    results: list[dict] = []
    cached_score: int | None = None

    for row in game_rows:
        board = chess.Board(row["fen_before"])

        score_before = cached_score if cached_score is not None else _eval(board)

        move = chess.Move.from_uci(row["move_uci"])
        board.push(move)

        score_after = _eval(board)
        cached_score = score_after

        delta = -score_after - score_before

        if delta >= THRESHOLD_GOOD:
            label = "bom"
        elif delta <= THRESHOLD_BAD:
            label = "ruim"
        else:
            label = "descartado"

        out = dict(row)
        out["score_before"] = score_before
        out["score_after"] = score_after
        out["delta_cp"] = delta
        out["label"] = label
        results.append(out)

    return results


# ── Main pipeline ──────────────────────────────────────────────────

def _load_games(csv_path: Path) -> dict[int, list[dict]]:
    """Read CSV and group rows by game_id, sorted within each game."""
    df = pd.read_csv(csv_path)
    games: dict[int, list[dict]] = {}
    for _, row in df.iterrows():
        gid = int(row["game_id"])
        games.setdefault(gid, []).append(row.to_dict())

    color_order = {"white": 0, "black": 1}
    for gid in games:
        games[gid].sort(key=lambda r: (r["move_number"], color_order.get(r["color"], 2)))

    return games


def _load_checkpoint() -> tuple[set[int], list[dict]]:
    completed: set[int] = set()
    results: list[dict] = []
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON) as f:
            data = json.load(f)
        completed = set(data.get("completed_game_ids", []))
        if CHECKPOINT_PARTIAL.exists():
            results = pd.read_csv(CHECKPOINT_PARTIAL).to_dict("records")
        print(f"Checkpoint found: {len(completed)} games done, {len(results)} moves scored")
    return completed, results


def _save_checkpoint(completed: set[int], results: list[dict]):
    pd.DataFrame(results, columns=CSV_OUT_COLUMNS).to_csv(CHECKPOINT_PARTIAL, index=False)
    with open(CHECKPOINT_JSON, "w") as f:
        json.dump({"completed_game_ids": sorted(completed)}, f)


def _remove_checkpoint():
    for p in (CHECKPOINT_JSON, CHECKPOINT_PARTIAL):
        if p.exists():
            p.unlink()


def run(num_workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_CSV} …")
    games = _load_games(INPUT_CSV)
    game_ids = sorted(games.keys())
    total_games = len(game_ids)
    total_moves = sum(len(v) for v in games.values())
    print(f"Loaded {total_games} games, {total_moves} moves")

    completed, all_results = _load_checkpoint()
    remaining = [gid for gid in game_ids if gid not in completed]
    print(f"Remaining: {len(remaining)} games")

    if not remaining:
        print("Nothing to do — all games already processed.")
    else:
        print(f"Starting {num_workers} workers (Stockfish depth {_cfg['depth']}) …\n")
        work = [games[gid] for gid in remaining]
        done_count = 0
        t0 = time.time()

        with Pool(num_workers, initializer=_init_engine) as pool:
            for game_results in pool.imap(_label_game, work):
                gid = int(game_results[0]["game_id"])
                all_results.extend(game_results)
                completed.add(gid)
                done_count += 1

                if done_count % 10 == 0 or done_count == len(remaining):
                    elapsed = time.time() - t0
                    rate = done_count / elapsed
                    eta = (len(remaining) - done_count) / rate if rate > 0 else 0
                    pct = (len(completed)) / total_games * 100
                    print(
                        f"\r  [{len(completed)}/{total_games}] {pct:5.1f}%  |  "
                        f"{len(all_results):,} moves scored  |  "
                        f"{elapsed:.0f}s elapsed  |  ETA {eta:.0f}s",
                        end="", flush=True,
                    )

                if done_count % CHECKPOINT_EVERY == 0:
                    _save_checkpoint(completed, all_results)

        print()

    # ── Save final outputs ──────────────────────────────────────────
    df_all = pd.DataFrame(all_results, columns=CSV_OUT_COLUMNS)
    df_all.to_csv(OUTPUT_ALL, index=False)
    print(f"\nAll scored moves → {OUTPUT_ALL}  ({len(df_all)} rows)")

    df_labeled = df_all[df_all["label"].isin(["bom", "ruim"])].copy()
    df_labeled.to_csv(OUTPUT_LABELED, index=False)
    print(f"Labeled (bom/ruim) → {OUTPUT_LABELED}  ({len(df_labeled)} rows)")

    _remove_checkpoint()

    # ── Distribution summary ────────────────────────────────────────
    print("\n=== Label Distribution ===")
    counts = df_all["label"].value_counts()
    for label in ["bom", "ruim", "descartado"]:
        n = counts.get(label, 0)
        pct = n / len(df_all) * 100
        print(f"  {label:11s}  {n:>7,}  ({pct:.1f}%)")

    if len(df_labeled) > 0:
        bom_n = counts.get("bom", 0)
        ruim_n = counts.get("ruim", 0)
        ratio = bom_n / ruim_n if ruim_n > 0 else float("inf")
        print(f"\n  bom:ruim ratio = {ratio:.2f}:1")

    print(f"\nDelta (cp) statistics:")
    print(df_all["delta_cp"].describe().to_string())


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Label chess moves with Stockfish.")
    parser.add_argument(
        "-w", "--workers", type=int, default=6,
        help="Number of parallel Stockfish workers (default: 6)",
    )
    parser.add_argument(
        "-d", "--depth", type=int, default=None,
        help=f"Stockfish search depth (default: {_cfg['depth']})",
    )
    args = parser.parse_args()

    if args.depth is not None:
        _cfg["depth"] = args.depth

    run(num_workers=args.workers)


if __name__ == "__main__":
    main()
