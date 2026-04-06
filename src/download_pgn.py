"""
Download Lichess open-database PGN files (.pgn.zst) with console progress.

Default: lichess_db_standard_rated_2015-01.pgn.zst (~270 MB compressed).
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_2015-01.pgn.zst"
)
DEFAULT_OUT = Path("data/raw") / "lichess_db_standard_rated_2015-01.pgn.zst"
CHUNK_SIZE = 1024 * 256  # 256 KiB


def download_file(url: str, dest: Path, chunk_size: int = CHUNK_SIZE) -> None:
    """Stream download from url to dest; prints percentage progress."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "ChessMoveClassifier/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0
            with open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(100.0, 100.0 * downloaded / total)
                        mb_done = downloaded / (1024 * 1024)
                        mb_tot = total / (1024 * 1024)
                        sys.stdout.write(
                            f"\r  {pct:5.1f}%  ({mb_done:.1f} / {mb_tot:.1f} MiB)"
                        )
                        sys.stdout.flush()
            if total:
                sys.stdout.write("\n")
    except urllib.error.URLError as e:
        raise SystemExit(f"Download failed: {e}") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Lichess .pgn.zst database file.")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Direct HTTPS URL to the .pgn.zst file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path (default: data/raw/lichess_db_standard_rated_2015-01.pgn.zst).",
    )
    args = parser.parse_args()

    print(f"URL: {args.url}")
    print(f"Destination: {args.output.resolve()}")
    download_file(args.url, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
