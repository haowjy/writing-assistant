"""Download sources into data/raw. Does not commit them.

uv run python scripts/fetch_sources.py research
uv run python scripts/fetch_sources.py gutenberg 289
uv run python scripts/fetch_sources.py pg19
uv run python scripts/fetch_sources.py tasks
"""

import argparse
import urllib.request
from pathlib import Path

from writing_agent.acquisition import acquire_sources
from writing_agent.catalog import fingerprint

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "research"
PG19_ROOT = "https://storage.googleapis.com/deepmind-gutenberg/"
PG19_LIST = (
    "https://huggingface.co/datasets/deepmind/pg19/resolve/"
    "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b/data/train_files.txt"
)
TASKS = ROOT / "data" / "tasks" / "sources.json"


def _stream(url: str, destination: Path, *, limit: int = 80_000_000) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "WritingResearch/0.1"})
    written = 0
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Download exceeds {limit} bytes: {url}")
            handle.write(chunk)
    temporary.replace(destination)
    return written


def research() -> None:
    result = acquire_sources(RAW)
    failed = [item for item in result["receipts"] if item["status"] != "acquired"]
    print(f"research receipts: {len(result['receipts'])}; failed: {len(failed)}")
    for item in failed:
        print(f"  {item['path']}: {item.get('error')}")


def gutenberg(book_ids: list[str]) -> None:
    for book_id in book_ids:
        destination = RAW / "gutenberg" / f"{book_id}.txt"
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        if destination.exists() and destination.stat().st_size > 0:
            print(f"skip {destination}")
            continue
        size = _stream(url, destination)
        print(f"gutenberg {book_id}: {size} bytes sha256 {fingerprint(destination.read_bytes())}")


def pg19() -> None:
    listing = RAW / "pg19" / "train_files.txt"
    if not listing.exists():
        _stream(PG19_LIST, listing, limit=2_000_000)
    names = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    print(f"pg19 train files: {len(names)}")
    for index, relative in enumerate(names, 1):
        destination = RAW / "pg19" / "train" / relative.rsplit("/", 1)[-1]
        if destination.exists() and destination.stat().st_size > 0:
            continue
        _stream(PG19_ROOT + relative, destination)
        if index % 500 == 0 or index == len(names):
            print(f"{index}/{len(names)}")


def tasks() -> None:
    import json

    sources = json.loads(TASKS.read_text())
    gutenberg_ids = sorted(
        {
            item["url"].rstrip("/").rsplit("/", 1)[-1]
            for item in sources
            if "gutenberg.org/ebooks/" in item.get("url", "")
        }
    )
    gutenberg(gutenberg_ids)
    research()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sources into data/raw")
    parser.add_argument("command", choices=("research", "gutenberg", "pg19", "tasks"))
    parser.add_argument("ids", nargs="*")
    args = parser.parse_args()
    if args.command == "research":
        research()
    elif args.command == "gutenberg":
        if not args.ids:
            raise SystemExit("Pass one or more Gutenberg ebook ids")
        gutenberg(args.ids)
    elif args.command == "pg19":
        pg19()
    else:
        tasks()


if __name__ == "__main__":
    main()
