"""Turn downloaded raw files into the repo catalog. Raw files stay in data/raw.

uv run python scripts/organize_sources.py
"""

import argparse
import json
from pathlib import Path

from writing_agent.acquisition import compile_downloads
from writing_agent.catalog import fingerprint, save_json, validate_catalog
from writing_agent.text_clean import opening_prose

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "research"
DESTINATION = ROOT / "data" / "processed" / "research"


def _author(title: str) -> tuple[str, str]:
    if " by " not in title:
        return title, "unknown"
    name, author = title.rsplit(" by ", 1)
    return name.strip(), author.strip() or "unknown"


def pg19_openings() -> list[dict]:
    train = RAW / "pg19" / "train"
    metadata = RAW / "pg19" / "metadata.csv"
    if not train.is_dir() or not metadata.exists():
        return []
    held_out = {"11", "84", "1661", "113"}
    for split in ("validation", "test"):
        listing = RAW / "pg19" / f"{split}_files.txt"
        if listing.exists():
            held_out.update(
                line.strip().rsplit("/", 1)[-1].removesuffix(".txt")
                for line in listing.read_text().splitlines()
                if line.strip()
            )
    titles = {}
    for line in metadata.read_text().splitlines():
        if not line.strip():
            continue
        book_id, rest = line.split(",", 1)
        title, _year, url = rest.rsplit(",", 2)
        titles[book_id] = (title, url)
    records = []
    for path in sorted(train.glob("*.txt")):
        book_id = path.stem
        if book_id in held_out or book_id not in titles:
            continue
        title, url = titles[book_id]
        work_title, author = _author(title)
        try:
            excerpt = opening_prose(path.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        records.append(
            {
                "id": f"pg19-{book_id}-opening",
                "text": excerpt,
                "sha256": fingerprint(excerpt),
                "work_id": f"pg19-{book_id}",
                "author_id": author.casefold().replace(" ", "-"),
                "author": author,
                "title": work_title,
                "provenance": "human",
                "role": "train",
                "parents": [],
                "transformations": [],
                "upstream_split": "train",
                "url": url,
                "raw_file": str(path.relative_to(ROOT)),
                "raw_sha256": fingerprint(path.read_bytes()),
                "terms": {
                    "license": "Apache-2.0 dataset packaging; US public-domain Gutenberg text",
                    "evidence": "https://github.com/google-deepmind/pg19",
                },
            }
        )
    return records


def main() -> None:
    argparse.ArgumentParser(
        description="Catalog downloaded sources under data/processed"
    ).parse_args()
    imported = compile_downloads(RAW, DESTINATION)
    books = pg19_openings()
    if books:
        validate_catalog(books)
        save_json(DESTINATION / "pg19-openings.json", books)
    summary = {
        "compiled_sources": len(imported["catalog"]),
        "failures": imported["failures"],
        "pg19_openings": len(books),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
