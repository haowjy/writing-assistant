"""Pinned public releases and normalization into the shared source catalog."""

import csv
import json
import zipfile
from pathlib import Path

from writing_agent.catalog import download_source, fingerprint, read_jsonl, save_json
from writing_agent.text_clean import strip_gutenberg

TELL_REV = "e4910ea1d2bae82efcaf8ba9fde50ab3a419320e"
HANNA_REV = "282f27536a5d05ad4ce14298abcd70c45668fed2"
IFEVAL_REV = "26d8ccdab6fec61b5c83ad6327ea8bda9e580288"
EQBENCH_LONGFORM_REV = "34f60a028c3f973c19cde98dc5a9e8f9875a87e3"
EQBENCH_LONGFORM_FILES = (
    "README.md",
    "data/criteria_weights.json",
    "data/longform_creative_writing_criteria_chapter.txt",
    "data/longform_creative_writing_criteria_final.txt",
    "data/longform_creative_writing_judging_prompt_chapter.txt",
    "data/longform_creative_writing_judging_prompt_final.txt",
    "data/longform_creative_writing_prompts_minimalist.json",
    "data/longform_negative_criteria_chapter.txt",
    "data/longform_negative_criteria_final.txt",
    "data/prompt1.txt",
    "data/prompt2.txt",
    "data/prompt3.txt",
    "data/prompt4.txt",
    "data/prompt5.txt",
    "data/prompt_chapter_first.txt",
    "data/prompt_chapter_intermediate.txt",
    "data/prompt_chapter_last.txt",
)
BOOKS = {
    84: ("Frankenstein", "Mary Shelley"),
    1661: ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    11: ("Alice's Adventures in Wonderland", "Lewis Carroll"),
}


def acquire_sources(
    destination: Path,
    *,
    selections=("tell_me_a_story", "hanna", "ifeval", "gutenberg", "eqbench_longform"),
) -> dict:
    """Acquire small releases / three books; do not run benchmarks or fetch model weights."""
    urls = []
    if "tell_me_a_story" in selections:
        root = f"https://raw.githubusercontent.com/google-deepmind/tell_me_a_story/{TELL_REV}"
        urls.extend(
            ("tell_me_a_story/" + name, root + "/" + name)
            for name in ("README.md", "LICENSE", "keys.zip")
        )
        urls.extend(
            (
                f"tell_me_a_story/{split}_encrypted.jsonl",
                f"https://storage.googleapis.com/tell-me-a-story/tell-me-a-story-{split}"
                "_encrypted.jsonl",
            )
            for split in ("train", "validation", "test")
        )
    if "hanna" in selections:
        root = f"https://raw.githubusercontent.com/dig-team/hanna-benchmark-asg/{HANNA_REV}"
        urls.extend(
            ("hanna/" + name, root + "/" + name)
            for name in (
                "README.md",
                "LICENSE",
                "hanna_stories_annotations.csv",
                "hanna_llm_stories.csv",
                "hanna_metric_scores_llm.csv",
                "user_study.csv",
            )
        )
    if "ifeval" in selections:
        root = f"https://raw.githubusercontent.com/google-research/google-research/{IFEVAL_REV}"
        urls.extend(
            ("ifeval/" + name, root + "/instruction_following_eval/" + name)
            for name in (
                "README.md",
                "data/input_data.jsonl",
                "instructions.py",
                "instructions_registry.py",
                "evaluation_main.py",
                "instructions_util.py",
            )
        )
        urls.append(("ifeval/LICENSE", root + "/LICENSE"))
    if "gutenberg" in selections:
        urls.extend(
            (f"gutenberg/{book}.txt", f"https://www.gutenberg.org/ebooks/{book}.txt.utf-8")
            for book in BOOKS
        )
    if "eqbench_longform" in selections:
        root = (
            "https://raw.githubusercontent.com/EQ-bench/longform-writing-bench/"
            + EQBENCH_LONGFORM_REV
        )
        urls.extend(
            ("eqbench-longform/" + name, root + "/" + name) for name in EQBENCH_LONGFORM_FILES
        )
    if set(selections) - {
        "tell_me_a_story",
        "hanna",
        "ifeval",
        "gutenberg",
        "eqbench_longform",
    }:
        raise ValueError("Unknown source selection")
    receipts = []
    for name, url in urls:
        try:
            receipt = download_source(url, destination / name)
            receipts.append({"path": name, "status": "acquired", **receipt})
        except (OSError, ValueError) as exc:
            receipts.append({"path": name, "url": url, "status": "failed", "error": str(exc)})
    result = {"receipts": receipts}
    save_json(destination / "acquisition.json", result)
    return result


def _source(key, text, *, provenance, work, role="development", **metadata):
    return {
        "id": key,
        "text": text,
        "sha256": fingerprint(text),
        "work_id": work,
        "provenance": provenance,
        "role": role,
        "parents": [],
        "review_status": "pending",
        "transformations": [],
        **metadata,
    }


def compile_downloads(raw: Path, destination: Path) -> dict:
    """Keep original rows linked; separate prompt, human story and generated artifacts."""
    catalog, failures, inspections = [], [], []
    tell = raw / "tell_me_a_story"
    if (tell / "keys.zip").exists():
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            with zipfile.ZipFile(tell / "keys.zip") as archive:
                private = next(n for n in archive.namelist() if n.endswith("private_key.pem"))
                symmetric = next(n for n in archive.namelist() if n.endswith("skey.key"))
                key = serialization.load_pem_private_key(archive.read(private), password=None)
                symmetric_key = key.decrypt(
                    archive.read(symmetric),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )
            for split in ("train", "validation", "test"):
                encrypted = tell / f"{split}_encrypted.jsonl"
                if not encrypted.exists():
                    continue
                plaintext = Fernet(symmetric_key).decrypt(encrypted.read_bytes())
                rows = [
                    json.loads(line) for line in plaintext.decode().splitlines() if line.strip()
                ]
                inspections.append(
                    {
                        "source": "tell_me_a_story",
                        "split": split,
                        "rows": len(rows),
                        "fields": sorted(rows[0]),
                    }
                )
                for row in rows:
                    identity = "tmas-" + split + "-" + str(row["example_id"])
                    common = {
                        "revision": TELL_REV,
                        "upstream_split": split,
                        "url": "https://github.com/google-deepmind/tell_me_a_story",
                        "terms": {
                            "license": "CC-BY-4.0",
                            "evidence": "README.md",
                            "training": "not_selected",
                            "evaluation": "allowed",
                        },
                        "raw_file": str(encrypted),
                        "raw_sha256": fingerprint(encrypted.read_bytes()),
                    }
                    catalog.append(
                        _source(
                            identity + "-prompt",
                            row["inputs"],
                            provenance="unknown",
                            work=identity,
                            artifact_type="prompt",
                            **common,
                        )
                    )
                    catalog.append(
                        _source(
                            identity + "-story",
                            row["targets"],
                            provenance="human",
                            work=identity,
                            artifact_type="prose_reference",
                            **common,
                        )
                    )
        except (ImportError, OSError, ValueError, StopIteration) as exc:
            failures.append({"source": "tell_me_a_story", "error": str(exc)})
    hanna = raw / "hanna/hanna_stories_annotations.csv"
    if hanna.exists():
        with hanna.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        inspections.append({"source": "hanna", "rows": len(rows), "fields": sorted(rows[0])})
        grouped = {}
        for row in rows:
            grouped.setdefault(row["Story ID"], []).append(row)
        seen_humans = set()
        for key, annotations in grouped.items():
            row = annotations[0]
            work = "writingprompts-" + fingerprint(row["Prompt"])[:16]
            common = {
                "revision": HANNA_REV,
                "upstream_split": "evaluation",
                "url": "https://github.com/dig-team/hanna-benchmark-asg",
                "terms": {
                    "license": "MIT repository; upstream WritingPrompts terms retained",
                    "training": "unresolved",
                    "redistribution": "unresolved",
                    "evaluation": "local_research",
                },
                "raw_file": str(hanna),
                "prompt": row["Prompt"],
                "artifact_type": "prose",
                "annotations": [
                    {
                        k: r[k]
                        for k in (
                            "Relevance",
                            "Coherence",
                            "Empathy",
                            "Surprise",
                            "Engagement",
                            "Complexity",
                        )
                    }
                    for r in annotations
                ],
            }
            catalog.append(
                _source(
                    "hanna-" + key,
                    row["Story"],
                    work=work,
                    provenance="human" if row["Model"].casefold() == "human" else "synthetic",
                    generator=row["Model"],
                    **common,
                )
            )
            if work not in seen_humans:
                seen_humans.add(work)
                catalog.append(
                    _source(
                        work + "-reference",
                        row["Human"],
                        provenance="human",
                        work=work,
                        **{**common, "artifact_type": "prose_reference"},
                    )
                )
    for filename in ("hanna_llm_stories.csv", "hanna_metric_scores_llm.csv", "user_study.csv"):
        path = raw / "hanna" / filename
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            extra_rows = list(csv.DictReader(handle))
        inspections.append(
            {
                "source": "hanna/" + filename,
                "rows": len(extra_rows),
                "fields": sorted(extra_rows[0]) if extra_rows else [],
            }
        )
        if filename == "hanna_llm_stories.csv":
            for index, row in enumerate(extra_rows):
                catalog.append(
                    _source(
                        f"hanna-extra-{index}",
                        row["Story"],
                        provenance="synthetic",
                        work="writingprompts-" + fingerprint(row["Prompt"])[:16],
                        revision=HANNA_REV,
                        artifact_type="prose",
                        prompt=row["Prompt"],
                        generator=row["Model"],
                        raw_file=str(path),
                        upstream_split="evaluation",
                        terms={
                            "license": "MIT repository; upstream WritingPrompts terms retained",
                            "evaluation": "local_research",
                            "training": "unresolved",
                        },
                    )
                )
    ifeval = raw / "ifeval/data/input_data.jsonl"
    if ifeval.exists():
        rows = read_jsonl(ifeval)
        inspections.append({"source": "ifeval", "rows": len(rows), "fields": sorted(rows[0])})
        for row in rows:
            catalog.append(
                _source(
                    "ifeval-" + str(row["key"]),
                    row["prompt"],
                    provenance="unknown",
                    work="ifeval-" + str(row["key"]),
                    role="regression",
                    artifact_type="instruction_task",
                    original=row,
                    revision=IFEVAL_REV,
                    upstream_split="test",
                    url="https://github.com/google-research/google-research",
                    terms={
                        "license": "Apache-2.0",
                        "evaluation": "allowed",
                        "training": "not_selected",
                    },
                )
            )
    for book, (title, author) in BOOKS.items():
        path = raw / f"gutenberg/{book}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        # Preserve the raw file, including the license. Derived reading text excludes boilerplate.
        try:
            body = strip_gutenberg(text)
        except ValueError as exc:
            failures.append({"source": str(book), "error": str(exc)})
            continue
        catalog.append(
            _source(
                f"gutenberg-{book}",
                body,
                provenance="human",
                work=f"gutenberg-{book}",
                author_id=author,
                title=title,
                artifact_type="book",
                upstream_split="unspecified",
                revision=fingerprint(text),
                raw_file=str(path),
                url=f"https://www.gutenberg.org/ebooks/{book}",
                terms={
                    "license": "Project Gutenberg license in raw file; US public domain",
                    "evaluation": "local_research",
                    "redistribution": "review_terms",
                },
            )
        )
        inspections.append({"source": f"gutenberg-{book}", "characters": len(body), "title": title})
    result = {"catalog": catalog, "inspections": inspections, "failures": failures}
    save_json(destination / "sources.json", catalog)
    save_json(destination / "inspection.json", {k: v for k, v in result.items() if k != "catalog"})
    return result
