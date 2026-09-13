"""Source lineage, local story imports, and reproducible JSON artifacts."""

import hashlib
import json
import posixpath
import re
import urllib.request
import zipfile
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree

PROVENANCE = {"human", "synthetic", "half_synthetic", "synthetic_fanfic", "unknown"}
ROLES = {"train", "development", "final_eval", "regression"}


def fingerprint(value) -> str:
    content = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    )
    return hashlib.sha256(content).hexdigest()


def save_json(path: Path, value) -> None:
    """Atomic replacement; never leave a partially written cache or result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_relative(path: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError("Expected a nonempty POSIX relative path")
    if path.startswith("/") or ".." in path.split("/") or path in {".", ""}:
        raise ValueError("Unsafe relative path")
    return path


def validate_catalog(records: list[dict]) -> dict[str, str]:
    """Return connected source groups; reject lineage, role and upstream split leaks."""
    indexed = {r["id"]: r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate source IDs")
    parents = {key: key for key in indexed}

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    def merge(a, b):
        a, b = sorted((root(a), root(b)))
        parents[b] = a

    identities = {}
    for record in records:
        key = record["id"]
        if record["provenance"] not in PROVENANCE or record["role"] not in ROLES:
            raise ValueError("Unknown provenance or role")
        if not record.get("work_id") or not record.get("sha256") or not record.get("terms"):
            raise ValueError("Source needs work identity, hash and terms evidence")
        if record["role"] == "train" and record.get("upstream_split") in {
            "test",
            "validation",
            "dev",
        }:
            raise ValueError("Upstream held-out source cannot become training data")
        for field in ("work_id", "author_id", "series_id", "sha256"):
            value = record.get(field)
            if value:
                identity = (field, value)
                if identity in identities:
                    merge(key, identities[identity])
                identities[identity] = key
        for parent in record.get("parents", []):
            if parent not in indexed:
                raise ValueError(f"Missing parent source {parent}")
            merge(key, parent)
    # A cycle obscures which artifact preceded a transformation.
    visited, active = set(), set()

    def visit(key):
        if key in active:
            raise ValueError("Cyclic source lineage")
        if key in visited:
            return
        active.add(key)
        for parent in indexed[key].get("parents", []):
            visit(parent)
        active.remove(key)
        visited.add(key)

    for key in indexed:
        visit(key)
    roles = {}
    for record in records:
        group = root(record["id"])
        if group in roles and roles[group] != record["role"]:
            raise ValueError(f"Source group crosses research roles: {group}")
        roles[group] = record["role"]
    return {key: root(key) for key in indexed}


def select_sources(records: list[dict], *, provenance=None, role=None) -> list[dict]:
    validate_catalog(records)  # Check globally before filtering away conflicts.
    return [
        r
        for r in records
        if (provenance is None or r["provenance"] in provenance)
        and (role is None or r["role"] == role)
    ]


class _StoryHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav"}:
            self.hidden += 1
        if not self.hidden and tag in {"p", "div", "br", "h1", "h2", "h3", "li"}:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav"}:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in {"p", "div", "h1", "h2", "h3", "li"}:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def html_text(text: str) -> str:
    parser = _StoryHTML()
    parser.feed(text)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", "".join(parser.parts)).strip()


def story_text(path: Path) -> str:
    """EPUB follows its spine; ZIP members are read, never extracted to disk."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8-sig")
    if suffix in {".html", ".htm", ".xhtml"}:
        return html_text(path.read_text(encoding="utf-8-sig"))
    if suffix != ".epub":
        raise ValueError(f"Unsupported story format: {suffix}")
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 100_000_000:
            raise ValueError("EPUB exceeds uncompressed import budget")
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        opf = next(n.attrib["full-path"] for n in container.iter() if n.tag.endswith("rootfile"))
        package = ElementTree.fromstring(archive.read(safe_relative(opf)))
        items = {
            n.attrib["id"]: n.attrib["href"] for n in package.iter() if n.tag.endswith("}item")
        }
        chapters = []
        for node in package.iter():
            if node.tag.endswith("}itemref") and node.attrib.get("linear", "yes") != "no":
                member = posixpath.normpath(
                    posixpath.join(posixpath.dirname(opf), items[node.attrib["idref"]])
                )
                chapters.append(html_text(archive.read(safe_relative(member)).decode("utf-8-sig")))
        return "\n\n".join(chapters)


def import_story(path: Path, metadata: dict, destination: Path) -> dict:
    """Import with caller-supplied identity/permissions; review normalization separately."""
    raw = path.read_bytes()
    text = story_text(path)
    record = {
        **metadata,
        "sha256": fingerprint(raw),
        "text": text,
        "text_sha256": fingerprint(text),
        "transformations": [
            *metadata.get("transformations", []),
            {"operation": "story_text", "version": 1},
        ],
        "acquired_at": datetime.now(UTC).isoformat(),
    }
    if record.get("provenance") not in PROVENANCE:
        raise ValueError("Explicit provenance is required")
    destination.mkdir(parents=True, exist_ok=True)
    raw_target = destination / (record["sha256"] + ".source" + path.suffix.lower())
    raw_target.write_bytes(raw)
    record["raw_path"] = str(raw_target.resolve())
    target = destination / (record["sha256"] + ".txt")
    target.write_text(text, encoding="utf-8")
    record["text_path"] = str(target.resolve())
    return record


def download_source(url: str, destination: Path, *, max_bytes=25_000_000) -> dict:
    """Bounded, atomic raw acquisition with hashes; existing files are reused."""
    receipt = destination.with_suffix(destination.suffix + ".receipt.json")
    if destination.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["url"] == url and fingerprint(destination.read_bytes()) == saved["sha256"]:
            return saved
    request = urllib.request.Request(url, headers={"User-Agent": "WritingResearch/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("Download exceeds acquisition budget")
        result = {
            "url": url,
            "resolved_url": response.url,
            "bytes": len(raw),
            "sha256": fingerprint(raw),
            "acquired_at": datetime.now(UTC).isoformat(),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(raw)
    temporary.replace(destination)
    save_json(receipt, result)
    return result


def overlap_audit(records: list[dict], *, threshold=0.85) -> list[dict]:
    """Candidate discovery via bottom-12 shingle hashes, verified by full 5-gram Jaccard.

    This is a near-duplicate heuristic, not proof of absence of contamination.
    Identity grouping remains the primary boundary for translations and derivatives.
    """
    index, signatures, findings = {}, {}, []
    for record in records:
        words = re.findall(r"\w+", record.get("text", "").casefold())
        shingles = {tuple(words[i : i + 5]) for i in range(len(words) - 4)}
        if not shingles:
            continue
        signature = sorted(hashlib.sha256(" ".join(s).encode()).digest()[:8] for s in shingles)[:12]
        candidates = set().union(*(index.get(h, set()) for h in signature))
        for candidate in candidates:
            other, previous = signatures[candidate]
            similarity = len(shingles & previous) / len(shingles | previous)
            if similarity >= threshold:
                findings.append(
                    {
                        "left": candidate,
                        "right": record["id"],
                        "jaccard_5gram": similarity,
                        "crosses_role": other["role"] != record["role"],
                    }
                )
        signatures[record["id"]] = (record, shingles)
        for h in signature:
            index.setdefault(h, set()).add(record["id"])
    return findings
