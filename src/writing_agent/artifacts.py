"""Locate requested prose and inspect the supported Markdown wiki format."""

import posixpath
import re
from urllib.parse import unquote, urlsplit

from writing_agent.catalog import fingerprint


def extract_prose(result: dict, selectors: list[dict]) -> list[dict]:
    """Selectors name one final artifact, a turn, or a reviewed character span.

    Duplicate suppression requires an explicit shared delivery_group. Identical
    independent alternatives remain distinct samples for duplicate-rate metrics.
    """
    artifacts, seen = [], {}
    for index, selector in enumerate(selectors):
        item = {
            "id": selector.get("id", str(index)),
            "selector": selector,
            "extraction_version": 2,
            "status": "missing_prose",
            "text": None,
            "provenance": result.get("provenance", "unknown"),
            "session_identity": result.get("identity"),
        }
        try:
            if selector["kind"] == "file":
                snapshot = (
                    result["after"]
                    if "turn" not in selector
                    else result["turns"][selector["turn"]]["snapshot"]
                )
                raw = snapshot[selector["path"]]
            elif "turn" in selector:
                raw = result["turns"][selector["turn"]]["output"]
            else:
                raw = result["output"]
            item["source_hash"] = fingerprint(raw)
            selection = selector.get("selection", "whole")
            if selection == "whole":
                start, end = 0, len(raw)
            elif selection == "delimited":
                left, right = selector["start"], selector["end"]
                if not left or not right:
                    raise ValueError("Empty prose delimiter configuration")
                if left not in raw or right not in raw:
                    item["reason"] = (
                        "Required prose delimiters are missing from the designated artifact"
                    )
                    artifacts.append(item)
                    continue
                if raw.count(left) != 1 or raw.count(right) != 1:
                    raise ValueError("Prose delimiters are ambiguous")
                start, end = raw.index(left) + len(left), raw.index(right)
            elif selection == "span":
                if selector.get("source_hash") != fingerprint(raw):
                    raise ValueError("Reviewed span does not match source hash")
                start, end = selector["start"], selector["end"]
            else:
                raise ValueError("Unknown selection method")
            if not 0 <= start <= end <= len(raw):
                raise ValueError("Invalid prose bounds")
            prose = raw[start:end]
            item.update(
                spans=[[start, end]],
                text=prose,
                prose_hash=fingerprint(prose),
                status="ok" if prose.strip() else "missing_prose",
            )
            group = selector.get("delivery_group")
            key = (group, item["prose_hash"])
            if group and key in seen:
                item.update(status="duplicate_delivery", duplicate_of=seen[key])
            elif group:
                seen[key] = item["id"]
        except (KeyError, IndexError):
            item["reason"] = "Designated reply or file is missing"
        except (ValueError, TypeError) as exc:
            item.update(status="needs_review", reason=str(exc))
        artifacts.append(item)
    return artifacts


def markdown_graph(files: dict[str, str], entrypoints: list[str]) -> dict:
    """ATX headings and inline local links; reference-style links are unsupported.

    Ignore fenced code. Percent-decode links; recognize GitHub-like heading slugs
    and duplicate-heading suffixes. External URLs do not enter the denominator.
    """
    pages = {p: text for p, text in files.items() if p.endswith(".md")}
    anchors, links, unsupported = {}, [], []
    for path, text in pages.items():
        text = re.sub(r"(?ms)^\s*(```|~~~).*?^\s*\1[^\n]*$", "", text)
        counts, slugs = {}, set()
        for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
            slug = re.sub(r"[^\w\s-]", "", heading.casefold()).replace(" ", "-")
            count = counts.get(slug, 0)
            slugs.add(slug + (f"-{count}" if count else ""))
            counts[slug] = count + 1
        anchors[path] = slugs
        if re.search(r"^\s*\[[^\]]+\]:", text, re.M):
            unsupported.append(path)
        for href in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
            href = href.strip().strip("<>")
            parts = urlsplit(href)
            if parts.scheme or parts.netloc:
                continue
            target = (
                posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(parts.path)))
                if parts.path
                else path
            )
            links.append({"from": path, "to": target, "anchor": unquote(parts.fragment)})
    adjacency = {p: set() for p in pages}
    for link in links:
        link["valid"] = link["to"] in pages and (
            not link["anchor"] or link["anchor"] in anchors[link["to"]]
        )
        if link["valid"]:
            adjacency[link["from"]].add(link["to"])
    reached, pending = set(), list(entrypoints)
    while pending:
        current = pending.pop()
        if current in reached or current not in pages:
            continue
        reached.add(current)
        pending.extend(adjacency[current] - reached)
    incoming = set().union(*adjacency.values()) if adjacency else set()
    return {
        "links": links,
        "valid_fraction": sum(link["valid"] for link in links) / len(links) if links else None,
        "pages": len(pages),
        "reachable": sorted(reached),
        "reachability": len(reached) / len(pages) if pages else None,
        "orphans": sorted(set(pages) - incoming - set(entrypoints)),
        "missing_entrypoints": sorted(set(entrypoints) - set(pages)),
        "unsupported_reference_links": unsupported,
        "dialect": "inline-atx-v1",
    }
