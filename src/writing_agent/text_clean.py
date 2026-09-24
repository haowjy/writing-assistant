"""Deterministic cleanup. Raw downloads stay intact; these return derived text."""

import re

_GUTENBERG_START = re.compile(
    r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
    re.I | re.S,
)
_GUTENBERG_END = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_PARAGRAPH = re.compile(r"\S.*?(?=\n\s*\n|\Z)", re.S)
_PRODUCED = re.compile(r"^produced by\b", re.I)
_CHAPTER = re.compile(r"^(chapter|book|part|letter)\b", re.I)
_WEB_RESIDUE = re.compile(
    r"(copyright|all rights reserved|privacy policy|\bcookies?\b|subscribe|sign in|"
    r"log in|share this|follow us|contact us|terms of use|click here|read more|"
    r"advertisement|sponsored|powered by)",
    re.I,
)


def strip_gutenberg(text: str) -> str:
    """Drop the Project Gutenberg license wrapper. The raw file keeps it."""
    parts = _GUTENBERG_START.split(text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Gutenberg start marker absent")
    return _GUTENBERG_END.split(parts[1], maxsplit=1)[0].strip()


def opening_prose(text: str, *, min_chars: int = 800, max_chars: int = 2200) -> str:
    """First prose paragraphs, skipping a title page and production credit."""
    chunks = []
    for match in _PARAGRAPH.finditer(text):
        paragraph = match.group().strip()
        if not paragraph or _PRODUCED.match(paragraph):
            continue
        letters = [char for char in paragraph if char.isalpha()]
        mostly_title = letters and sum(char.isupper() for char in letters) / len(letters) > 0.8
        if mostly_title and len(paragraph) < 120:
            continue
        if _CHAPTER.match(paragraph) and len(paragraph) < 40:
            continue
        if len(paragraph) < 80 and paragraph.count(" ") < 8:
            continue
        chunks.append(paragraph)
        if sum(len(chunk) for chunk in chunks) >= min_chars:
            break
    if not chunks:
        raise ValueError("No prose opening found")
    body = "\n\n".join(chunks)
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0].strip()
    return body


def clean_web_text(text: str, *, min_words: int = 40) -> str | None:
    """Drop navigation, credits, and other non-writing lines.

    This is a deterministic filter, not the uncommitted model pass that cleaned the
    FineWeb pilot. A document that falls under ``min_words`` is rejected.
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.count("|") >= 2:
            continue
        if _WEB_RESIDUE.search(stripped) and len(stripped.split()) < 12:
            continue
        if re.fullmatch(r"[\W\d_]+", stripped):
            continue
        kept.append(stripped)
    if len(" ".join(kept).split()) < min_words:
        return None
    return "\n".join(kept)
