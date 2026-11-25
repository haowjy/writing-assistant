"""Turning cached bytes into a work's text, and the one chapter splitter.

The cache keeps bytes exactly as fetched (so `content_hash` can be checked against the
source URL); everything here runs at read time and is pure.
"""

from __future__ import annotations

import re
import unicodedata

# Gutenberg plain text: the body sits between these markers. Mirror files published since
# about 2021 start at the START line; older ones carry the full licence header, and the
# oldest end their header with the "SMALL PRINT" line instead.
_PG_START = re.compile(
    r"^\s*(\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG E-?BOOK.*"
    r"|\*END\*THE SMALL PRINT!.*)$",
    re.I | re.M,
)
_PG_END = re.compile(
    r"^\s*(\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG E-?BOOK"
    r"|End of (the )?Project Gutenberg'?s? (E-?Book|E-?text|Etext)"
    r"|End of this Project Gutenberg)",
    re.I | re.M,
)
# A credits paragraph right after the START marker ("Produced by ...") is transcriber text.
_PG_CREDITS = re.compile(r"\A\s*(Produced by|E-?text prepared by|Transcribed (from|by))[^\n]*(\n[^\n]+)*\n", re.I)


def decode(raw: bytes) -> str:
    """UTF-8 (with or without BOM), else Windows-1252, else Latin-1 (never fails). Gutenberg's
    `-8.txt` files are ISO-8859-1 or, in practice, often cp1252; plain `.txt` is usually
    ASCII. Line endings become `\\n`."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def gutenberg_body(text: str) -> str:
    """The book between Project Gutenberg's START and END markers, without the transcriber
    credits. ValueError when the START marker is missing: guessing would leave licence text
    in the body, where it would count as memorization of every book."""
    start = _PG_START.search(text)
    if start is None:
        raise ValueError("no Project Gutenberg START marker")
    body = text[start.end():]
    end = _PG_END.search(body)
    if end is not None:
        body = body[: end.start()]
    body = _PG_CREDITS.sub("", body.lstrip("\n"), count=1)
    return body.strip() + "\n"


# -- chapters ---------------------------------------------------------------------------------

_NUMBER_WORDS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last"
)
_NUMERAL = rf"(\d+|[IVXLCDM]+|(?i:{_NUMBER_WORDS})([- ](?i:{_NUMBER_WORDS}))?)"
# A keyword heading: "CHAPTER I.", "Chapter 1", "Chapter the First", "BOOK II", "ACT III",
# "SCENE 2.", "Letter 4", "STAVE ONE". The keyword is capitalised and the numeral must follow,
# so a hard-wrapped prose line ("part of the house") never matches.
_KEYWORD = re.compile(
    rf"^(?P<kw>CHAPTER|Chapter|CHAP\.|Chap\.|BOOK|Book|PART|Part|VOLUME|Volume|ACT|Act|SCENE|Scene|"
    rf"LETTER|Letter|STAVE|Stave|CANTO|Canto|ADVENTURE|Adventure)\s+(the\s+)?(?P<num>{_NUMERAL})\b[.:]?(?P<rest>.*)$"
)
# A bare numeral heading: "IV.", "IV. THE POOL OF TEARS", "12", "XII". Used only when a book
# has no keyword headings, because numbered lists and dates also start lines this way.
_BARE = re.compile(r"^(?P<num>[IVXLCDM]+|\d{1,3})(\.|\.\s+(?P<rest>\S.*)|)$")
_MAX_HEADING = 80  # characters; a longer line is prose
_MIN_BODY = 50  # words; a shorter segment is a contents line or a part title page
_TOC_BODY = 300  # words; a segment this short whose label recurs later is a contents entry
FALLBACK_WORDS = 5000  # chunk size for a long text with no headings


def _headings(lines: list[str], pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    """(line index, label) for each heading line. A heading starts a paragraph (blank line
    before), or follows another heading as in a contents list."""
    out: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) > _MAX_HEADING:
            continue
        m = pattern.match(s)
        if m is None:
            continue
        prev = lines[i - 1].strip() if i else ""
        if prev and not (out and out[-1][0] == i - 1):
            continue
        kw = m.groupdict().get("kw") or ""
        out.append((i, f"{kw.casefold()} {m.group('num').casefold()}".strip()))
    return out


def _words(text: str) -> int:
    return len(text.split())


def split_chapters(body: str) -> list[str]:
    """The work's chapters, each starting with its heading line.

    Headings are keyword lines (Chapter, Book, Part, Act, Scene, Letter, Stave, ...) or, in a
    book with none, bare numerals. Every heading level splits (a play yields scenes, a book in
    parts yields its chapters), then segments with under 50 words of body are dropped (contents
    lines, part title pages, an ACT line directly followed by SCENE 1), as are segments under
    300 words whose label recurs later (a contents list with a synopsis per entry). Text before
    the first heading (title page, preface, contents) is dropped. A text with fewer than two
    chapters is cut at paragraph breaks into ~5,000-word pieces, so no caller gets one giant
    chapter from a long book."""
    lines = body.split("\n")
    heads = _headings(lines, _KEYWORD)
    if len(heads) < 2:
        heads = _headings(lines, _BARE)
    segments = []
    for n, (i, label) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        text = "\n".join(lines[i:end]).strip()
        segments.append((label, text, _words(text) - _words(lines[i])))
    later_labels = [{label for label, _, _ in segments[n + 1:]} for n in range(len(segments))]
    chapters = [
        text
        for n, (label, text, body_words) in enumerate(segments)
        if body_words >= _MIN_BODY and not (body_words < _TOC_BODY and label in later_labels[n])
    ]
    if len(chapters) >= 2 or _words(body) <= 2 * FALLBACK_WORDS:
        return chapters or [body.strip()]
    return _chunks(body, FALLBACK_WORDS)


def _chunks(body: str, size: int) -> list[str]:
    chunks, current, count = [], [], 0
    for para in re.split(r"\n\s*\n", body.strip()):
        current.append(para)
        count += _words(para)
        if count >= size:
            chunks.append("\n\n".join(current))
            current, count = [], 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# -- tokens -----------------------------------------------------------------------------------

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Words and punctuation marks: within ~20% of a BPE count on English prose, with no
    tokenizer dependency. `passage_tokens` budgets only need to be the same for every author."""
    return len(_TOKEN.findall(text))


def truncate_tokens(text: str, budget: int) -> str:
    """The longest prefix of `text` with at most `budget` tokens, cut back to the last sentence
    end when one falls in the final half of the budget."""
    matches = list(_TOKEN.finditer(text))
    if len(matches) <= budget:
        return text.strip()
    cut = text[: matches[budget - 1].end()]
    sentence_ends = [m.end() for m in re.finditer(r"[.!?][\"'”’)]?(?=\s)", cut)]
    if sentence_ends and count_tokens(cut[: sentence_ends[-1]]) >= budget // 2:
        cut = cut[: sentence_ends[-1]]
    return cut.strip()


# -- normalisation for overlap ----------------------------------------------------------------

_WORD = re.compile(r"[^\W_]+")  # letters and digits; "_" is Gutenberg's italics marker
_APOSTROPHES = str.maketrans("", "", "'’ʼ")


def norm_words(text: str) -> list[str]:
    """Case, punctuation and whitespace normalised words: NFKC, casefolded, apostrophes
    dropped inside words (don't -> dont), every other non-letter a separator."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold().translate(_APOSTROPHES))
