"""Psycholinguistic processing-difficulty profile (after Clark et al. 2026, arXiv:2610.29162):
spaCy `en_core_web_sm` parses plus `wordfreq` frequencies, no model call.

Rates are per 1,000 tokens (punctuation and spaces excluded) so texts of different length
compare. A feature with nothing to average over (empty text, no multi-word sentence, no
content word) is NaN, not 0: a 0 would read as "easy prose" and drag a run's mean; reports
average with NaN skipped.

Not covered from the paper: unattested bigrams and n-gram perplexity (they need an n-gram
table; `wordfreq` is unigram only) and denominal verbs. Animacy is a lexicon rule, where the
paper used a small classifier.
"""

from __future__ import annotations

import math
from functools import cache

import spacy
from wordfreq import zipf_frequency

VERSION = "readability-v1"
SPACY_MODEL = "en_core_web_sm"
RARE_ZIPF = 3.0  # below ~1 occurrence per million words

# Verbs whose subject is normally a person: an inanimate subject with one ("the report
# argues", "silence demanded") is the construction the paper found models over-use.
ANIMATE_VERBS = frozenset(
    "admit agree argue ask beg believe claim complain confess decide demand deny desire doubt "
    "dream expect fear feel forgive guess hate hope imagine insist intend know laugh lie like "
    "love mean mourn notice pray pretend promise realize realise recall recognize refuse regret "
    "remember say scream shout sigh smile sob suggest suppose suspect swear think tell "
    "understand want whisper wish wonder worry".split()
)
ANIMATE_NOUNS = frozenset(
    "man woman men women boy girl child children kid baby person people mother father mom dad "
    "parent sister brother son daughter wife husband friend stranger king queen prince princess "
    "lord lady soldier guard captain doctor teacher student priest wizard witch knight servant "
    "master mistress officer detective crowd family everyone someone somebody anyone nobody "
    "everybody anybody one god".split()
)
PERSONAL_PRONOUNS = frozenset("i you he she we they me him her us them who whoever myself yourself himself herself".split())
RELATIVISERS = frozenset({"WDT", "WP", "WP$"})


@cache
def _nlp():
    return spacy.load(SPACY_MODEL)


def profile(text: str) -> dict[str, float]:
    doc = _nlp()(text)
    words = [t for t in doc if not (t.is_punct or t.is_space)]
    n = len(words)

    def per_1k(count: int) -> float:
        return 1000 * count / n if n else math.nan

    return {
        "n_tokens": float(n),
        "n_sentences": float(sum(1 for s in doc.sents if any(not (t.is_punct or t.is_space) for t in s))),
        "mean_dep_length": _mean([abs(t.i - t.head.i) for t in words if t.head.i != t.i and not t.head.is_punct]),
        "norm_dep_length": _mean([d for s in doc.sents if (d := _ndd(s)) is not None]),
        "stacked_nouns_per_1k": per_1k(sum(1 for t in words if t.pos_ == "NOUN" and _noun_compound(t))),
        "inanimate_animate_per_1k": per_1k(sum(1 for t in words if _inanimate_subject_animate_verb(t))),
        "object_relatives_per_1k": per_1k(sum(1 for t in words if _object_relative(t) is not None)),
        "reduced_object_relatives_per_1k": per_1k(sum(1 for t in words if _object_relative(t) is False)),
        "rare_word_rate": _mean([float(zipf_frequency(t.lower_, "en") < RARE_ZIPF) for t in _content(words)]),
        "mean_zipf": _mean([zipf_frequency(t.lower_, "en") for t in _content(words)]),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def _content(words):
    # Names are rare by nature in fiction; counting them would score a cast list as hard prose.
    return [t for t in words if t.is_alpha and t.pos_ not in ("PROPN", "X")]


def _ndd(sent) -> float | None:
    """Normalised dependency distance (Lei & Jockers 2020): |ln(MDD / sqrt(root * len))|,
    which takes sentence length out of mean dependency distance. None for one-word sentences."""
    words = [t for t in sent if not (t.is_punct or t.is_space)]
    dists = [abs(t.i - t.head.i) for t in words if t.head.i != t.i and not t.head.is_punct]
    if not dists:
        return None
    root = next((k for k, t in enumerate(words, 1) if t.head.i == t.i), 1)
    return abs(math.log((sum(dists) / len(dists)) / math.sqrt(root * len(words))))


def _noun_compound(t) -> bool:
    """The head of an N N (N ...) compound; a chain counts once, at its head."""
    return t.dep_ != "compound" and any(c.dep_ == "compound" and c.pos_ == "NOUN" for c in t.children)


def _inanimate_subject_animate_verb(t) -> bool:
    if t.dep_ != "nsubj" or t.head.pos_ != "VERB" or t.head.lemma_.lower() not in ANIMATE_VERBS:
        return False
    if t.pos_ == "PROPN" or t.ent_type_ in ("PERSON", "NORP"):
        return False
    if t.pos_ == "PRON":
        return t.lower_ not in PERSONAL_PRONOUNS
    return t.pos_ == "NOUN" and t.lemma_.lower() not in ANIMATE_NOUNS and t.lower_ not in ANIMATE_NOUNS


def _object_relative(t) -> bool | None:
    """For a relative-clause verb whose gap is not its subject ("the book (that) I read"):
    True with a relativiser, False when reduced. None for anything else."""
    if t.dep_ != "relcl":
        return None
    rel = [c for c in t.children if c.tag_ in RELATIVISERS or (c.lower_ == "that" and c.dep_ in ("dobj", "mark", "nsubj"))]
    has_subject = any(c.dep_ in ("nsubj", "nsubjpass") and c not in rel for c in t.children)
    has_object = any(c.dep_ == "dobj" and c not in rel for c in t.children)
    if not has_subject or has_object:
        return None
    return bool(rel)
