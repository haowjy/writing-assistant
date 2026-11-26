"""Gate tests for corpus/ (plan/benchmark-code.md §4, Slice 1). No network, no GPU."""

import json
import zlib
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from writing_assistant.corpus import Corpus, Work
from writing_assistant.corpus.authors import K_VALUES, partition_author_corpus
from writing_assistant.corpus.collect import collect
from writing_assistant.corpus.release import export_public, stage_sync
from writing_assistant.episode import Episode
from writing_assistant.episode.segments import Store
from writing_assistant.episode.store import local_only, split_of
from writing_assistant.episode.tools import dispatch_tool

VOCAB = [f"w{i}" for i in range(2000)]


def prose(seed: int, words: int) -> str:
    """Seeded word salad: its 8-grams are unique to the seed."""
    rng = np.random.default_rng(seed)
    toks = rng.choice(VOCAB, size=words)
    return "\n\n".join(" ".join(toks[i : i + 60]) + "." for i in range(0, words, 60))


def book(seed: int, n_chapters: int, words: int = 700) -> str:
    chapters = "\n\n".join(f"CHAPTER {i + 1}.\n\n{prose(seed * 100 + i, words)}" for i in range(n_chapters))
    return (
        f"*** START OF THE PROJECT GUTENBERG EBOOK {seed} ***\n\nA BOOK\n\nCONTENTS\n\n"
        + "\n".join(f"CHAPTER {i + 1}." for i in range(n_chapters))
        + f"\n\n{chapters}\n\n*** END OF THE PROJECT GUTENBERG EBOOK {seed} ***\n\nSection 1. General Terms of Use\n"
    )


def add(corpus: Corpus, n: int, author: str, text: str, **kw) -> Work:
    work = Work(
        work_id=f"gutenberg:{n}", source="gutenberg", source_group=author, title=f"Book {n}",
        licence=kw.pop("licence", "public-domain"), url=f"http://x/{n}", content_hash="",
        split=kw.pop("split", split_of(author)), **kw,
    )
    return corpus.add(replace(work, content_hash=corpus.store_text(text.encode())))


async def outline(text: str) -> list[dict]:
    return [{"role": "beat", "paraphrase": text[:20]}]


async def test_author_partition_is_disjoint_and_draws_nest(tmp_path):
    corpus = Corpus(tmp_path, SimpleNamespace(outline=outline))
    many = [add(corpus, n, "Austen, Jane", book(n, 12)) for n in (1, 2, 3)]
    one = [add(corpus, 4, "Shelley, Mary", book(4, 25))]
    assert [len(corpus.chapters(w)) for w in many + one] == [12, 12, 12, 25]  # contents dropped
    for works in (many, one):
        for seed in range(5):
            split = await partition_author_corpus(corpus, works, seed=seed)
            roles = [
                {(p.work_id, p.chapter) for p in split.exemplar_pool},
                {(t.y_star.work_id, t.y_star.chapter) for t in split.targets},
                {(p.work_id, p.chapter) for p in split.verifier_refs},
            ]
            assert all(roles), roles
            assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
            if len(works) >= 3:  # by book: no work serves two roles
                books = [{w for w, _ in r} for r in roles]
                assert not (books[0] & books[1] or books[0] & books[2] or books[1] & books[2])
            assert all(t.beat_sheet for t in split.targets)
            for draw_seed in range(5):
                draws = [split.draw(k, draw_seed) for k in K_VALUES]
                assert [len(d) for d in draws] == list(K_VALUES)
                assert all(set(a) <= set(b) for a, b in zip(draws, draws[1:]))


GUTENDEX_PAGES = {
    "1": {
        "count": 4,
        "next": "https://gutendex.com/books/?ids=11%2C12%2C13%2C14&page=2",
        "results": [
            {"id": 11, "title": "Alice", "authors": [{"name": "Carroll, Lewis"}], "copyright": False,
             "media_type": "Text", "download_count": 50000, "summaries": ["A girl falls."],
             "subjects": ["Fantasy"], "bookshelves": [], "languages": ["en"]},
            {"id": 12, "title": "Old Book", "authors": [{"name": "Carroll, Lewis"}, {"name": "Tenniel, John"}],
             "copyright": None, "media_type": "Text", "download_count": 30, "summaries": [],
             "subjects": [], "bookshelves": [], "languages": ["en"]},
        ],
    },
    "2": {
        "count": 4,
        "next": None,
        "results": [
            {"id": 13, "title": "Missing", "authors": [{"name": "Nobody, A"}], "copyright": False,
             "media_type": "Text", "download_count": 5, "languages": ["en"]},
            {"id": 14, "title": "Folk Tales", "authors": [], "copyright": False,
             "media_type": "Text", "download_count": 900, "languages": ["en"]},
            {"id": 15, "title": "Audio", "authors": [], "copyright": False, "media_type": "Sound"},
        ],
    },
}
LATIN1_BOOK = (
    "The Project Gutenberg EBook of Old Book\r\n\r\nThis eBook is for the use of anyone anywhere\r\n\r\n"
    "*** START OF THIS PROJECT GUTENBERG EBOOK OLD BOOK ***\r\n\r\nProduced by A. Volunteer\r\n\r\n\r\n\r\n"
    "CHAPTER I.\r\n\r\nThe caf\xe9 was na\xefve.\r\n\r\n*** END OF THIS PROJECT GUTENBERG EBOOK OLD BOOK ***\r\n"
)
MIRROR = {
    "/1/11/11-0.txt": book(11, 3).encode(),
    "/1/12/12-8.txt": LATIN1_BOOK.encode("latin-1"),
    "/1/14/14.txt": book(14, 2).encode(),
}


def gutenberg_transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "gutendex.com":
            assert request.url.path == "/books/"
            return httpx.Response(200, json=GUTENDEX_PAGES[request.url.params.get("page", "1")])
        assert request.url.host == "aleph.gutenberg.org", request.url
        body = MIRROR.get(request.url.path)
        return httpx.Response(200, content=body) if body is not None else httpx.Response(404)

    return httpx.MockTransport(handle)


async def test_gutenberg_importer_against_fixture(tmp_path):
    requests: list[httpx.Request] = []
    corpus = Corpus(tmp_path)
    with httpx.Client(transport=gutenberg_transport(requests)) as http:
        works = collect(corpus, {"gutenberg": {"ids": [11, 12, 13, 14]}}, http=http, delay=0)
    by_id = {w.work_id: w for w in Corpus(tmp_path).select()}  # what a fresh process reads back
    assert [w.work_id for w in works] == sorted(by_id) == ["gutenberg:11", "gutenberg:12", "gutenberg:14"]
    alice, old, folk = by_id.values()
    assert (alice.source_group, alice.split, alice.licence) == ("Carroll, Lewis", split_of("Carroll, Lewis"), "public-domain")
    assert (old.source_group, old.split, old.licence) == ("Carroll, Lewis", alice.split, None)
    assert old.authors == ["Carroll, Lewis", "Tenniel, John"]
    assert folk.source_group == "gutenberg:14" and folk.split == split_of("gutenberg:14")
    assert alice.synopsis_source == "catalogue" and old.synopsis is None
    assert alice.familiarity > folk.familiarity > old.familiarity
    assert alice.url == "http://aleph.gutenberg.org/1/11/11-0.txt"
    assert corpus.text(old) == "CHAPTER I.\n\nThe café was naïve.\n"  # cp1252 decoded, header, credits, footer gone
    assert "General Terms" not in corpus.text(alice)
    assert len(corpus.chapters(alice)) == 3

    before = (tmp_path / "works.jsonl").read_text()
    requests.clear()
    with httpx.Client(transport=gutenberg_transport(requests)) as http:
        again = collect(corpus, {"gutenberg": {"ids": [11, 12, 13, 14]}}, http=http, delay=0)
    assert [w.work_id for w in again] == [w.work_id for w in works]
    assert (tmp_path / "works.jsonl").read_text() == before  # no duplicate rows
    assert {r.url.host for r in requests} == {"gutendex.com", "aleph.gutenberg.org"}
    assert {r.url.path for r in requests if r.url.host == "aleph.gutenberg.org"} == {
        "/1/13/13-0.txt", "/1/13/13-8.txt", "/1/13/13.txt"
    }  # cached books are not refetched; only the missing one is retried


async def test_overlap_separates_context_from_memorization(tmp_path):
    corpus = Corpus(tmp_path)
    work = add(corpus, 7, "Austen, Jane", book(7, 4))
    exemplar, unseen = corpus.chapters(work)[0], corpus.chapters(work)[2]
    ep = Episode(
        ep_id="e", task_id="t", suite="3", split=work.split, actors={"writer": "qwen"},
        actor_tools={"writer": ["episode_read", "episode_write"]},
        files={"exemplar.md": exemplar}, file_revs={"exemplar.md": 1}, state={"work_ids": [work.work_id]},
    )
    ep.deliver("writer", "Write the next scene in this author's voice. Read exemplar.md first.")

    copied = corpus.overlap(exemplar, ep)
    assert copied["context_copy"][1] == 1.0 and copied["context_copy"][0] > 300
    assert copied["memorization"] == (0, 0.0) and not corpus.flag(copied)

    # The subject recites a chapter it was never shown, saves it and reads it back: its own
    # file is not context, so the recitation still counts as memorization.
    await dispatch_tool(ep, "writer", "episode_write", {"path": "draft.md", "text": unseen, "rev": 0}, {})
    await dispatch_tool(ep, "writer", "episode_read", {"path": "draft.md"}, {})
    recited = corpus.overlap(unseen, ep)
    assert recited["context_copy"] == (0, 0.0)
    assert recited["memorization"][1] == 1.0 and recited["memorization"][0] > 300 and corpus.flag(recited)

    no_work = Episode(ep_id="f", task_id="t", suite="1a", split="dev", actors={"writer": "qwen"})
    assert "memorization" not in corpus.overlap(unseen, no_work)


async def test_export_never_writes_restricted_text(tmp_path):
    corpus = Corpus(tmp_path / "cache", local_dev_dir=tmp_path / "local")
    texts = {n: book(n, 2) for n in (1, 2, 3, 4)}
    add(corpus, 1, "A", texts[1])
    add(corpus, 2, "B", texts[2], licence="copyrighted")
    add(corpus, 3, "C", texts[3], licence=None)
    add(Corpus(tmp_path / "local"), 4, "D", texts[4])  # under local_dev/: stamped local_dev on load
    corpus = Corpus(tmp_path / "cache", local_dev_dir=tmp_path / "local")
    assert corpus.get("gutenberg:4").split == "local_dev" and corpus.select(include_local_dev=True)[-1].work_id == "gutenberg:4"
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "1a.jsonl").write_text(json.dumps({"task_id": "t", "suite": "1a", "kind": "dm_play", "split": "dev",
                                                "work_ids": [], "payload": {}}) + "\n")
    (tasks / "test-split-uses.jsonl").write_text(json.dumps({"run": "/home/someone/runs/x"}) + "\n")  # not a task file
    export_public(corpus, tmp_path / "out", {"1a": {"dm_play": []}}, tasks_dir=tasks)
    assert sorted(p.name for p in (tmp_path / "out" / "tasks").iterdir()) == ["1a.jsonl"]
    out = (tmp_path / "out" / "works.jsonl").read_text()
    rows = {r["work_id"]: r for r in map(json.loads, out.splitlines())}
    assert sorted(rows) == ["gutenberg:1", "gutenberg:2", "gutenberg:3"]
    assert rows["gutenberg:1"]["text"] == corpus.text(corpus.get("gutenberg:1"))
    assert all("text" not in rows[w] and rows[w]["content_hash"] for w in ("gutenberg:2", "gutenberg:3"))
    with local_only():  # work 4 is local_dev text, readable only under the guard
        for n in (2, 3, 4):
            assert corpus.text(corpus.get(f"gutenberg:{n}"))[:200] not in out

    # A mistyped mark hashes nothing, so the copyrighted passage would leave: export refuses.
    passage = {"work_id": "gutenberg:2", "text": texts[2].split("\n\n")[5]}
    (tasks / "3.jsonl").write_text(json.dumps({"task_id": "e", "suite": "3", "kind": "critique_edit", "split": "dev",
                                               "work_ids": ["gutenberg:2"], "payload": {"passage": passage}}) + "\n")
    with pytest.raises(ValueError, match="still holds text of gutenberg:2"):
        export_public(corpus, tmp_path / "out", {"3": {"critique_edit": ["passage.txt"]}}, tasks_dir=tasks)


def test_sync_holds_back_an_episode_when_any_segment_of_its_chain_may_not_leave(tmp_path):
    """`sync` writes a new store of the episodes that may leave, and each is copied with the
    whole chain it is stored after. So the licence test is of every segment in the chain at
    every checkpoint: an unlisted ancestor's text, or a work the state named for a while,
    would otherwise leave inside an episode whose final state is clean."""
    corpus = Corpus(tmp_path / "cache")
    add(corpus, 1, "A", book(1, 2))
    add(corpus, 2, "B", book(2, 2), licence="copyrighted")
    free, held = "gutenberg:1", "gutenberg:2"

    def ep(ep_id, split="dev", works=(), note=""):
        return Episode(ep_id=ep_id, task_id=ep_id, suite="1a", split=split, state={"work_ids": list(works), "note": note})

    runs = tmp_path / "runs"
    for run in ("r", "mine"):
        (runs / run).mkdir(parents=True)
        (runs / run / "run.json").write_text("{}")
    with Store(runs / "r" / "store.db") as store:
        base = ep("free", works=[free], note="FREE TEXT")
        fork = base.fork("fork")
        store.close(base, fork)
        clean = ep("ancestor", works=[held], note="HELD ANCESTOR").fork("clean")  # the ancestor is saved unlisted
        clean.state = {"work_ids": [free], "note": ""}
        store.close(clean)
        middle = ep("middle", works=[free])
        store.append(middle)
        middle.state = {"work_ids": [free, held], "note": "HELD MIDDLE"}
        middle.log("dm", "say", {"text": "a turn"}, {"*": "full"})
        store.append(middle)
        middle.state = {"work_ids": [free], "note": ""}
        store.close(middle)
    with Store(runs / "mine" / "store.db") as store:
        store.close(ep("personal", split="local_dev", note="HELD PERSONAL"))  # names no work at all
    for run, ids in (("r", ["free", "free~fork", "ancestor~clean", "middle"]), ("mine", ["personal"])):
        (runs / run / "scores-eval.jsonl").write_text("".join(json.dumps({"ep_id": i}) + "\n" for i in ids))
        (runs / run / "judgments-eval.jsonl").write_text("".join(json.dumps({"ep_id": i, "scored": i}) + "\n" for i in ids))

    counts = stage_sync(tmp_path / "stage", runs, corpus, {})

    assert dict(counts) == {"episodes": 2, "episodes_left_out": 3}
    staged = tmp_path / "stage" / "runs"
    text = b""
    for run, names in (("r", ["free", "free~fork"]), ("mine", [])):
        with Store(staged / run / "store.db", readonly=True) as store:
            assert [e.ep_id for e in store.episodes()] == names
            text += b"".join(zlib.decompress(body) for (body,) in store.db.execute("SELECT body FROM checkpoints"))
        for file in ("scores-eval.jsonl", "judgments-eval.jsonl"):
            assert [json.loads(line)["ep_id"] for line in (staged / run / file).read_text().splitlines()] == names
    text += b"".join(p.read_bytes() for p in (tmp_path / "stage").rglob("*") if p.is_file())
    assert b"FREE TEXT" in text and b"HELD" not in text
