"""Situations end to end through `wa-bench situate` and `run --from`: a new task set up on
the files a closed episode ended with, stored once, played only as rollouts."""

import json

import pytest

from writing_assistant import cli
from writing_assistant.episode import Episode
from writing_assistant.episode.segments import Store, _unpack
from writing_assistant.suites import SUBJECT, SUITES, one_message

KB = "The harbour town of Sel. " * 400


class Notes:
    """A suite for these tests: one writer, one message, set up at a position. The payload
    may grant the writer restricted files (`grant`) and ask for a twin (`twin`)."""

    tools: dict = {}
    judged_by: dict = {}
    about: dict = {}
    metric_about: dict = {}
    text_fields: dict = {}

    def __init__(self) -> None:
        self.first_replies: dict[str, str] = {}  # ep_id -> the reply its first step was given

    def tasks(self, split, where=None):
        return []

    def setup(self, task, services, at=None):
        ep = at.begin(task.task_id, task.task_id, task.suite)
        ep.actors, ep.actor_tools = {"writer": SUBJECT}, {"writer": ["episode_read", "episode_write"]}
        ep.meta["temperature"] = 0.7
        for path in task.payload.get("grant", []):
            ep.state["file_acl"][path] = ["writer"]
        return [ep] + ([ep.fork("twin", baseline=True)] if task.payload.get("twin") else [])

    async def next_delivery(self, ep, last_reply):
        self.first_replies.setdefault(ep.ep_id, last_reply)
        return one_message(ep, "writer", lambda ep: "Add a line to draft.md.")

    async def score(self, ep, services, tally, twin=None):
        return {"draft_words": float(len(ep.files.get("draft.md", "").split()))}


@pytest.fixture
def notes(tmp_path, monkeypatch, fake_llm, capsys):
    """A run directory whose store holds a root (a big file, and one only `gm` may read),
    and `wa(*argv) -> stdout`, with a writer that writes draft.md and then replies."""
    suite = Notes()
    monkeypatch.setitem(SUITES, "notes", suite)

    def writer(messages, tools, endpoint):
        if messages[-1]["role"] == "user":
            rev = 1 if "draft.md" in json.dumps(messages) and '"rev": 1' in json.dumps(messages) else 0
            return {"tool_calls": [{"name": "episode_write", "arguments": {"path": "draft.md", "rev": rev, "text": "A line."}}]}
        return "Done."

    llm = fake_llm({"gemma-4-26b": [writer] * 100})
    monkeypatch.setattr(cli, "LLMClient", lambda models: llm)
    out = tmp_path / "run"
    out.mkdir()
    with Store(out / "store.db") as store:
        store.add_starts(Episode(ep_id="sel", task_id="sel", suite="notes", split="dev",
                                 files={"kb.md": KB, "secret.md": "Orin is the heir."}, file_revs={"kb.md": 1, "secret.md": 1},
                                 state={"file_acl": {"secret.md": ["gm"]}}))

    def wa(*argv) -> str:
        cli.main([*argv])
        return capsys.readouterr().out

    def situate(at: str, run=out, **row) -> list[str]:
        (tmp_path / "row.json").write_text(json.dumps({"suite": "notes", "kind": "note"} | row))
        return wa("situate", "--run", str(run), "--at", at, "--row", str(tmp_path / "row.json"), "--judges", "none").split()

    def play(source: str, run=out, *argv) -> str:
        return wa("run", "--suite", "notes", "--endpoint", "gemma-4-26b", "--judges", "none", "--out", str(run), "--from", source, *argv)

    wa.out, wa.suite, wa.situate, wa.play = out, suite, situate, play
    return wa


def rolled(run, source: str) -> list[str]:
    with Store(run / "store.db", readonly=True) as store:
        return store.rollouts(source)


def test_a_new_task_takes_a_closed_episodes_files_and_nothing_of_its_session(notes):
    """The position is a rollout forked mid-play, so its log holds a whole earlier session.
    The task set up there starts with none of it, the first step of its play is given no
    reply. A start stores what its parent's own play changed: the one file the first
    rollout wrote, and nothing for the fork that inherited it."""
    [first] = notes.situate("sel")
    notes.play(f"{first}@0")
    [played] = rolled(notes.out, first)
    notes.play(f"{played}@1")  # forked after its delivery returned: it ends where `played` did
    [position] = rolled(notes.out, played)
    [then], [early] = notes.situate(position, task_id="then"), notes.situate(played)
    with Store(notes.out / "store.db", readonly=True) as store:
        at, new = store.load(position), store.load(then)
        stored = [list(_unpack(store.db.execute("SELECT body FROM checkpoints WHERE seg = ? AND n = 0",
                                                (store.load(name).seg,)).fetchone()[0])["files"]) for name in (early, then)]
        assert store.chain(then)[0] == store.chain("sel")[0] and early.startswith("sel/")
    assert at.events and at.calls and at.files["draft.md"] == "A line."
    assert (new.events, new.calls, new.last_reply(), new.actors) == ([], [], "", {"writer": SUBJECT})
    assert (new.files, new.file_revs) == (at.files, at.file_revs)
    assert stored == [["draft.md"], []]
    notes.play("then@0")
    [again] = rolled(notes.out, "then")
    assert notes.suite.first_replies[again] == ""


def test_a_file_restricted_at_the_position_is_closed_to_every_new_seat_until_setup_grants_it(notes):
    [closed] = notes.situate("sel")
    [granted] = notes.situate("sel", payload={"grant": ["secret.md"]})
    with Store(notes.out / "store.db", readonly=True) as store:
        closed, granted = store.load(closed), store.load(granted)
    assert not any(closed.readable(seat, "secret.md") for seat in ("writer", "gm", "anyone"))
    assert "secret.md" not in closed.view("writer").files and closed.readable("writer", "kb.md")
    assert granted.readable("writer", "secret.md") and not granted.readable("gm", "secret.md")


def test_a_start_is_played_only_as_a_rollout_and_no_reader_of_played_episodes_meets_it(notes):
    """Readers (`run`'s scoring of what has no row, its resume, `score`, `report`, the page
    and calibration sampling) list episodes through `Store.names`; a start is closed and
    nobody played it, so it must not be there to be scored or continued."""
    [base, twin] = notes.situate("sel", payload={"twin": True})
    with Store(notes.out / "store.db") as store:
        assert store.load(twin).baseline_of == base
        assert (store.starts(), store.situations(), store.names(), store.names(closed=False)) == (["sel", base, twin], [base, twin], [], [])
        with pytest.raises(ValueError, match="is closed; nothing is appended"):
            store.append(store.load(base))
    out = notes.play(f"{twin}@0")
    assert "open episodes to continue" not in out and "to score first" not in out
    [member] = rolled(notes.out, twin)
    assert "1 episodes, 1 scored" in notes("score", "--run", str(notes.out), "--judges", "none")
    with Store(notes.out / "store.db", readonly=True) as store:
        assert store.names() == list(store.scores("none")) == [member]
        assert store.load(member).status == "completed"


def test_an_open_episode_is_no_position(notes):
    with Store(notes.out / "store.db") as store:
        store.append(Episode(ep_id="open", task_id="open", suite="notes", split="dev"))
    with pytest.raises(SystemExit, match="no closed episode named 'open'"):
        notes.situate("open")


def test_a_situation_copied_into_another_store_is_one_there_and_plays_by_name(notes, tmp_path):
    [name] = notes.situate("sel")
    other = tmp_path / "other"
    other.mkdir()
    with Store(notes.out / "store.db", readonly=True) as store, Store(other / "store.db") as copy:
        store.copy([name], copy)
        assert (copy.situations(), copy.names()) == ([name], [])
    notes.play(f"{name}@0", other)
    with Store(other / "store.db", readonly=True) as copy:
        [member] = copy.rollouts(name)
        assert copy.load(member).files["draft.md"] == "A line."
