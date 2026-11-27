"""Situations and the cast end to end through `wa-bench situate` and `run --from`: a new task
set up on the files a closed episode ended with, stored once, played only as rollouts, by
whichever models the run's cast names."""

import json

import pytest

from writing_assistant import cli
from writing_assistant.run import directory
from writing_assistant.episode import Episode
from writing_assistant.episode.segments import Store, _unpack
from writing_assistant.suites import SUBJECT, SUITES, one_message

KB = "The harbour town of Sel. " * 400


class Notes:
    """A suite for these tests: a writer (the subject) is sent one message, set up at a
    position. The payload may grant the writer restricted files (`grant`) and seat a reviewer
    played by the cast's `counterpart`, with limits of its own, who is sent one message
    after the writer has answered (`review`)."""

    tools: dict = {}
    judged_by: dict = {}
    about: dict = {}
    metric_about: dict = {}
    text_fields: dict = {}
    rubrics = {"words": ("note",)}

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
        if task.payload.get("review"):
            ep.actors["reviewer"] = "counterpart"
            ep.state["sampling"] = {"reviewer": {"max_tokens": 300, "temperature": 0.2}}
        return [ep]

    async def next_delivery(self, ep, last_reply):
        self.first_replies.setdefault(ep.ep_id, last_reply)
        sent = [e.actor for e in ep.events if e.kind == "deliver"]
        if sent == ["writer"] and "reviewer" in ep.actors:
            return "reviewer", "Say what you think of draft.md."
        return one_message(ep, "writer", lambda ep: "Add a line to draft.md.")

    async def score(self, ep, services, tallies, twin=None):
        return {"words": {"draft_words": float(len(ep.files.get("draft.md", "").split()))}}


@pytest.fixture
def notes(tmp_path, monkeypatch, fake_llm, capsys):
    """A run directory whose store holds a root (a big file, and one only `gm` may read),
    and `wa(*argv) -> stdout`, with a writer that writes draft.md and then replies."""
    suite = Notes()
    monkeypatch.setitem(SUITES, "notes", suite)

    def writer(messages, tools, endpoint):
        if messages[-1]["role"] == "user" and tools:
            return {"tool_calls": [{"name": "episode_write", "arguments": {"path": "draft.md", "rev": 0, "text": "A line."}}]}
        return "Done."

    llm = fake_llm({"gemma-4-26b": [writer] * 100, "qwen3.5-9b": [writer] * 100, "qwen3-32b-mini": ["It reads well."] * 100})
    monkeypatch.setattr(directory, "LLMClient", lambda models: llm)
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

    def play(source: str, run=out, *argv, **kw) -> str:
        return wa(*wa.argv(source, run, *argv, **kw))

    wa.argv = lambda source, run=out, *argv, endpoint="gemma-4-26b", home=out: [
        "run", "--suite", "notes", "--endpoint", endpoint, "--judges", "none", "--out", str(run), "--from", source,
        *(["--from-run", str(home)] if run != home else []), *argv]
    wa.out, wa.suite, wa.situate, wa.play, wa.llm = out, suite, situate, play, llm
    return wa


def rolled(run, source: str) -> list[str]:
    with Store(run / "store.db", readonly=True) as store:
        return store.rollouts(source)


def members(run, source: str) -> list[Episode]:
    with Store(run / "store.db", readonly=True) as store:
        return [store.load(name) for name in store.rollouts(source)]


def played_by(ep: Episode) -> dict[str, set[str]]:
    """seat -> the endpoints its calls went to."""
    return {seat: {c.endpoint for c in ep.calls if c.actor == seat} for seat in ep.actors}


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
    [name] = notes.situate("sel")
    with Store(notes.out / "store.db") as store:
        assert (store.starts(), store.situations(), store.names(), store.names(closed=False)) == (["sel", name], [name], [], [])
        with pytest.raises(ValueError, match="is closed; nothing is appended"):
            store.append(store.load(name))
    out = notes.play(f"{name}@0")
    assert "open episodes to continue" not in out and "to score first" not in out
    [member] = rolled(notes.out, name)
    with Store(notes.out / "store.db", readonly=True) as store:
        assert store.load(member).status == "completed" and store.judgings() == []  # `run` plays a situation and scores nothing
    assert "1 episodes, 1 scored" in notes("score", "--run", str(notes.out), "--judges", "none")
    with Store(notes.out / "store.db", readonly=True) as store:
        assert store.names() == list(store.scores("none/words")) == [member] and store.judgings() == ["none/words"]
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
    notes.play(f"{name}@0", other, home=other)
    with Store(other / "store.db", readonly=True) as copy:
        [member] = copy.rollouts(name)
        assert copy.load(member).files["draft.md"] == "A line."


MODELS = {"gemma-4-26b": "gemma-4-26B_q4_0-it.gguf", "qwen3.5-9b": "Qwen3.5-9B-Q8_0.gguf", "qwen3-32b-mini": "qwen3-32b"}


def test_one_situation_is_played_by_whichever_cast_the_run_names_and_says_who_played_each_seat(notes, tmp_path, capsys, monkeypatch):
    """Three runs of one stored situation: two subjects against the benchmark cast, and one
    subject with itself in the other seat. A seat keeps its cast name in every one, each
    episode is stamped with the models that played (and whether an endpoint's recorded
    thinking is a trace or a summary, when models.toml says), the reviewer keeps the situation's
    limits whatever `--max-tokens` the run has, and `report --baseline` compares only the
    runs whose reviewer was the same model."""
    [name] = notes.situate("sel", payload={"review": True})
    monkeypatch.setitem(notes.llm.endpoints["qwen3.5-9b"], "thinking", "summary")  # what models.toml says its reasoning is
    a, b, own = tmp_path / "a", tmp_path / "b", tmp_path / "own"
    notes.play(f"{name}@0", a, "--cast", "bench", "--max-tokens", "1000")
    notes.play(f"{name}@0", b, "--cast", "bench", "--max-tokens", "2000", endpoint="qwen3.5-9b")
    notes.play(f"{name}@0", own, "--cast", "self")
    [in_a], [in_b], [in_own] = members(a, name), members(b, name), members(own, name)
    assert in_a.actors == in_b.actors == in_own.actors == {"writer": SUBJECT, "reviewer": "counterpart"}
    assert played_by(in_a) == {"writer": {"gemma-4-26b"}, "reviewer": {"qwen3-32b-mini"}}
    assert played_by(in_b) == {"writer": {"qwen3.5-9b"}, "reviewer": {"qwen3-32b-mini"}}
    assert played_by(in_own) == {"writer": {"gemma-4-26b"}, "reviewer": {"gemma-4-26b"}}
    for ep, subject, counterpart in ((in_a, "gemma-4-26b", "qwen3-32b-mini"), (in_b, "qwen3.5-9b", "qwen3-32b-mini"),
                                     (in_own, "gemma-4-26b", "gemma-4-26b")):
        assert {seat: ep.meta["cast"][cast_name] for seat, cast_name in ep.actors.items()} == {
            "writer": {"endpoint": subject, "model": MODELS[subject]} | ({"thinking": "summary"} if subject == "qwen3.5-9b" else {}),
            "reviewer": {"endpoint": counterpart, "model": MODELS[counterpart]}}
    infos = [json.loads((run / "run.json").read_text()) for run in (a, b, own)]
    assert len({info["id"] for info in infos}) == 3 and infos[0]["cast"] == infos[1]["cast"] != infos[2]["cast"]
    asked = {(r["endpoint"], r["max_tokens"], r["temperature"]) for r in notes.llm.requests}
    assert asked == {("gemma-4-26b", 1000, 0.7), ("qwen3.5-9b", 2000, 0.7), ("qwen3-32b-mini", 300, 0.2),
                     ("gemma-4-26b", 16384, 0.7), ("gemma-4-26b", 300, 0.2)}
    for run in (a, b, own):
        notes("score", "--run", str(run), "--judges", "none")
    cli.main(["report", "--run", str(b), "--judges", "none", "--baseline", str(a)])
    assert "draft_words" in (out := capsys.readouterr().out) and "Δ 0.000 vs baseline" in out, out
    with pytest.raises(SystemExit, match=r"different casts: counterpart is gemma-4-26b \(.*\) here and qwen3-32b-mini"):
        cli.main(["report", "--run", str(own), "--judges", "none", "--baseline", str(a)])


def test_a_seat_the_cast_does_not_name_stops_the_run_before_any_call(notes, monkeypatch):
    [name] = notes.situate("sel", payload={"review": True})
    with pytest.raises(SystemExit, match="seat 'reviewer' is played by 'counterpart', and this run's cast names only subject"):
        notes.play(f"{name}@0")
    monkeypatch.setitem(notes.llm.config["cast"], "mine", {"counterpart": "qwen3-32b-mini", "subject": "qwen3.5-9b"})
    with pytest.raises(SystemExit, match=r"\[cast.mine\] defines subject"):
        notes.play(f"{name}@0", notes.out, "--cast", "mine")
    assert notes.llm.requests == [] and rolled(notes.out, name) == []


def test_a_run_killed_at_the_reviewers_turn_is_resumed_by_the_same_cast_and_by_no_other(notes, tmp_path, killed):
    [name] = notes.situate("sel", payload={"review": True})

    def dies(messages, tools, endpoint):
        raise killed.Killed

    notes.llm.replies["qwen3-32b-mini"].insert(0, dies)
    run = tmp_path / "a"
    killed(*notes.argv(f"{name}@0", run, "--cast", "bench"))
    with pytest.raises(SystemExit, match="holds a different run"):
        notes.play(f"{name}@0", run, "--cast", "self")
    assert "1 open episodes to continue" in notes.play(f"{name}@0", run, "--cast", "bench")
    [member] = members(run, name)
    assert member.status == "completed" and played_by(member) == {"writer": {"gemma-4-26b"}, "reviewer": {"qwen3-32b-mini"}}


def test_a_rollout_of_an_episode_stored_with_its_subject_on_an_endpoint_is_played_by_this_runs_subject(notes):
    """The ten base-* stores were written when a run overwrote its subject's seats with its
    endpoint and listed them in `meta["subject_actors"]`. A rollout of such an episode onto
    another endpoint re-points those seats and no other."""
    with Store(notes.out / "store.db") as store:
        old = Episode(ep_id="old", task_id="old", suite="notes", split="dev", actors={"writer": "gemma-4-26b", "author": "scripted"},
                      actor_tools={"writer": ["episode_write"]},
                      meta={"temperature": 0.7, "endpoint": "gemma-4-26b", "subject_actors": ["writer"]})
        store.append(old)
        store.close(old)
    notes.play("old@0", endpoint="qwen3.5-9b")
    [member] = members(notes.out, "old")
    assert member.actors == {"writer": SUBJECT, "author": "scripted"} and "subject_actors" not in member.meta
    assert played_by(member)["writer"] == {"qwen3.5-9b"} and member.meta["cast"]["subject"]["endpoint"] == "qwen3.5-9b"
