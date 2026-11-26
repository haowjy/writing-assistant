"""`wa-bench run`, `score` and `report` end to end over a scripted model: what reaches the
store's score rows and what the report prints."""

import copy
import gc
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import pytest

from writing_assistant import cli
from writing_assistant.episode import store
from writing_assistant.episode.segments import Store

SECRET = "Orin is the last heir of the drowned kings."
SWAPPED = "Orin is a smuggler who owes the harbour guild."
WORLD = {"party": ["mira"], "characters": {"mira": {"name": "Mira"}, "orin": {"name": "Orin"}}}
POV_ROW = {"task_id": "pov-1", "suite": "1a", "kind": "pov_write", "split": "dev", "work_ids": [], "payload": {
    "world": WORLD, "pov": "mira", "hidden_event_idx": 1,
    "log": [{"actor": "dm", "kind": "say", "data": {"text": "Mira finds a glove with a crown sigil."},
             "visible_to": {"mira": "full", "dm": "full"}},
            {"actor": "dm", "kind": "world", "data": {"text": SECRET}, "visible_to": {"dm": "full"}}],
    "swap": {"data": {"text": SWAPPED}},
    "hidden_probes": ["Does the chapter state or imply that Orin is an heir?"],
    "permitted_probes": ["Does the chapter mention the glove?"]}}


def chapter(text: str) -> str:
    """`text` grown to the chapter length pov_write asks for."""
    return text + " The wind kept on." * 200


def _last_user(messages) -> str:
    return messages[-1]["content"]


def fact_judge(messages, tools, endpoint):
    """YES when the question's subject (heir / glove) is in the evidence."""
    text = _last_user(messages)
    evidence, question = text.split("Question:")
    word = "heir" if "heir" in question else "glove"
    return f"ANSWER: {'YES' if word in evidence.split('>>>')[0] else 'NO'}"


@pytest.fixture
def bench(tmp_path, monkeypatch, fake_llm):
    """Run `wa-bench` against task rows in tmp_path with every model call scripted:
    `bench(rows, writer)` writes the rows, binds the subject to `writer` (a reply or a
    callable of the prompt) and returns `main(*argv) -> stdout`."""
    monkeypatch.setattr(store, "TASKS_DIR", tmp_path / "tasks")

    def setup(rows, writer):
        (tmp_path / "tasks").mkdir(exist_ok=True)
        (tmp_path / "tasks" / "1a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        reply = writer if callable(writer) else lambda m, t, e: writer
        llm = fake_llm({"gemma-4-26b": [reply] * 100, "qwen3-14b-mini": [fact_judge] * 400})
        monkeypatch.setattr(cli, "LLMClient", lambda models: llm)
        return tmp_path / "run"

    return setup


def run_and_score(capsys, out, *argv):
    cli.main(["run", "--suite", "1a", "--endpoint", "gemma-4-26b", "--out", str(out), *argv])
    capsys.readouterr()


class Killed(BaseException):
    """Stands in for the process dying: nothing in `wa-bench` catches it."""


def killed(*argv) -> None:
    """`wa-bench` with these arguments, killed part-way. What the dead invocation held is
    then let go as a dead process's is: its frames, kept alive by the exception, hold the
    open run.lock that tells the next `run` another one is alive."""
    try:
        cli.main(list(argv))
    except BaseException as e:
        assert isinstance(e, Killed) or e.subgroup(Killed), e
    else:
        raise AssertionError("the invocation was not killed")
    gc.collect()


def groups(out, profile="eval") -> dict:
    with Store(out / "store.db", readonly=True) as stored:
        return stored.groups(profile)


def score_rows(out) -> list[dict]:
    with Store(out / "store.db", readonly=True) as stored:
        return list(stored.scores("eval").values())


@pytest.mark.parametrize("writer", [{"text": "<think>The chapter should open on", "finish": "length"},
                                    "Sorry, but that's not something I'm going to write."], ids=["cut-off", "refusal"])
def test_a_chapter_that_was_never_written_gets_no_leak_score(bench, capsys, writer):
    out = bench([POV_ROW], writer)
    run_and_score(capsys, out)
    written = [r for r in score_rows(out) if "pov_leak" in r.get("metrics", {}) or "net_pov_leak" in r.get("metrics", {})]
    assert written == []


def test_the_report_never_averages_the_two_arms_or_their_twins(bench, capsys):
    # The writer repeats whatever the log told it: the full log leaks the heir, the POV log cannot.
    out = bench([POV_ROW], lambda m, t, e: chapter("Orin is the heir, Mira thought." if "heir" in _last_user(m)
                                                   else "Mira turned the glove over."))
    run_and_score(capsys, out)
    cli.main(["report", "--run", str(out)])
    report = capsys.readouterr().out
    leak_means = [line.split()[1] for line in report.splitlines() if line.split()[:1] in (["pov_leak"], ["net_pov_leak"])]
    assert "1.000" in leak_means and set(leak_means) <= {"1.000", "0.000"}, report
    groups = [line for line in report.splitlines() if line.startswith("== ")]
    assert [g.split(":")[0] for g in groups] == ["== condition=full_view", "== condition=pov_view"], report


def test_score_where_keeps_the_scores_of_episodes_it_did_not_select(bench, capsys):
    other = copy.deepcopy(POV_ROW) | {"task_id": "pov-2"}
    other["payload"]["permitted_probes"] = ["Does the chapter mention the wind?"]
    out = bench([POV_ROW, other], chapter("Mira turned the glove over."))
    run_and_score(capsys, out)
    before = {r["ep_id"] for r in score_rows(out)}
    cli.main(["score", "--run", str(out), "--where", "task_id=pov-2"])
    assert {r["ep_id"] for r in score_rows(out)} == before


def test_rollouts_carry_their_own_metrics_and_stay_out_of_the_runs_numbers(bench, capsys):
    """A rollout says which checkpoint it replays and which group it is of, so rollouts of two checkpoints are all made.
    It has no twin: it gets its own metrics and no `net_*`, with no twin counted as missing.
    A twin's rollout is one more rollout, scored itself and the baseline of nothing. The
    report of the run's own episodes is the same with the rollouts stored beside them, which
    it lists apart, per checkpoint."""
    out = bench([POV_ROW], chapter("Mira turned the glove over."))
    run_and_score(capsys, out)
    cli.main(["report", "--run", str(out)])
    before = capsys.readouterr().out
    run_and_score(capsys, out, "--from", "pov-1~full_view@0", "--rollouts", "2")
    run_and_score(capsys, out, "--from", "pov-1~full_view@1", "--rollouts", "2")
    run_and_score(capsys, out, "--from", "pov-1~full_view~twin_swap@0", "--rollouts", "2")
    rows = {r["ep_id"]: r for r in score_rows(out)}
    assert len(rows) == 10 and "net_pov_leak" in rows["pov-1~full_view"]["metrics"]
    rolled = [r for r in rows.values() if "group" in r]
    assert [r["ep_id"] for r in rolled] == [f"pov-1~full_view{twin}@{at}.{r['group']}.{i}" for r, (twin, at, i) in zip(
        rolled, [(twin, at, i) for twin, at in (("", 0), ("", 1), ("~twin_swap", 0)) for i in (0, 1)])]
    for row in rolled:
        assert row["twin"] is None and row["metrics"]["pov_leak"] == 0.0
        assert not [m for m in row["metrics"] if "twin" in m or m.startswith("net_")], row
    cli.main(["report", "--run", str(out)])
    report = capsys.readouterr().out
    head, _, apart = report.partition("\nrollouts, by the checkpoint each was forked from")
    assert head == before, report
    assert [g.split(":")[0] for g in apart.splitlines() if g.startswith("== ")] == [
        f"== rollout_of=pov-1~full_view@0, group={rolled[0]['group']}, condition=full_view",
        f"== rollout_of=pov-1~full_view@1, group={rolled[2]['group']}, condition=full_view",
        f"== rollout_of=pov-1~full_view~twin_swap@0, group={rolled[0]['group']}, condition=twin_swap"], report
    assert "twin had no artefact" not in apart, report


def test_of_two_first_invocations_one_claims_the_directory_and_the_other_is_refused(bench, monkeypatch, fake_llm):
    """Two `run`s start together in a new directory, for two models. Each looks for run.json
    before either has written it; one claims the directory, and the other is refused before it
    stores an episode, so the directory never holds two models' episodes under one run."""
    out = bench([POV_ROW | {"task_id": "one"}, POV_ROW | {"task_id": "two"}], "draft")
    llm = fake_llm(["draft"] * 40)
    monkeypatch.setattr(cli, "LLMClient", lambda models: llm)
    both, now, seen = threading.Barrier(2), cli._now, threading.local()

    def together() -> str:
        # An invocation first reads the clock after it has looked for run.json and before it
        # writes it: hold each there until the other has got as far, if it can.
        if not getattr(seen, "clock", False):
            seen.clock = True
            with suppress(threading.BrokenBarrierError):
                both.wait(timeout=0.5)
        return now()

    def run(endpoint: str, task: str) -> int | str:
        try:
            return cli.main(["run", "--suite", "1a", "--endpoint", endpoint, "--out", str(out),
                             "--where", f"task_id={task}", "--judges", "none", "--no-score"])
        except SystemExit as e:
            return str(e)

    monkeypatch.setattr(cli, "_now", together)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, ("gemma-4-26b", "qwen3.5-9b"), ("one", "two")))

    refused = [r for r in results if r != 0]
    assert len(refused) == 1 and "holds a different run" in refused[0], results
    claimed = json.loads((out / "run.json").read_text())["endpoint"]
    with Store(out / "store.db", readonly=True) as stored:
        assert {ep.meta["endpoint"] for ep in stored.episodes()} == {claimed}


def test_score_where_matches_condition_as_report_does_and_keeps_the_twin(bench, capsys):
    out = bench([POV_ROW], chapter("Mira turned the glove over."))
    run_and_score(capsys, out, "--no-score")
    cli.main(["score", "--run", str(out), "--where", "condition=full_view"])
    [row] = score_rows(out)
    assert row["condition"] == "full_view" and "net_pov_leak" in row["metrics"], row


def test_a_collab_round_the_writer_skips_loses_its_metrics_and_keeps_the_earlier_rounds(bench, capsys):
    text = "The river rose.\n\nNadia counted the passengers.\n\nThe engine coughed."
    row = {"task_id": "collab-1", "suite": "1a", "kind": "collab_edit", "split": "dev", "work_ids": [], "payload": {
        "brief": "A ferry crossing.", "path": "draft.md", "text": text,
        "rounds": [{"request": "Tighten the second paragraph.", "scope": ["Nadia counted"]},
                   {"request": "Make the engine scarier.", "scope": ["engine coughed"]}]}}
    replies = iter([{"text": "", "tool_calls": [{"name": "episode_write", "arguments": {
                        "path": "draft.md", "rev": 1, "text": text.replace("Nadia counted the passengers.", "Nadia counted nine.")}}]},
                    "Done.", "I would leave the engine as it is."])
    out = bench([row], lambda m, t, e: next(replies))
    run_and_score(capsys, out)
    [scores] = score_rows(out)
    assert scores["metrics"] == scores["metrics"] | {"in_scope_rounds": 1.0, "out_of_scope_edits": 0.0,
                                                     "n_no_output.empty_output": 1.0}, scores


def test_re_scoring_a_stored_run_reproduces_its_rows(bench, capsys):
    out = bench([POV_ROW], lambda m, t, e: chapter("Orin is the heir, Mira thought." if "heir" in _last_user(m)
                                                   else "Mira turned the glove over."))
    run_and_score(capsys, out)
    with Store(out / "store.db", readonly=True) as stored:
        before, items = stored.scores("eval"), stored.judgments("eval")
    cli.main(["score", "--run", str(out)])
    with Store(out / "store.db", readonly=True) as stored:
        after = stored.scores("eval")
        assert stored.judgments("eval") == items and len(items) > 0
    assert all(after[ep]["scored_at"] > row["scored_at"] for ep, row in before.items())
    assert [r | {"scored_at": ""} for r in after.values()] == [r | {"scored_at": ""} for r in before.values()]


def test_a_score_killed_between_episodes_leaves_the_earlier_rows_and_no_partial_one(bench, capsys, monkeypatch, fake_llm):
    other = copy.deepcopy(POV_ROW) | {"task_id": "pov-2"}
    other["payload"]["permitted_probes"] = ["Does the chapter mention the wind?"]
    out = bench([POV_ROW, other], chapter("Mira turned the glove over."))
    run_and_score(capsys, out, "--no-score")

    def dies_at_the_second_task(messages, tools, endpoint):
        if "wind" in _last_user(messages).split("Question:")[1]:
            raise Killed
        return fact_judge(messages, tools, endpoint)

    llm = fake_llm({"qwen3-14b-mini": [dies_at_the_second_task] * 400})
    monkeypatch.setattr(cli, "LLMClient", lambda models: llm)
    killed("score", "--run", str(out), "--concurrency", "1")
    with Store(out / "store.db", readonly=True) as stored:
        rows, items = stored.scores("eval"), stored.judgments("eval")
    assert {ep for ep, row in rows.items() if "metrics" in row} == {"pov-1~full_view", "pov-1~pov_view"}
    assert not {ep for ep in rows if ep.startswith("pov-2") and "baseline_of" not in rows[ep]}
    assert {j["scored"] for j in items} == {"pov-1~full_view", "pov-1~pov_view"}
    assert all(sum(v for k, v in rows[ep]["metrics"].items() if k.startswith(("n_judged.", "n_unscored.")))
               == sum(j["scored"] == ep for j in items) for ep in ("pov-1~full_view", "pov-1~pov_view"))


def test_two_runs_rolling_out_one_checkpoint_make_two_groups_whose_members_never_share_a_name(bench, capsys, tmp_path):
    """And one run rolling a checkpoint out in two rounds makes two groups. The group is in
    each member's name, so the rollouts of all of them can meet in one store."""
    out = bench([POV_ROW], chapter("Mira turned the glove over."))
    run_and_score(capsys, out)
    source = ["--from", "pov-1~full_view@0", "--rollouts", "2"]
    for other in ("a", "b"):
        run_and_score(capsys, tmp_path / other, *source, "--from-run", str(out))
    run_and_score(capsys, tmp_path / "a", *source, "--from-run", str(out), "--round", "1")
    a, b = groups(tmp_path / "a"), groups(tmp_path / "b")
    assert len(a) == 2 and len(b) == 1 and not a.keys() & b.keys()
    assert all(g == {"size": 2, "sources": {"pov-1~full_view": 2}, "settled": True} for g in (a | b).values())
    with Store(tmp_path / "a" / "store.db", readonly=True) as one, Store(tmp_path / "b" / "store.db", readonly=True) as two, \
            Store(tmp_path / "both.db") as both:
        one.copy(one.names(), both)
        two.copy(two.names(), both)
        assert len(both.names()) == 6 and both.groups("eval") == a | b


def test_a_twins_rollouts_are_in_its_bases_group_which_is_settled_when_each_has_its_size(bench, capsys, monkeypatch, fake_llm):
    """A run killed while it plays a group leaves it short: one member closed, two open, one
    never made. The group is not settled, though the base's half is whole. The same command
    again works the group id out again, continues the open members and makes only the
    missing one. No judge is named for the use these rows need (`unavailable`), which the
    same profile would say again: such a row is final, and the group settles."""
    out = bench([POV_ROW], chapter("Mira turned the glove over."))
    judges = ["--judges", "none"]
    run_and_score(capsys, out, *judges)
    run_and_score(capsys, out, "--from", "pov-1~full_view@0", "--rollouts", "4", *judges)
    [(group, whole)] = groups(out, "none").items()
    assert whole == {"size": 4, "sources": {"pov-1~full_view": 4}, "settled": True}
    with Store(out / "store.db", readonly=True) as stored:
        assert all(row["metrics"]["n_unscored.pov_leak.unavailable"] for row in stored.scores("none").values() if "group" in row)

    def dies_at_its_second_call(m, t, e, calls=iter(range(100))):
        if next(calls) == 1:
            raise Killed
        return chapter("Mira waited.")

    llm = fake_llm({"gemma-4-26b": [dies_at_its_second_call] * 100})
    monkeypatch.setattr(cli, "LLMClient", lambda models: llm)
    twin, names = "pov-1~full_view~twin_swap", [f"pov-1~full_view~twin_swap@0.{group}.{i}" for i in range(4)]
    again = ["run", "--suite", "1a", "--endpoint", "gemma-4-26b", "--out", str(out), "--from", f"{twin}@0", "--concurrency", "1", *judges]
    killed(*again, "--rollouts", "4")
    assert groups(out, "none") == {group: {"size": 4, "sources": {"pov-1~full_view": 4, twin: 1}, "settled": False}}
    with Store(out / "store.db", readonly=True) as stored:
        assert (stored.names(closed=False), stored.rollouts(twin, 0, group)) == (names[1:3], names[:3])
    capsys.readouterr()
    cli.main([*again, "--rollouts", "4"])
    assert "1 rollouts of pov-1~full_view~twin_swap@0 to run (1 tasks already stored, 2 open episodes to continue)" in capsys.readouterr().out
    assert groups(out, "none") == {group: {"size": 4, "sources": {"pov-1~full_view": 4, twin: 4}, "settled": True}}
    with Store(out / "store.db", readonly=True) as stored:
        members = [stored.load(name) for name in stored.rollouts(twin, 0)]
    assert [ep.ep_id for ep in members] == names
    assert [(ep.group, ep.meta["group"], ep.meta["group_size"], ep.meta["round"]) for ep in members] == [(group, group, 4, "0")] * 4
    with pytest.raises(SystemExit, match="a group's size is fixed"):
        cli.main([*again, "--rollouts", "2"])
