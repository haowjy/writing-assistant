"""`wa-bench run`, `score` and `report` end to end over a scripted model: what reaches the
scores file and what the report prints."""

import copy
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


def score_rows(out) -> list[dict]:
    return [json.loads(line) for line in (out / "scores-eval.jsonl").read_text().splitlines()]


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


def test_report_warns_about_any_scores_row_without_the_scoring_code(bench, capsys):
    out = bench([POV_ROW], chapter("Mira turned the glove over."))
    run_and_score(capsys, out)
    rows = score_rows(out)
    rows[0].pop("code", None)
    (out / "scores-eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cli.main(["report", "--run", str(out)])
    assert "1 scores row written before the scoring code was stamped" in capsys.readouterr().out


def test_rollouts_carry_their_own_metrics_and_stay_out_of_the_runs_numbers(bench, capsys):
    """A rollout says which checkpoint it replays, so rollouts of two checkpoints are all made.
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
    for row in (rows[f"pov-1~full_view{twin}@{at}.{i}"] for twin, at in (("", 0), ("", 1), ("~twin_swap", 0)) for i in (0, 1)):
        assert row["twin"] is None and row["metrics"]["pov_leak"] == 0.0
        assert not [m for m in row["metrics"] if "twin" in m or m.startswith("net_")], row
    cli.main(["report", "--run", str(out)])
    report = capsys.readouterr().out
    head, _, apart = report.partition("\nrollouts, by the checkpoint each was forked from")
    assert head == before, report
    assert [g.split(":")[0] for g in apart.splitlines() if g.startswith("== ")] == [
        "== rollout_of=pov-1~full_view@0, condition=full_view", "== rollout_of=pov-1~full_view@1, condition=full_view",
        "== rollout_of=pov-1~full_view~twin_swap@0, condition=twin_swap"], report
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
