"""`wa-bench run`, `score` and `report` end to end over a scripted model: what reaches the
scores file and what the report prints."""

import copy
import json

import pytest

from writing_assistant import cli
from writing_assistant.episode import store

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


def test_rollouts_are_scored_against_the_source_twin_and_stay_out_of_the_runs_numbers(bench, capsys):
    """A rollout says which checkpoint it replays, so rollouts of two checkpoints are all made;
    it gets `net_*` from its source's stored twin; a twin has no rollouts; and the report of
    the run's own episodes is the same with the rollouts stored beside them, which it lists
    apart, per checkpoint."""
    out = bench([POV_ROW], chapter("Mira turned the glove over."))
    run_and_score(capsys, out)
    cli.main(["report", "--run", str(out)])
    before = capsys.readouterr().out
    run_and_score(capsys, out, "--from", "pov-1~full_view@0", "--rollouts", "2")
    run_and_score(capsys, out, "--from", "pov-1~full_view@1", "--rollouts", "2")
    with pytest.raises(SystemExit, match="is a twin"):
        cli.main(["run", "--suite", "1a", "--endpoint", "gemma-4-26b", "--out", str(out),
                  "--from", "pov-1~full_view~twin_swap@0", "--rollouts", "2"])
    rows = {r["ep_id"]: r for r in score_rows(out)}
    arm = rows["pov-1~full_view"]
    rollouts = [rows[f"pov-1~full_view@{at}.{i}"] for at in (0, 1) for i in (0, 1)]
    assert len(rows) == 8 and [r["twin"] for r in rollouts] == [arm["twin"]] * 4
    assert [r["metrics"]["net_pov_leak"] for r in rollouts] == [arm["metrics"]["net_pov_leak"]] * 4
    cli.main(["report", "--run", str(out)])
    report = capsys.readouterr().out
    head, _, apart = report.partition("\nrollouts, by the checkpoint each was forked from")
    assert head == before, report
    assert [g.split(":")[0] for g in apart.splitlines() if g.startswith("== ")] == [
        "== rollout_of=pov-1~full_view@0, condition=full_view", "== rollout_of=pov-1~full_view@1, condition=full_view"], report


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
