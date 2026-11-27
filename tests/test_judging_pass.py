"""The judging pass (`wa-bench score --world`): a world's closed rollouts scored under the
rubrics their situations list, by judges that read the stored record and nothing else."""

import asyncio
import json
import re
import threading
from pathlib import Path

import pytest

from writing_assistant import cli
from writing_assistant.episode import Episode
from writing_assistant.episode.library import Library
from writing_assistant.episode.segments import judging
from writing_assistant.run import directory, situate
from writing_assistant.run.control import Settings, Work, play_groups
from writing_assistant.run.directory import services_for
from writing_assistant.run.scoring import score_episode
from writing_assistant.suites import SUITES

MODELS_TOML = Path(__file__).resolve().parents[1] / "models.toml"
JUDGE = "qwen3-14b-mini"


def talk(brief: str, **payload) -> dict:
    return {"request": "Plan the crossing.", "max_turns": 6, "script": [{"say": "And then?"}] * 4,
            "seats": {"author": {"cast": "scripted"}, "writer": {"cast": "subject", "brief": brief}}} | payload


def subject(messages, tools, endpoint):
    """Thinks before it answers only when its brief asks."""
    return ("<think>Short clauses, for the tide.</think>" if "think" in messages[0]["content"] else "") + "We sail at dawn."


def judge(messages, tools, endpoint):
    """Answers each prompt as it asks, citing turns it was shown."""
    prompt, record = messages[0]["content"], messages[1]["content"]
    turns = [int(n) for n in re.findall(r"^## Turn (\d+)", record, re.MULTILINE)]
    if prompt.startswith("# Thinking about prose"):
        target = int(re.search(r"^Target: turn (\d+)", record, re.MULTILINE)[1])
        found = [{"covers": {"turn": target}, "dimension": "thinking", "criterion": "writing_decision", "value": 1,
                  "cites": [{"turn": target, "note": "chose short clauses"}], "evidence": []}]
    elif prompt.startswith("# World coherence"):
        found = [{"covers": "task", "dimension": "output", "criterion": "continuity", "value": 2,
                  "cites": [{"turn": turns[0], "note": "names the tide"}, {"turn": turns[-1], "note": "keeps to it"}],
                  "evidence": [{"path": "kb.md", "revision": 1}]},
                 {"covers": "task", "dimension": "output", "criterion": "visibility", "unscored": "not_applicable"}]
    else:
        found = [{"covers": "task", "dimension": "output", "criterion": "voice_consistency", "value": 0, "cites": [], "evidence": []}]
    return "```json\n" + json.dumps({"findings": found}) + "\n```"


@pytest.fixture
def world(tmp_path, monkeypatch, fake_llm, capsys):
    """The world `sel` with three situations, a group of two rollouts of each played into
    it, and `world.score(*argv) -> stdout`. `plain` lists nothing (the kind's defaults),
    `judged` lists two judge-read rubrics and a name no rubric answers to, and `quiet` is
    played without thinking. The profile `j` names an endpoint for every rubric's judge."""
    monkeypatch.setenv("WA_LIBRARY", str(tmp_path / "library"))
    llm = fake_llm({"gemma-4-26b": [subject] * 400, JUDGE: [judge] * 400})
    llm.config["judges"]["j"] = dict.fromkeys(("world_coherence", "thinking_about_prose", "character_voice_consistency"), JUDGE)
    monkeypatch.setattr(directory, "LLMClient", lambda models: llm)
    lib = Library()
    lib.create(Episode(ep_id="sel", task_id="sel", suite="root", split="dev", files={"kb.md": "The tide turns at six."},
                       file_revs={"kb.md": 1}))
    rubrics = ["world_coherence", "thinking_about_prose", "made_up"]
    with lib.writing("sel") as store:
        for name, payload in (("plain", talk("You are the writer.")), ("judged", talk("You think aloud.", rubrics=rubrics)),
                              ("quiet", talk("You are the writer.", rubrics=["thinking_about_prose"]))):
            situate(store, services_for(MODELS_TOML, "none"), "situations", "converse", payload, "sel", name=f"sel/{name}")
    lib.settings = Settings(endpoint="gemma-4-26b", cast=None, max_tokens=256, concurrency=4, models=MODELS_TOML,
                            runs=tmp_path / "runs")
    lib.work = [Work("sel", f"sel/{name}", 2) for name in ("plain", "judged", "quiet")]
    lib.groups = dict(zip(("plain", "judged", "quiet"), asyncio.run(play_groups(lib, lib.work, lib.settings)).values(), strict=True))
    capsys.readouterr()

    def score(*argv, code=0) -> str:
        assert cli.main(["--models", str(MODELS_TOML), "score", "--world", "sel", "--judges", "j", *argv]) == code
        return capsys.readouterr().out

    lib.llm, lib.score = llm, score
    return lib


def stored(lib) -> dict[str, dict[str, dict]]:
    """rubric -> ep_id -> the row read, for profile `j`."""
    with lib.open("sel") as home:
        return {name.partition("/")[2]: home.scores(name) for name in home.judgings("j")}


def all_rows(lib) -> list:
    with lib.open("sel") as home:
        return home.db.execute("SELECT id, ep_id, profile, row FROM scores ORDER BY id").fetchall()


def test_each_rollout_gets_its_situations_rubrics_once_and_findings_of_a_task_and_of_a_turn_are_stored_alike(world):
    out = world.score()
    assert "rubric 'made_up': listed by 2 episodes" in out
    rows = stored(world)
    with world.open("sel") as home:
        name_of = {ep_id: source.removeprefix("sel/") for ep_id, (_, source, _) in home.judged("j").items()}
        by_rubric = {rubric: sorted({name_of[ep_id] for ep_id in found}) for rubric, found in rows.items()}
        assert by_rubric == {"counts": ["plain"], "world_coherence": ["judged"], "thinking_about_prose": ["judged", "quiet"]}
        assert all(len(found) == 2 * len(by_rubric[rubric]) for rubric, found in rows.items())
        # Settled is per rubric: a group is settled for the rubrics its rollouts were scored under.
        assert home.settled(judging("j", "world_coherence"))[world.groups["judged"]]["settled"]
        assert not home.settled(judging("j", "world_coherence"))[world.groups["plain"]]["settled"]
        items = home.judgments(judging("j", "world_coherence")) + home.judgments(judging("j", "thinking_about_prose"))
        ep = home.load(next(iter(rows["world_coherence"])))
    [row] = [r for r in rows["world_coherence"].values() if r["ep_id"] == ep.ep_id]
    assert row["metrics"] == {"output.continuity": 1.0, "n_judged.output.continuity": 1.0,
                              "n_unscored.output.visibility.not_applicable": 1.0}
    assert [stamp["use"] for stamp in row["judges"]] == ["world_coherence"] and row["judges"][0]["endpoint"] == JUDGE
    turns = [e.data["turn_idx"] for e in ep.events if e.kind == "deliver"]
    task = next(j for j in items if j["scored"] == ep.ep_id and j.get("criterion") == "continuity")
    per_turn = [j for j in items if j["scored"] == ep.ep_id and j.get("criterion") == "writing_decision"]
    assert (task["covers"], task["seat"], task["value"]) == ("task", "writer", 2)
    assert [c["turn"] for c in task["cites"]] == [turns[0], turns[-1]] and task["evidence"] == [{"path": "kb.md", "revision": 1}]
    assert [j["covers"] for j in per_turn] == [{"turn": n} for n in turns] and {j["value"] for j in per_turn} == {1}
    # A rollout played without thinking: nobody is asked, and each turn says why it has no score.
    quiet = [j for j in items if name_of[j["scored"]] == "quiet"]
    assert quiet and all(j["unscored"] == "no_thinking" and "judge" not in j for j in quiet)
    assert all(r["metrics"] == {"n_unscored.thinking.writing_decision.no_thinking": float(len(turns))}
               for ep_id, r in rows["thinking_about_prose"].items() if name_of[ep_id] == "quiet")
    # An item holds references and what the judge answered, never the text it was shown.
    assert all("tide turns at six" not in json.dumps(j) and "We sail at dawn" not in json.dumps(j) for j in items)
    # A second pass has nothing to do: no row and no judge call.
    before, calls = all_rows(world), len(world.llm.requests)
    assert ": 0 episodes" in world.score()
    assert all_rows(world) == before and len(world.llm.requests) == calls


def test_another_rubric_run_over_a_judged_world_adds_its_rows_and_changes_none(world):
    world.score()
    before = all_rows(world)
    reading = stored(world)
    assert "6 episodes, 6 rows: 6 scored" in world.score("--rubric", "character_voice_consistency")
    after = stored(world)
    assert set(before) < set(all_rows(world)) and len(all_rows(world)) == len(before) + 6
    assert {k: v for k, v in after.items() if k != "character_voice_consistency"} == reading
    assert {r["metrics"]["output.voice_consistency"] for r in after["character_voice_consistency"].values()} == {0.0}
    with world.open("sel") as home:
        assert all(g["settled"] for g in home.settled(judging("j", "character_voice_consistency")).values())
        assert not all(g["settled"] for g in home.settled(judging("j", "world_coherence")).values())
    with pytest.raises(SystemExit, match="no suite played in sel"):
        world.score("--rubric", "no_such")


def test_a_judge_that_is_down_leaves_pending_rows_which_pending_scores_again(world):
    def down(messages, tools, endpoint):
        raise ConnectionError("judge down")

    world.llm.replies[JUDGE] = [down] * 50
    assert "scorer errors" in world.score(code=1)
    rows = stored(world)
    assert all("error" in r for r in rows["world_coherence"].values()) and all("metrics" in r for r in rows["counts"].values())
    with world.open("sel") as home:
        assert home.judged("j")[next(iter(rows["world_coherence"]))][2]["world_coherence"] is None
        assert not home.settled(judging("j", "world_coherence"))[world.groups["judged"]]["settled"]
    world.llm.replies[JUDGE] = [judge] * 50
    out = world.score("--pending")
    rows = stored(world)
    assert "2 episodes" in out and all("metrics" in r for found in rows.values() for r in found.values())
    with world.open("sel") as home:
        assert home.settled(judging("j", "world_coherence"))[world.groups["judged"]]["settled"]
    assert ": 0 episodes" in world.score("--pending") and ": 0 episodes" in world.score()


def test_a_pass_beside_a_run_that_is_pushing_scores_what_is_closed_and_the_next_scores_the_rest(world):
    """The pass and the controller write one world file at once; neither waits for the other
    to finish, and after a last pass every rollout has exactly one row read per rubric."""
    more = [Work("sel", f"sel/{name}", 3, round="1") for name in ("plain", "judged", "quiet")]
    passes, pushed = [], threading.Event()
    push = Library.push

    def push_then_judge(self, src, names=None, scores=True):
        added = push(self, src, names, scores)
        if names and not pushed.is_set():  # the first rollout is in the world and the others are still in play
            pushed.set()
            thread = threading.Thread(target=lambda: passes.append(world.score()))
            thread.start()
            thread.join()
        return added

    Library.push = push_then_judge
    try:
        asyncio.run(play_groups(world, more, world.settings))
    finally:
        Library.push = push
    assert passes and "episodes" in passes[0]
    world.score()
    with world.open("sel") as home:
        state = home.judged("j")
        assert len(state) == 15 and all(has and None not in has.values() for _, _, has in state.values())
        assert all(len(rows) == len(set(ep for ep, *_ in rows)) for rows in
                   [home.db.execute("SELECT ep_id FROM scores WHERE profile = ?", (name,)).fetchall() for name in home.judgings("j")])
    assert ": 0 episodes" in world.score()


class Shared:
    """Two rubrics a suite answers from one judge call: the call's findings are split
    between them by what each covers."""

    rubrics = {"whole": (), "turns": ()}
    needs: dict = {}

    async def score(self, ep, services, tallies, twin=None):
        judge_ = services.judges.findings["world_coherence"]
        j = await judge_.judge("the record")
        for rubric, tally in tallies.items():
            tally.found([f for f in j.value if (f["covers"] == "task") == (rubric == "whole")], judge_, j.replies)
        return {rubric: tally.rates() for rubric, tally in tallies.items()}


async def test_two_rubrics_may_share_one_judge_call_and_each_gets_its_own_row(world):
    both = json.dumps({"findings": [
        {"covers": "task", "dimension": "output", "criterion": "arc", "value": 2},
        {"covers": {"turn": 1}, "dimension": "output", "criterion": "line", "value": 1},
        {"covers": {"turn": 3}, "dimension": "output", "criterion": "line", "value": 2}]})
    world.llm.replies[JUDGE] = [both]
    with world.open("sel") as home:
        ep = home.load(home.names()[0])
    calls = len(world.llm.requests)
    scored = await score_episode(Shared(), ep, None, services_for(MODELS_TOML, "j"), ["whole", "turns"])
    assert len(world.llm.requests) == calls + 1
    assert scored["whole"][0]["metrics"] == {"output.arc": 1.0, "n_judged.output.arc": 1.0}
    assert scored["turns"][0]["metrics"] == {"output.line": 0.75, "n_judged.output.line": 2.0}
    assert [j["covers"] for j in scored["turns"][1]] == [{"turn": 1}, {"turn": 3}]
    assert SUITES["situations"].rubrics.keys() >= {"counts", "world_coherence", "character_voice_consistency", "thinking_about_prose"}


def test_a_label_file_holds_the_record_the_judge_read_and_the_labelling_page_shows_it(world, tmp_path, monkeypatch):
    """Stored findings hold references only (asserted above); the sampler builds the text
    again from the stored rollout, so the standalone labelling page needs nothing else."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_calibration

    from writing_assistant import label_ui

    world.score()
    with world.open("sel") as home:
        items = build_calibration.finding_items(home, "sel", "world_coherence", "j")
        per_turn = build_calibration.finding_items(home, "sel", "thinking_about_prose", "j")
    assert len(items) == 2 and len(per_turn) == 6  # scored findings only: `not_applicable` and `no_thinking` are left out
    file = tmp_path / "labels.jsonl"
    file.write_text("".join(json.dumps({"id": f"i{n}"} | item) + "\n" for n, item in enumerate(items + per_turn)))
    shown = label_ui.view(label_ui.Store(file).next_item())
    assert shown["kind"] == "yes_no" and "scored output.continuity at 2 of 2" in shown["question"]
    assert "The tide turns at six." in shown["evidence"] and "We sail at dawn." in shown["evidence"]
    # A turn's finding is shown the record up to that turn and no further.
    first = min(per_turn, key=lambda item: len(item["evidence"]))
    assert "Target: turn" in first["evidence"] and "# Files when the rollout ended" not in first["evidence"]


def test_a_profile_that_names_no_judge_for_a_listed_rubric_is_refused_before_any_row_is_written(world):
    world.llm.config["judges"]["half"] = {"thinking_about_prose": JUDGE}
    with pytest.raises(SystemExit, match="world_coherence.*no endpoint|no endpoint.*world_coherence"):
        cli.main(["--models", str(MODELS_TOML), "score", "--world", "sel", "--judges", "half"])
    with world.open("sel") as home:
        assert home.judgings("half") == []
    # What needs no such judge can still be asked for by name.
    assert cli.main(["--models", str(MODELS_TOML), "score", "--world", "sel", "--judges", "half", "--rubric", "counts"]) == 0
    with world.open("sel") as home:
        assert home.judgings("half") == ["half/counts"]


def test_a_row_stands_only_while_its_judges_are_the_ones_the_profile_names_now(world, monkeypatch):
    """Already judged is the row's stamp against today's: a prompt edit, another endpoint, and
    a row an earlier build wrote with no judge configured are each judged again, once."""
    from writing_assistant.judges.findings import PROMPTS, version

    world.score()
    with world.open("sel") as home:
        judged = sorted(home.scores(judging("j", "world_coherence")))
        # What a pass left before it refused an unconfigured profile: a final row, nothing scored, no endpoint.
        stale = [{"ep_id": ep_id, "rubric": "character_voice_consistency", "scored_at": "2026-11-26T10:00:00.000000Z",
                  "metrics": {"n_unscored.output.voice_consistency.unavailable": 1.0},
                  "judges": [{"use": "character_voice_consistency", "endpoint": "", "model": "", "prompt": "x"}]} for ep_id in judged]
    with world.writing("sel") as store:
        store.put_scores(judging("j", "character_voice_consistency"), stale)
    out = world.score("--rubric", "character_voice_consistency")
    assert "6 episodes" in out and stored(world)["character_voice_consistency"][judged[0]]["metrics"]["output.voice_consistency"] == 0.0
    assert ": 0 episodes" in world.score() and ": 0 episodes" in world.score("--rubric", "character_voice_consistency")

    monkeypatch.setitem(PROMPTS, "world_coherence", PROMPTS["world_coherence"] + "\nBe strict.")
    calls = len(world.llm.requests)
    out = world.score()
    assert "2 episodes" in out and "rubrics world_coherence:" in out and len(world.llm.requests) == calls + 2
    assert {r["judges"][0]["prompt"] for r in stored(world)["world_coherence"].values()} == {version("world_coherence")}
    assert ": 0 episodes" in world.score()

    world.llm.config["judges"]["j"]["thinking_about_prose"] = "qwen3-32b-mini"  # another judge under the same profile name
    world.llm.replies["qwen3-32b-mini"] = [judge] * 20
    out = world.score()
    assert "2 episodes" in out and "rubrics thinking_about_prose:" in out  # `quiet` asked no judge, so its rows stand
    # --all scores everything listed again, whatever stands.
    assert "6 episodes" in world.score("--all")


def test_the_closing_line_counts_a_row_in_which_nothing_was_scored_apart(world):
    out = world.score()
    # plain: counts (2 rows); judged: 2 rubrics x 2; quiet: 2 rows holding only `no_thinking`.
    assert "6 episodes, 8 rows: 6 scored, 2 with nothing scored (no_thinking), 0 scorer errors" in out


def test_a_second_pass_on_one_world_and_profile_is_refused_while_the_first_runs(world, monkeypatch):
    """Two at once would each pay the judge for the same rollouts: the first has looked, the
    second scores everything, and the first then scores what it saw."""
    from writing_assistant.episode.segments import Store

    judged, second = Store.judged, []

    def another_pass_starts(self, profile):
        found = judged(self, profile)
        if not second:
            second.append(None)

            def run():
                try:
                    second.append(cli.main(["--models", str(MODELS_TOML), "score", "--world", "sel", "--judges", "j"]))
                except SystemExit as e:
                    second.append(str(e))

            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
        return found

    monkeypatch.setattr(Store, "judged", another_pass_starts)
    world.score()
    assert "another judging pass" in str(second[1]) and "sel" in second[1] and "'j'" in second[1]
    with world.open("sel") as home:
        assert {n for (n,) in home.db.execute("SELECT COUNT(*) FROM scores GROUP BY ep_id, profile")} == {1}
    # Another profile is another judge's money, and a pass that has ended holds nothing.
    world.llm.config["judges"]["k"] = world.llm.config["judges"]["j"]
    second.clear()
    assert cli.main(["--models", str(MODELS_TOML), "score", "--world", "sel", "--judges", "k", "--rubric", "counts"]) == 0
    assert ": 0 episodes" in world.score()
