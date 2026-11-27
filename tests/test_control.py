"""The controller (`run/control.py`) and `wa-bench run --world`: groups handed over end up
played in the library, by play alone."""

import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from writing_assistant import cli
from writing_assistant.episode import Episode
from writing_assistant.episode import store as episode_store
from writing_assistant.episode.library import Library
from writing_assistant.run import directory, situate
from writing_assistant.run.control import Settings, Unplayed, Work, play_groups
from writing_assistant.run.directory import Refused, services_for

MODELS_TOML = Path(__file__).resolve().parents[1] / "models.toml"

TALK = {"request": "Plan the crossing.", "max_turns": 4, "script": [{"say": "And then?"}] * 4,
        "seats": {"author": {"cast": "scripted"}, "writer": {"cast": "subject", "brief": "You are the writer."}}}


class NoJudge:
    """Stands in for every judge: any use of it is a failure of the boundary."""

    def __getattr__(self, name):
        raise AssertionError(f"a judge was reached while playing ({name})")


@pytest.fixture
def world(tmp_path, monkeypatch, fake_llm):
    """A library with the world `sel` (and `held`, of the test split), two situations in
    each, and `world.settings`; `world.llm.replies` scripts the subject."""
    monkeypatch.setenv("WA_LIBRARY", str(tmp_path / "library"))
    monkeypatch.setattr(episode_store, "TEST_USES", tmp_path / "test-split-uses.jsonl")
    monkeypatch.setattr(directory, "TEST_USES", tmp_path / "test-split-uses.jsonl")
    llm = fake_llm({"gemma-4-26b": ["We sail at dawn."] * 200})
    monkeypatch.setattr(directory, "LLMClient", lambda models: llm)
    monkeypatch.setattr(directory, "judges_from", lambda client, profile: NoJudge())
    lib = Library()
    for name, split in (("sel", "dev"), ("held", "test")):
        lib.create(Episode(ep_id=name, task_id=name, suite="root", split=split, files={"kb.md": "The harbour town of Sel."},
                           file_revs={"kb.md": 1}))
        with lib.writing(name) as store:
            for n in (1, 2):
                situate(store, services_for(MODELS_TOML, "none"), "situations", "converse", TALK, name, name=f"{name}/s{n}")
    lib.llm, lib.uses = llm, tmp_path / "test-split-uses.jsonl"
    lib.settings = Settings(endpoint="gemma-4-26b", cast=None, max_tokens=256, concurrency=4, models=MODELS_TOML,
                            runs=tmp_path / "runs")
    return lib


def members(lib, world: str, group: str) -> dict[str, str]:
    """name -> status of the group's closed members in the library."""
    with lib.open(world) as home:
        return {ep.ep_id: ep.status for ep in home.episodes() if ep.group == group}


async def test_groups_are_played_into_the_world_reaching_no_judge_and_no_score(world, monkeypatch):
    statements = []
    connect = sqlite3.connect

    def traced(*a, **kw):
        db = connect(*a, **kw)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(sqlite3, "connect", traced)
    work = [Work("sel", "sel/s1", 3), Work("sel", "sel/s2", 2), Work("sel", "sel/s1", 2, round="1")]
    groups = await play_groups(world, work, world.settings)
    assert len(set(groups.values())) == 3
    for w in work:
        assert list(members(world, "sel", groups[w]).values()) == ["completed"] * w.rollouts
    writes = [s for s in statements if "scores" in s.lower() and not s.lstrip().upper().startswith(("CREATE", "PRAGMA"))]
    assert statements and not writes
    with world.open("sel") as home:
        assert home.judgings() == [] and home.names() and set(home.situations()) == {"sel/s1", "sel/s2"}
    # Asked for again, the same groups come back and nothing is played.
    calls = len(world.llm.requests)
    assert await play_groups(world, work, world.settings) == groups and len(world.llm.requests) == calls


async def test_the_run_log_has_a_line_per_closed_rollout_and_is_read_after_a_write_cut_short(world, capsys):
    """What `status` and a person tracing a failure read: every rollout that closed is a
    line naming it, and a writer killed mid-line costs that line only."""
    groups = await play_groups(world, [Work("sel", "sel/s1", 3)], world.settings)
    [run_dir] = world.settings.runs.iterdir()
    for path in (run_dir / "log.jsonl", world.log_path("sel")):
        events = episode_store.read_jsonl(path)
        assert [e["event"] for e in events][0] == "started" and events[-1]["event"] == "finished"
        assert {e["ep_id"] for e in events if e["event"] == "closed"} == set(members(world, "sel", *groups.values()))
    with (run_dir / "log.jsonl").open("a") as f:
        f.write('{"at": "2026-11-27T00:00:00.000Z", "pid": 1, "event": "ope')  # killed here
    assert len(episode_store.read_jsonl(run_dir / "log.jsonl")) == len(events)
    assert cli.main(["--models", str(MODELS_TOML), "status", "--world", "sel"]) == 0
    assert "3 closed, 3 pushed" in capsys.readouterr().out
    await play_groups(world, [Work("sel", "sel/s1", 3)], world.settings)  # the next writer starts on a new line
    assert [e["event"] for e in episode_store.read_jsonl(run_dir / "log.jsonl")][len(events):] == ["started", "finished"]


async def test_a_member_that_raises_once_is_played_by_the_second_pass(world):
    world.llm.replies["gemma-4-26b"][1:1] = [lambda *a: 1 / 0]
    [(w, group)] = (await play_groups(world, [Work("sel", "sel/s1", 3)], world.settings)).items()
    assert list(members(world, "sel", group).values()) == ["completed"] * 3


async def test_a_member_that_raises_twice_is_reported_with_its_group_and_no_round_is_made_up(world):
    def down(messages, tools, endpoint):
        raise ConnectionError("endpoint down")

    world.llm.replies["gemma-4-26b"] = ["We sail at dawn."] * 2 + [down] * 20  # one member plays; the others never can
    with pytest.raises(Unplayed) as e:
        await play_groups(world, [Work("sel", "sel/s1", 3)], world.settings)
    [(w, group)] = e.value.groups.items()
    played = members(world, "sel", group)
    assert len(played) == 1 and group in str(e.value)
    with world.open("sel") as home:
        short = set(home.rollouts("sel/s1")) | {f"sel/s1@0.{group}.{i}" for i in range(3)}
        assert home.groups() == {group: {"size": 3, "sources": {"sel/s1": 1}, "played": False}}  # one group, one round
    assert all(name in str(e.value) for name in short - played.keys())
    # The endpoint back, the same work finishes the same group and replays nothing closed.
    world.llm.replies["gemma-4-26b"] = ["We sail at dawn."] * 20
    assert await play_groups(world, [w], world.settings) == {w: group}
    assert members(world, "sel", group).keys() >= played.keys() and len(members(world, "sel", group)) == 3


def test_a_test_world_is_played_only_on_the_second_flag_and_each_use_is_logged(world, capsys):
    with pytest.raises(Refused, match="held-out"):
        asyncio.run(play_groups(world, [Work("held", "held/s1", 1)], world.settings))
    with pytest.raises(SystemExit, match="held-out"):
        cli.main(["--models", str(MODELS_TOML), "run", "--world", "held", "--endpoint", "gemma-4-26b",
                  "--out", str(world.settings.runs)])
    assert not world.uses.exists()
    with world.open("held") as home:
        assert home.names() == []
    asyncio.run(play_groups(world, [Work("held", "held/s1", 1)], replace(world.settings, held_out_test=True)))
    assert cli.main(["--models", str(MODELS_TOML), "run", "--world", "held", "--endpoint", "gemma-4-26b", "--held-out-test",
                     "--rollouts", "2", "--round", "b", "--out", str(world.settings.runs)]) == 0
    uses = [json.loads(line) for line in world.uses.read_text().splitlines()]
    assert [(u["world"], u["split"], u["work"]) for u in uses] == [("held", "test", 1), ("held", "test", 2)]
    with world.open("held") as home:
        assert sorted(g["size"] for g in home.groups().values() if g["played"]) == [1, 2, 2]
