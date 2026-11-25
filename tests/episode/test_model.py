import time

import pytest

from writing_assistant.episode import CallRecord, Episode
from writing_assistant.episode.tools import dispatch_tool

WS = ["episode_read", "episode_write", "episode_edit", "episode_grep", "episode_list"]


def _ep(**kw) -> Episode:
    return Episode(ep_id="e", task_id="t", suite="1a", split="dev", **kw)


def test_view_masks_events_and_hides_acl_files():
    ep = _ep(files={"notes.md": "open", "secrets.json": "the key is red", "brief/a.md": "author only"},
             file_revs={"notes.md": 1, "secrets.json": 1, "brief/a.md": 1},
             state={"file_acl": {"secrets.json": ["dm"], "brief/": ["author"]}})
    ep.log("dm", "say", {"text": "hello all"}, {"*": "full"})
    ep.log("dm", "world", {"text": "the key is red", "redacted": "the DM rolls"}, {"dm": "full", "pc": "redacted"})
    ep.log("dm", "world", {"text": "trap armed"}, {"dm": "full", "pc": "existence"})
    ep.log("dm", "decision", {"text": "the duke lied"}, {"dm": "full"})
    ep.log("dm", "say", {"text": "aside"}, {"*": "full", "pc": "existence"})  # seat entry beats "*"

    v = ep.view("pc")
    assert [(e.idx, e.actor, e.kind, e.data) for e in v.events] == [
        (0, "dm", "say", {"text": "hello all"}),
        (1, "dm", "world", {"redacted": "the DM rolls"}),
        (2, "", "world", {}),
        (3, "", "say", {}),
    ]
    assert all(set(e.visible_to) == {"pc"} for e in v.events)  # who else perceives is not shown
    assert set(v.files) == {"notes.md"} and set(v.file_revs) == {"notes.md"}
    assert set(ep.view("dm").files) == {"notes.md", "secrets.json"}
    assert set(ep.view("author").files) == {"notes.md", "brief/a.md"}
    for alias in ("./secrets.json", "x/../secrets.json", "secrets.json/."):
        assert not ep.readable("pc", alias)


async def test_acl_hides_files_from_every_workspace_tool():
    ep = _ep(files={"notes.md": "the key", "secrets.json": "the key is red"},
             file_revs={"notes.md": 1, "secrets.json": 1},
             state={"file_acl": {"secrets.json": ["dm"]}}, actor_tools={"pc": WS})
    missing = await dispatch_tool(ep, "pc", "episode_read", {"path": "secrets.json"}, {})
    nofile = await dispatch_tool(ep, "pc", "episode_read", {"path": "nope.json"}, {})
    assert missing == nofile | {"error": "no such file: secrets.json"}  # hidden reads as absent
    assert (await dispatch_tool(ep, "pc", "episode_read", {"path": "./secrets.json"}, {}))["ok"] is False
    grep = await dispatch_tool(ep, "pc", "episode_grep", {"pattern": "key"}, {})
    assert [m["path"] for m in grep["matches"]] == ["notes.md"]
    listing = await dispatch_tool(ep, "pc", "episode_list", {}, {})
    assert [f["path"] for f in listing["files"]] == ["notes.md"]
    for tool, args in [("episode_write", {"path": "secrets.json", "text": "x", "rev": 1}),
                       ("episode_edit", {"path": "secrets.json", "old": "red", "new": "blue", "rev": 1})]:
        assert (await dispatch_tool(ep, "pc", tool, args, {}))["ok"] is False
    assert ep.files["secrets.json"] == "the key is red"


def _big_episode(n: int = 5000) -> Episode:
    ep = _ep(state={"world": {"hp": {"pc": 10, "orc": 7}, "seed": 3}, "file_acl": {}},
             files={f"ch{i}.md": "text " * 200 for i in range(20)}, file_revs={f"ch{i}.md": 1 for i in range(20)},
             actors={"w": "m"}, actor_tools={"w": WS + ["hurt"]})
    for i in range(n):
        ep.log("w", "say", {"text": f"line {i}"}, {"*": "full"})
    ep.calls = [CallRecord(actor="w", turn_idx=i) for i in range(200)]
    return ep


def test_fork_is_fast():
    ep = _big_episode()
    times = []
    for _ in range(200):
        t = time.perf_counter()
        ep.fork("twin")
        times.append(time.perf_counter() - t)
    assert sorted(times)[len(times) // 2] <= 1e-4


def _hurt(ep, actor, args):
    ep.state["world"]["hp"][args["who"]] -= 1
    return {"ok": True}


HURT = {"hurt": ({"name": "hurt", "description": "Lose 1 HP.", "parameters": {
    "type": "object", "properties": {"who": {"type": "string"}}, "required": ["who"],
    "additionalProperties": False}}, _hurt)}


async def test_fork_tool_calls_never_reach_parent():
    parent = _big_episode(10)
    before = parent.to_dict()
    child = parent.fork("twin")
    await dispatch_tool(child, "w", "hurt", {"who": "orc"}, HURT)
    await dispatch_tool(child, "w", "episode_write", {"path": "ch0.md", "text": "new", "rev": 1}, HURT)
    child.deliver("w", "go on")
    child.calls.append(CallRecord(actor="w", turn_idx=0))
    child.actor_tools["w"].append("episode_submit")
    child.actors["x"] = "m"
    child.meta["k"] = 1
    assert child.state["world"]["hp"]["orc"] == 6 and child.files["ch0.md"] == "new"
    assert parent.to_dict() == before
    assert (child.parent_id, child.condition, child.ep_id) == ("e", "twin", "e~twin")


def test_fork_swap_replaces_one_leaf_and_refuses_perceived_events():
    ep = _ep(state={"secret": {"fact": "red", "holder": "duke"}, "other": 1})
    ep.log("dm", "world", {"fact": "red"}, {"dm": "full"})
    ep.log("dm", "say", {"text": "public"}, {"*": "full"})
    ep.log("dm", "world", {"fact": "seen"}, {"dm": "full", "pc": "redacted"})
    twin = ep.fork("twin_swap", swap={"event_idx": 0, "data": {"fact": "blue"},
                                      "state": {"secret": {"fact": "blue"}}, "restricted": ["pc"]})
    assert twin.events[0].data == {"fact": "blue"} and ep.events[0].data == {"fact": "red"}
    assert twin.state == {"secret": {"fact": "blue", "holder": "duke"}, "other": 1}
    assert ep.state["secret"]["fact"] == "red"
    for i in (1, 2):
        with pytest.raises(ValueError, match="perceived"):
            ep.fork("bad", swap={"event_idx": i, "data": {}, "restricted": ["pc"]})


def test_complete_requires_an_attesting_call_per_delivery():
    def run(calls_for):
        ep = _ep(actors={"w": "m"}, actor_tools={"w": ["episode_read", "episode_submit"]})
        for t in range(2):
            idx = ep.deliver("w", f"turn {t}")
            ep.calls += calls_for(idx)
        ep.complete()
        return ep.status, ep.failure_reason

    full = ["episode_read", "episode_submit"]
    assert run(lambda t: [CallRecord(actor="w", turn_idx=t, tools_advertised=[]),
                          CallRecord(actor="w", turn_idx=t, tools_advertised=full + ["x"])]) == ("completed", None)
    assert run(lambda t: [] if t == 1 else [CallRecord(actor="w", turn_idx=t, tools_advertised=full)]) == ("failed", "mcp_unattested")
    assert run(lambda t: [CallRecord(actor="w", turn_idx=t, tools_advertised=["episode_read"])]) == ("failed", "mcp_unattested")
    assert run(lambda t: [CallRecord(actor="other", turn_idx=t, tools_advertised=full)]) == ("failed", "mcp_unattested")
    assert run(lambda t: [CallRecord(actor="w", turn_idx=t + 1, tools_advertised=full)]) == ("failed", "mcp_unattested")

    ep = _ep()
    ep.fail("driver_crashed")
    ep.complete()
    assert (ep.status, ep.failure_reason) == ("failed", "driver_crashed")


def test_a_saved_episode_loads_back_whole(tmp_path):
    """Re-scoring reads only the saved episode: a field left out of the save, or one that
    loads back as another type, silently changes what a re-score sees."""
    from dataclasses import MISSING, fields

    from writing_assistant.episode.store import load_episodes, save_episodes

    ep = Episode(
        ep_id="t~pov_view", task_id="t", suite="1a", split="test", parent_id="t", condition="pov_view",
        meta={"driver": "direct", "row": {"k": 3, "mode": "instruction"}},
        files={"a.md": "Zoë «x»"}, file_revs={"a.md": 2},
        state={"world": {"secrets": {"s": {"text": "x", "known_by": ["npc"]}}}, "pov": {"arm": "pov_view"}},
        actors={"writer": "gemma-4-26b", "author": "scripted"}, actor_tools={"writer": ["episode_read"]},
        status="truncated", failure_reason="writer was cut off at max_tokens=10",
    )
    ep.log("dm", "world", {"text": "hidden", "rolls": [3, 17]}, {"dm": "full", "pc": "redacted"})
    ep.deliver("writer", "write it")
    ep.calls.append(CallRecord(
        actor="writer", turn_idx=0, endpoint="gemma-4-26b", messages=[{"role": "user", "content": "write it"}],
        response="It was dark.", tool_calls=[{"id": "c1", "name": "episode_read", "arguments": {"path": "a.md"}}],
        tools_advertised=["episode_read"], usage={"input_tokens": 9, "output_tokens": 4, "cost_usd": 1.5e-05},
        ext={"logprobs": [-0.1]}, finish_reason="length", reasoning="Thinking about the dark.",
    ))
    # Every field carries a non-default value, so a field added later must be added here too.
    def defaults(obj):
        return [f.name for f in fields(obj)
                if getattr(obj, f.name) == (f.default_factory() if f.default_factory is not MISSING else f.default)]

    assert defaults(ep) == [] and defaults(ep.calls[0]) == []

    save_episodes([ep], tmp_path)
    [back] = load_episodes(tmp_path)
    assert back == ep and back.to_dict() == ep.to_dict()
    assert back.events[0].data["rolls"] == [3, 17] and back.view("pc").events[0].data == {}
