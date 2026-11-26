import copy
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
    before = copy.deepcopy(parent)
    child = parent.fork("twin")
    await dispatch_tool(child, "w", "hurt", {"who": "orc"}, HURT)
    await dispatch_tool(child, "w", "episode_write", {"path": "ch0.md", "text": "new", "rev": 1}, HURT)
    child.deliver("w", "go on")
    child.calls.append(CallRecord(actor="w", turn_idx=0))
    child.actor_tools["w"].append("episode_grep")
    child.actors["x"] = "m"
    child.meta["k"] = 1
    assert child.state["world"]["hp"]["orc"] == 6 and child.files["ch0.md"] == "new"
    assert parent == before
    assert (child.condition, child.ep_id, child.baseline_of) == ("twin", "e~twin", None)
    assert child.parent == (parent.seg, len(parent.events), len(parent.calls)) and child.seg != parent.seg
    assert [(r.ep_id, r.rollout_of, r.condition) for r in (parent.rollout(3, 0), parent.rollout(3, 1))] == [
        ("e@3.0", ("e", 3), "base"), ("e@3.1", ("e", 3), "base")]
    twin = parent.fork("twin", baseline=True)
    assert twin.baseline_of == "e"
    again = twin.rollout(0, 0)  # a twin's rollout is the baseline of nothing
    assert (again.ep_id, again.rollout_of, again.condition, again.baseline_of) == ("e~twin@0.0", ("e~twin", 0), "twin", None)


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
        ep = _ep(actors={"w": "m"}, actor_tools={"w": ["episode_read", "episode_grep"]})
        for t in range(2):
            idx = ep.deliver("w", f"turn {t}")
            ep.calls += calls_for(idx)
        ep.complete()
        return ep.status, ep.failure_reason

    full = ["episode_read", "episode_grep"]
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
