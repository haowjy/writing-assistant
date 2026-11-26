from dataclasses import MISSING, fields

import pytest

from writing_assistant.episode import CallRecord, Episode
from writing_assistant.episode.segments import Store


def test_a_saved_episode_loads_back_whole(tmp_path):
    """Re-scoring reads only the stored episode: a field left out of the save, or one that
    loads back as another type, silently changes what a re-score sees."""
    arm = Episode(ep_id="t~pov_view", task_id="t", suite="1a", split="test", files={"a.md": "one"}, file_revs={"a.md": 1})
    arm.log("dm", "world", {"text": "hidden", "rolls": [3, 17]}, {"dm": "full", "pc": "redacted"})
    ep = arm.fork("twin_swap", baseline=True)
    ep.meta = {"driver": "direct", "row": {"k": 3, "mode": "instruction"}}
    ep.rollout_of = ("t~pov_view", 2)
    ep.files["a.md"], ep.file_revs["a.md"] = "Zoë «x»", 2
    ep.state = {"world": {"secrets": {"s": {"text": "x", "known_by": ["npc"]}}}, "pov": {"arm": "pov_view"}}
    ep.actors = {"writer": "gemma-4-26b", "author": "scripted"}
    ep.actor_tools = {"writer": ["episode_read"]}
    ep.status, ep.failure_reason = "truncated", "writer was cut off at max_tokens=10"
    ep.deliver("writer", "write it")
    ep.calls.append(CallRecord(
        actor="writer", turn_idx=0, at_event=2, endpoint="gemma-4-26b", messages=[{"role": "user", "content": "write it"}],
        response="It was dark.", tool_calls=[{"id": "c1", "name": "episode_read", "arguments": {"path": "a.md"}}],
        tools_advertised=["episode_read"], usage={"input_tokens": 9, "output_tokens": 4, "cost_usd": 1.5e-05},
        ext={"logprobs": [-0.1]}, finish_reason="length", reasoning="Thinking about the dark.",
    ))
    # Every field carries a non-default value, so a field added later must be added here too.
    def defaults(obj):
        return [f.name for f in fields(obj)
                if f.default_factory is MISSING and getattr(obj, f.name) == f.default
                or f.default_factory is not MISSING and f.name != "seg" and getattr(obj, f.name) == f.default_factory()]

    assert defaults(ep) == [] and defaults(ep.calls[0]) == []

    with Store(tmp_path / "store.db") as store:
        store.close(ep)
    with Store(tmp_path / "store.db", readonly=True) as store:
        [back] = store.episodes()
    assert back == ep
    assert back.events[0].data["rolls"] == [3, 17] and back.view("pc").events[0].data == {}


def test_a_fork_made_past_what_its_parent_has_stored_is_refused(tmp_path):
    """A fork is stored after a position in its parent's log. Past the parent's last
    checkpoint nothing is stored there yet, so the fork would load with whatever the parent
    put there later: a resumed parent that played on differently would rewrite its child."""
    def say(ep, text):
        ep.log("dm", "say", {"text": text}, {"*": "full"})

    with Store(tmp_path / "store.db") as store:
        parent = Episode(ep_id="p", task_id="t", suite="1a", split="dev")
        say(parent, "stored")
        store.append(parent)
        say(parent, "in flight")
        ahead = parent.fork("ahead")
        with pytest.raises(ValueError, match="past the 1 events and 0 calls stored"):
            store.append(ahead)
        assert store.names(closed=False) == ["p"]
        store.append(parent)
        store.append(ahead)
        assert store.load("p~ahead") == ahead
