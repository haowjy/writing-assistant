import pytest

from writing_assistant.episode import Episode
from writing_assistant.episode.tools import dispatch_tool

TOOLS = ["episode_read", "episode_write", "episode_edit", "episode_grep"]


@pytest.mark.parametrize(("name", "args", "error"), [
    ("episode_list", {}, "not available"),
    ("episode_roll", {"expr": "1d6"}, "not available"),
    ("episode_read", {"path": "../etc/passwd"}, "bad path"),
    ("episode_read", {"path": "/etc/passwd"}, "bad path"),
    ("episode_read", {}, "missing required 'path'"),
    ("episode_read", {"path": 3}, "'path' must be of type string"),
    ("episode_read", {"path": "a.md", "mode": "r"}, "unknown key 'mode'"),
    ("episode_read", "a.md", "must be of type object"),
    ("episode_write", {"path": "a.md", "text": "x", "rev": 1}, "stale revision"),
    ("episode_write", {"path": "new.md", "text": "x", "rev": 2}, "stale revision"),
    ("episode_write", {"path": "a.md", "text": "x", "rev": True}, "'rev' must be of type integer"),
    ("episode_write", {"path": "a.md", "text": "x", "rev": -1}, "'rev' must be >= 0"),
    ("episode_edit", {"path": "a.md", "old": "one", "new": "1", "rev": 1}, "stale revision"),
    ("episode_edit", {"path": "a.md", "old": "zzz", "new": "1", "rev": 2}, "occurs 0 times"),
    ("episode_edit", {"path": "a.md", "old": "o", "new": "1", "rev": 2}, "occurs 2 times"),
    ("episode_edit", {"path": "a.md", "old": "", "new": "1", "rev": 2}, "at least 1 characters"),
    ("episode_grep", {"pattern": "("}, "bad pattern"),
])
async def test_refusals_return_errors_and_log_the_call(name, args, error):
    ep = Episode(ep_id="e", task_id="t", suite="1a", split="dev", files={"a.md": "one two"},
                 file_revs={"a.md": 2}, actor_tools={"w": TOOLS})
    result = await dispatch_tool(ep, "w", name, args, {})
    assert result["ok"] is False and error in result["error"]
    assert ep.files == {"a.md": "one two"} and ep.file_revs == {"a.md": 2}
    [ev] = ep.events
    assert (ev.kind, ev.actor, ev.data["name"], ev.data["result"], ev.visible_to) == ("tool", "w", name, result, {"w": "full"})
