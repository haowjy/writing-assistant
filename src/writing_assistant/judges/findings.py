"""Judges that read a stored rollout and answer with findings.

One judge use per prompt file in `judges/rubrics/` (the file's name is the use, and a hash
of its text the prompt version, so a wording change can never keep an old version's
calibration). The prompt is the system message; the next message is the rollout as one
seat's judge is given it (`render`, over `episode.rollout_evidence`), whole or up to one
target turn. The reply is a JSON object with a `findings` array:

    covers      "task", or {"turn": turn_idx}
    seat        the seat it is about (optional)
    dimension   "output" | "thinking"
    criterion   the rubric's name for what was scored
    value       0 | 1 | 2, or absent with `unscored` saying why
    cites       [{"turn": turn_idx, "note": str}]
    evidence    [{"path": str, "revision": int}]

`findings` returns them as plain dicts in that shape and nothing else of the reply, so a
stored finding holds references and no evidence text. A reply that is not that shape is
unscored (`Unscored`), never read leniently into a score.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from writing_assistant.judges.reply import MAX_TOKENS, TEMPERATURE, Unscored, answer_text, why_unanswered

if TYPE_CHECKING:
    from writing_assistant.episode import CallRecord, RolloutEvidence
    from writing_assistant.llm.client import LLMClient

PROMPTS = {path.stem: path.read_text(encoding="utf-8") for path in sorted((Path(__file__).parent / "rubrics").glob("*.md"))}


def version(use: str) -> str:
    return f"{use}-{hashlib.sha256(PROMPTS[use].encode()).hexdigest()[:8]}"


async def findings(client: LLMClient, endpoint: str, use: str, record: str) -> tuple[list[dict[str, Any]], list[CallRecord]]:
    """(the findings the judge returned, [the call]) for one rendered rollout."""
    rec = await client.chat([{"role": "system", "content": PROMPTS[use]}, {"role": "user", "content": record}],
                            endpoint=endpoint, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    m = re.search(r"\{.*\}", answer_text(rec), re.DOTALL)  # the object may sit inside a ```json fence
    try:
        found = json.loads(m.group(0))["findings"] if m else None
        return [_finding(f) for f in found], [rec]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise Unscored(why_unanswered(rec), [rec]) from None


def _finding(f: dict[str, Any]) -> dict[str, Any]:
    """One finding in the stored shape; raises on anything else."""
    covers = f["covers"]
    if covers != "task":
        covers = {"turn": int(covers["turn"])}
    if f["dimension"] not in ("output", "thinking") or not isinstance(f["criterion"], str):
        raise ValueError(f)
    out = {"covers": covers, "dimension": f["dimension"], "criterion": f["criterion"]}
    if isinstance(f.get("seat"), str):
        out["seat"] = f["seat"]
    if f.get("value") in (0, 1, 2) and not isinstance(f["value"], bool):
        out["value"] = int(f["value"])
    elif isinstance(f.get("unscored"), str) and f["unscored"]:
        out["unscored"] = f["unscored"]
    else:
        raise ValueError(f)
    out["cites"] = [{"turn": int(c["turn"]), "note": str(c.get("note", ""))} for c in f.get("cites") or []]
    out["evidence"] = [{"path": str(e["path"]), "revision": int(e["revision"])} for e in f.get("evidence") or []]
    return out


def render(ev: RolloutEvidence, turn: int | None = None) -> str:
    """The rollout as the judge reads it, for one seat: the task, the seat's brief, the files
    as the rollout began (each marked when the seat could not read it), then every delivery
    to the seat with what it heard, thought, called, replied and wrote. With `turn`, the
    record stops after that turn, which is named as the target, and the end files are left
    out; without, the files as the rollout ended close it. It is a function of the stored
    rollout alone, so the record a stored finding was made from can be built again."""
    # Named keys only: a field a later task kind adds to its payload reaches no judge by default.
    task = {k: ev.task[k] for k in ("task_id", "kind") if k in ev.task} | {
        k: ev.task["payload"][k] for k in ("request", "opening", "to", "max_turns") if k in ev.task.get("payload", {})}
    out = ["# Task", json.dumps(task, ensure_ascii=False, indent=1), f"# Seat under judgment: {ev.seat}"]
    if turn is not None:
        out.append(f"Target: turn {turn} of this seat. The record below ends with it.")
    out += ["## Its brief", ev.brief or "(none)", "# Files when the rollout began", *_files(ev, ev.files_at_start)]
    out.append("# Turns of this seat" + {"trace": " (thinking is the model's own trace)",
                                         "summary": " (thinking is the host's summary of the model's)"}.get(ev.thinking, ""))
    for t in ev.turns:
        if turn is not None and t.turn_idx > turn:
            break
        out.append(f"## Turn {t.turn_idx}" + (" (played before this rollout began: context, not to be scored)" if t.inherited else ""))
        out += ["### Heard", t.heard, "### Thinking", t.thinking or "(none recorded)"]
        calls = [c for call in t.calls for c in call.tool_calls]
        if calls:
            out.append("### Tool calls")
            for n, c in enumerate(calls):
                result = json.dumps(t.results[n], ensure_ascii=False) if n < len(t.results) else "(no result recorded)"
                out.append(f"- {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)}) -> {result}")
        out += ["### Reply", t.reply or "(nothing)"]
        for path, rev, before, after in t.changes:
            out.append(f"### Wrote {path}, revision {rev}")
            out.append(after if before is None else "".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True), f"{path} before", f"{path} revision {rev}")))
    if turn is None:
        out += [f"# How it ended: {ev.status}" + (f" ({ev.failure_reason})" if ev.failure_reason else ""),
                "# Files when the rollout ended", *_files(ev, ev.files_at_end, ev.files_at_start)]
    return "\n\n".join(out)


def _files(ev: RolloutEvidence, files: dict[str, tuple[str, int]], unless: dict[str, tuple[str, int]] | None = None) -> list[str]:
    out = []
    for path, (text, rev) in sorted(files.items()):
        closed = "" if path in ev.visible else "; this seat could not read it"
        if unless and unless.get(path) == (text, rev):
            out.append(f"## {path} (revision {rev}{closed}): unchanged")
        else:
            out += [f"## {path} (revision {rev}{closed})", text]
    return out or ["(none)"]
