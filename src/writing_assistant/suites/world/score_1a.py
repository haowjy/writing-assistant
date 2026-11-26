"""Suite 1a scorers: pure over a frozen episode, reaching models only through `services`.

A metric that does not apply to a task is absent, never 0.0. Whether a text uses a fact is
asked of the `fact` judge with the text in view; an item the judge leaves unscored
(`judges.Unscored`) is left out and counted per metric by `Tally`, and a rate with no judged
item is absent. A `net_*` metric needs the twin's value too and is absent without it.
"""

from __future__ import annotations

import asyncio

from writing_assistant.episode import Episode
from writing_assistant.suites import Tally, last_reply, mean, norm, paragraphs, prose_metrics
from writing_assistant.suites.world import srd
from writing_assistant.suites.world.tasks_1a import AUTHOR, DM, WRITER, render_log

CONTRADICTION_Q = "Does the narration state anything that contradicts the recorded world events?"

# metric -> judge use (Suite.judged_by): everything the fact judge answers.
JUDGED_BY = dict.fromkeys(
    ["residue_rate", "post_pivot_residue", "pov_leak", "net_pov_leak", "permitted_fact_recall",
     "canon_snapback", "bent_rule_compliance", "canon_rule_compliance", "bent_rule_intrusion",
     "net_canon_snapback", "contradictions_per_1k"],
    "fact",
)


class _Judged:
    """Rates from the `fact` judge; unscored items are counted in `tally` under the metric."""

    def __init__(self, services) -> None:
        self.fact = services.judges.fact
        self.tally = Tally()

    async def ask(self, metric: str, question: str, evidence: str) -> bool | None:
        return await self.tally.ask(metric, self.fact(question, evidence))

    async def rate(self, metric: str, questions: list[str], text: str) -> float | None:
        return mean(await asyncio.gather(*(self.ask(metric, q, text) for q in questions)))


def _put(out: dict[str, float], name: str, value: float | None) -> None:
    if value is not None:
        out[name] = float(value)


# -- per task ------------------------------------------------------------------------------

async def _pivot(ep: Episode, judge: _Judged) -> tuple[dict, float | None, str]:
    s = ep.state["pivot"]
    scene = ep.files.get(s["scene_path"], "")
    out: dict[str, float] = {}
    residue = await judge.rate("residue_rate", s["residue_probes"], scene)
    _put(out, "residue_rate", residue)
    return out, residue, scene


async def _pov(ep: Episode, judge: _Judged) -> tuple[dict, float | None, str]:
    s = ep.state["pov"]
    chapter = last_reply(ep, WRITER)
    out: dict[str, float] = {}
    leak = await judge.rate("pov_leak", s["hidden_probes"], chapter)
    _put(out, "pov_leak", leak)
    _put(out, "permitted_fact_recall", await judge.rate("permitted_fact_recall", s["permitted_probes"], chapter))
    return out, leak, chapter


async def _rule_bend(ep: Episode, judge: _Judged) -> tuple[dict, float | None, str]:
    s = ep.state["rule_bend"]
    chapter = ep.files.get(s["out_path"], "")
    bent_arm = ep.condition == "bent"
    canon = await judge.rate("canon_snapback" if bent_arm else "canon_rule_compliance", s["canon_probes"], chapter)
    bent = await judge.rate("bent_rule_compliance" if bent_arm else "bent_rule_intrusion", s["bent_probes"], chapter)
    out: dict[str, float] = {}
    if bent_arm:
        _put(out, "canon_snapback", canon)
        _put(out, "bent_rule_compliance", bent)
        wrong = canon
    else:  # the twin keeps canon: following the bent rule is the wrong-rule rate here
        _put(out, "canon_rule_compliance", canon)
        _put(out, "bent_rule_intrusion", bent)
        wrong = bent
    return out, wrong, chapter


async def _dm(ep: Episode, judge: _Judged) -> tuple[dict, None, str]:
    world = ep.state["world"]
    out: dict[str, float] = {}
    calls = [e for e in ep.events if e.kind == "tool" and e.actor == DM and e.data["name"] in srd.TOOLS]
    out["srd_calls"] = float(len(calls))
    accepted = sum(1 for e in calls if e.data["result"]["ok"])
    if accepted + world["rules_rejections"]:
        out["srd_rejection_rate"] = world["rules_rejections"] / (accepted + world["rules_rejections"])
    placed = [e for e in calls if e.data["name"] in ("srd_propose_event", "srd_reveal")
              and (e.data["result"]["ok"] or "rule" in e.data["result"])]
    if placed:
        out["visibility_error_rate"] = sum(e.data["result"].get("rule") == "visibility" for e in placed) / len(placed)

    # Narration is checked against the world events recorded before it.
    contradictions, words, record = 0, 0, []
    for ev in ep.events:
        if ev.kind == "world":
            record.append(ev)
        elif ev.kind == "say" and ev.actor == DM and ev.data["text"].strip():
            narration = ev.data["text"]
            evidence = f"Recorded world events:\n{render_log(record, 'the party') or '(none)'}\n\nNarration:\n{narration}"
            answer = await judge.ask("contradictions_per_1k", CONTRADICTION_Q, evidence)
            if answer is not None:
                contradictions += answer
                words += len(narration.split())
    if words:
        out["contradictions_per_1k"] = 1000 * contradictions / words
    return out, None, "\n\n".join(e.data["text"] for e in ep.events if e.kind == "say" and e.actor == DM)


def _collab(ep: Episode) -> dict[str, float]:
    """Paragraph diff over each writer turn: from its delivery to the author's next edit or
    delivery. Paragraphs are compared with whitespace collapsed and by containment in the
    whole document, so reflowing, splitting or merging paragraphs changes nothing."""
    path = ep.state["collab"]["path"]
    rounds = ep.state["script"]["turns"]
    doc = ep.state["collab"]["initial"]
    human: set[str] = set()
    clobbered: set[str] = set()
    out_of_scope = accepted = 0
    open_turn: tuple[str, list[str]] | None = None
    n = 0

    def close() -> None:
        nonlocal out_of_scope, accepted, open_turn
        if open_turn is None:
            return
        before, scope = open_turn
        after = norm(doc)
        scope = [norm(p) for p in scope]
        in_scope, other = [], []
        for par in map(norm, paragraphs(before)):
            (in_scope if any(s in par for s in scope) else other).append(par)
        changed = [p for p in other if p not in after]
        out_of_scope += len(changed)
        clobbered.update(p for p in changed if p in human)
        accepted += any(p not in after for p in in_scope) and not changed
        open_turn = None

    for ev in ep.events:
        if ev.kind == "deliver" and ev.actor == WRITER:
            close()
            open_turn = (doc, rounds[n]["scope"])
            n += 1
        elif ev.kind == "file_write" and ev.data["path"] == path:
            if ev.actor == AUTHOR:
                close()
                old = {norm(p) for p in paragraphs(doc)}
                human.update(p for p in map(norm, paragraphs(ev.data["after"])) if p not in old)
            doc = ev.data["after"]
    close()

    out = {"clobbered_human_paragraphs": float(len(clobbered)), "out_of_scope_edits": float(out_of_scope),
           "accepted_edits": float(accepted)}
    usage = [c.usage.get("output_tokens") for c in ep.calls if c.actor == WRITER]
    if accepted and usage and None not in usage:
        out["tokens_per_accepted_edit"] = sum(usage) / accepted
    return out


SCORERS = {"pivot_unbinding": _pivot, "pov_write": _pov, "rule_bend": _rule_bend, "dm_play": _dm}
NET = {"pivot_unbinding": "post_pivot_residue", "pov_write": "net_pov_leak", "rule_bend": "net_canon_snapback"}


async def score(ep: Episode, services, twin: Episode | None = None) -> dict[str, float]:
    """Metrics for one episode. With its twin (`tasks_1a.TWINS`), also the net metric: the
    episode's rate minus the twin's baseline (for rule_bend, canon use under the bent rule
    minus bent-rule use under kept canon)."""
    if not _finished(ep):
        raise ValueError(f"{ep.ep_id} is {ep.status}; only finished episodes are scored")
    kind = ep.state["kind"]
    if kind == "collab_edit":
        return _collab(ep)
    judge = _Judged(services)
    out, own, text = await SCORERS[kind](ep, judge)
    judge.tally.into(out)
    prose_metrics(out, text, ep, services)
    if twin is not None and _finished(twin) and kind in NET:
        _, base, _ = await SCORERS[kind](twin, _Judged(services))
        if own is not None and base is not None:
            out[NET[kind]] = own - base
    return out


def _finished(ep: Episode) -> bool:
    """Completed or truncated; a failed episode is never scored."""
    return ep.status in ("completed", "truncated")
