"""Suite 1a scorers: pure over a frozen episode, reaching models only through `services`.

Outcome metrics read the one artefact their task kind names in `OUTPUT` (the rule is in
`suites/__init__.py`): the scene file, the chapter reply, the rule-bend chapter file, the
DM's narration over its recorded world events. Without it they are absent and
`n_no_output.<reason>` is written instead. `collab_edit` has one artefact per round, the
document after the writer's edits in that round (`_collab`): a round without one loses its
metrics and the other rounds keep theirs. The DM's tool bookkeeping reads the
log and is written whenever the episode ran.

A metric that does not apply to a task is absent, never 0.0. Whether a text uses a fact is
asked of the `fact` judge with the text in view; an item the judge leaves unscored
(`judges.Unscored`) is left out and counted per metric by `Tally`, and a rate with no judged
item is absent. The twin (`tasks_1a.TWINS`) is judged here, once, as the parent's baseline:
its rate is written as `twin_<metric>` and a `net_*` metric is the parent's rate minus it,
absent when either side has no artefact or no judged item. A twin's missing artefact is
counted as `n_twin_no_output.<reason>`; a twin that was never saved, as `missing`.
"""

from __future__ import annotations

import asyncio

from writing_assistant.episode import Episode
from writing_assistant.suites import NoOutput, Tally, file_output, mean, norm, paragraphs, prose_metrics, reply_output
from writing_assistant.suites.world import author, srd
from writing_assistant.suites.world.tasks_1a import AUTHOR, DM, POV_WORDS, WRITER, render_log

CONTRADICTION_Q = "Does the narration state anything that contradicts the recorded world events?"

# metric -> judge use (Suite.judged_by): everything the fact judge answers.
JUDGED_BY = dict.fromkeys(
    ["residue_rate", "twin_residue_rate", "post_pivot_residue", "pov_leak", "twin_pov_leak", "net_pov_leak",
     "permitted_fact_recall", "canon_snapback", "bent_rule_compliance", "twin_bent_rule_intrusion",
     "net_canon_snapback", "contradicted_narrations_per_1k_words"],
    "fact",
)
# metric -> what a value means (Suite.metric_about).
ABOUT = {
    "pov_leak": "Share of the secret's questions answered yes of the chapter: a yes is a leak.",
    "twin_pov_leak": "The same questions asked of the twin's chapter, whose secret was swapped: a yes is the judge "
                     "or chance.",
    "net_pov_leak": "pov_leak minus twin_pov_leak.",
    "permitted_fact_recall": "Share of questions about things the character did see answered yes, so a chapter "
                             "that leaves everything out does not score well.",
    "residue_rate": "Share of questions about the scrapped direction A answered yes of the scene.",
    "twin_residue_rate": "The same questions asked of the clean_b scene, where direction A was never given.",
    "post_pivot_residue": "residue_rate minus twin_residue_rate.",
    "canon_snapback": "Share of canon questions answered yes: the chapter followed the book over the rules file.",
    "bent_rule_compliance": "Share of changed-rule questions answered yes: the chapter followed the rules file.",
    "twin_bent_rule_intrusion": "The changed-rule questions asked of the twin's chapter, whose rules keep canon.",
    "net_canon_snapback": "canon_snapback minus twin_bent_rule_intrusion; one question per side, so -1, 0 or 1 per task.",
    "contradicted_narrations_per_1k_words": "Narrations the judge says contradict the world events recorded before "
                                            "them, per 1,000 words of judged narration.",
    "srd_calls": "Rules-tool calls the game master made.",
    "srd_rejection_rate": "Share of its rules-tool calls the rules engine rejected.",
    "visibility_error_rate": "Of its event and reveal calls, the share rejected for giving an event to a character "
                             "who could not perceive it.",
    "out_of_scope_edits": "Paragraphs outside a round's request that the model changed.",
    "clobbered_author_edits": "Of those, paragraphs the author had written during the episode.",
    "in_scope_rounds": "Rounds where a requested paragraph changed and nothing else did.",
    "tokens_per_in_scope_round": "Output tokens spent per in-scope round.",
}


async def _rate(tally: Tally, services, metric: str, questions: list[str], text: str,
                about: str | None = None) -> float | None:
    """Share of `questions` the fact judge answers yes about `text`; None if none was judged."""
    fact = services.judges.fact
    return mean(await asyncio.gather(*(tally.ask(metric, fact, q, text, about=about) for q in questions)))


def _put(out: dict[str, float], name: str, value: float | None) -> None:
    if value is not None:
        out[name] = float(value)


# -- the artefact each task kind's outcome metrics read --------------------------------------

def _script_done(ep: Episode) -> None:
    if author.remaining(ep.state):
        raise NoOutput("truncated")


def _written(ep: Episode, path: str) -> str:
    _script_done(ep)
    return file_output(ep, WRITER, path)


def _narration(ep: Episode) -> str:
    """The DM's narration, once the script has run to its end, the model DM's last reply
    ended normally, and the DM has recorded at least one world event to check it against."""
    _script_done(ep)
    if ep.actors[DM] != "scripted":
        reply_output(ep, DM)
    if not any(e.kind == "world" for e in ep.events):
        raise NoOutput("empty_output")
    return "\n\n".join(e.data["text"] for e in ep.events if e.kind == "say" and e.actor == DM)


OUTPUT = {
    "pivot_unbinding": lambda ep: _written(ep, ep.state["pivot"]["scene_path"]),
    "pov_write": lambda ep: reply_output(ep, WRITER, POV_WORDS),
    "rule_bend": lambda ep: _written(ep, ep.state["rule_bend"]["out_path"]),
    "dm_play": _narration,
}


# -- outcome metrics per task kind: (metrics, the rate a net metric subtracts from) ----------

async def _pivot(ep: Episode, scene: str, services, tally: Tally) -> tuple[dict, float | None]:
    residue = await _rate(tally, services, "residue_rate", ep.state["pivot"]["residue_probes"], scene)
    out: dict[str, float] = {}
    _put(out, "residue_rate", residue)
    return out, residue


async def _pov(ep: Episode, chapter: str, services, tally: Tally) -> tuple[dict, float | None]:
    s = ep.state["pov"]
    out: dict[str, float] = {}
    leak = await _rate(tally, services, "pov_leak", s["hidden_probes"], chapter)
    _put(out, "pov_leak", leak)
    _put(out, "permitted_fact_recall", await _rate(tally, services, "permitted_fact_recall", s["permitted_probes"], chapter))
    return out, leak


async def _rule_bend(ep: Episode, chapter: str, services, tally: Tally) -> tuple[dict, float | None]:
    """The bent arm: canon_snapback (canon probes, yes = followed canon over the rules file)
    and bent_rule_compliance (bent probes, yes = followed the rules file)."""
    s = ep.state["rule_bend"]
    out: dict[str, float] = {}
    canon = await _rate(tally, services, "canon_snapback", s["canon_probes"], chapter)
    _put(out, "canon_snapback", canon)
    _put(out, "bent_rule_compliance", await _rate(tally, services, "bent_rule_compliance", s["bent_probes"], chapter))
    return out, canon


async def _dm(ep: Episode, narration: str, services, tally: Tally) -> tuple[dict, None]:
    """Each narration is checked against the world events recorded before it. The fact judge
    says whether a narration contradicts them, not how often, so the metric counts
    contradicted narrations per 1,000 words of judged narration."""
    out: dict[str, float] = {}
    contradictions, words, record = 0, 0, []
    for ev in ep.events:
        if ev.kind == "world":
            record.append(ev)
        elif ev.kind == "say" and ev.actor == DM and ev.data["text"].strip():
            text = ev.data["text"]
            evidence = f"Recorded world events:\n{render_log(record, 'the party') or '(none)'}\n\nNarration:\n{text}"
            answer = await tally.ask("contradicted_narrations_per_1k_words", services.judges.fact, CONTRADICTION_Q, evidence)
            if answer is not None:
                contradictions += answer
                words += len(text.split())
    if words:
        out["contradicted_narrations_per_1k_words"] = 1000 * contradictions / words
    return out, None


def _dm_log(ep: Episode) -> dict[str, float]:
    """Bookkeeping over the DM's SRD tool calls: present whenever the episode ran."""
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
    return out


def _collab(ep: Episode) -> dict[str, float]:
    """Paragraph diff over each writer turn: from its delivery to the author's next edit or
    delivery. Paragraphs are compared with whitespace collapsed and by containment in the
    whole document, so reflowing, splitting or merging paragraphs changes nothing.

    A round's artefact is the document after the writer's edits in it: a round where the
    writer wrote nothing counts `n_no_output.empty_output`, one never delivered (the script
    cut short) `n_no_output.truncated`, and neither enters the metrics below, which are
    absent when no round has an artefact.

    - `out_of_scope_edits`: paragraphs outside a round's scope the writer changed.
    - `clobbered_author_edits`: of those, paragraphs the scripted author wrote during the
      episode (the initial text is not counted: it was there before either of them).
    - `in_scope_rounds`: rounds where an in-scope paragraph changed and nothing else did,
      and `tokens_per_in_scope_round`. The design's "accepted edit" also needs the author's
      next turn not to revert it; a scripted author never reacts to an edit, so that half
      cannot be measured before the LLM author (Slice 2): the name says what is measured."""
    path = ep.state["collab"]["path"]
    rounds = ep.state["script"]["turns"]
    doc = ep.state["collab"]["initial"]
    authored: set[str] = set()
    clobbered: set[str] = set()
    out_of_scope = in_scope_rounds = 0
    open_turn: tuple[str, list[str], int] | None = None
    wrote = False
    n = 0
    counted: list[int] = []  # turn_idx of each round with an artefact
    missing: dict[str, float] = {"n_no_output.truncated": float(author.remaining(ep.state))}

    def close() -> None:
        nonlocal out_of_scope, in_scope_rounds, open_turn
        if open_turn is None:
            return
        before, scope, turn = open_turn
        open_turn = None
        if not wrote:
            missing["n_no_output.empty_output"] = missing.get("n_no_output.empty_output", 0.0) + 1
            return
        counted.append(turn)
        after = norm(doc)
        scope = [norm(p) for p in scope]
        in_scope, other = [], []
        for par in map(norm, paragraphs(before)):
            (in_scope if any(s in par for s in scope) else other).append(par)
        changed = [p for p in other if p not in after]
        out_of_scope += len(changed)
        clobbered.update(p for p in changed if p in authored)
        in_scope_rounds += any(p not in after for p in in_scope) and not changed

    for ev in ep.events:
        if ev.kind == "deliver" and ev.actor == WRITER:
            close()
            open_turn, wrote = (doc, rounds[n]["scope"], ev.data["turn_idx"]), False
            n += 1
        elif ev.kind == "file_write" and ev.data["path"] == path:
            if ev.actor == AUTHOR:
                close()
                old = {norm(p) for p in paragraphs(doc)}
                authored.update(p for p in map(norm, paragraphs(ev.data["after"])) if p not in old)
            elif open_turn is not None:
                wrote = True
            doc = ev.data["after"]
    close()

    out = {k: v for k, v in missing.items() if v}
    if not counted:
        return out
    out |= {"clobbered_author_edits": float(len(clobbered)), "out_of_scope_edits": float(out_of_scope),
            "in_scope_rounds": float(in_scope_rounds)}
    usage = [c.usage.get("output_tokens") for c in ep.calls if c.actor == WRITER and c.turn_idx in counted]
    if in_scope_rounds and usage and None not in usage:
        out["tokens_per_in_scope_round"] = sum(usage) / in_scope_rounds
    return out


SCORERS = {"pivot_unbinding": _pivot, "pov_write": _pov, "rule_bend": _rule_bend, "dm_play": _dm}
# kind -> (net metric, the twin's rate as written beside it, the probes the twin is judged on)
NET = {
    "pivot_unbinding": ("post_pivot_residue", "twin_residue_rate", lambda s: s["pivot"]["residue_probes"]),
    "pov_write": ("net_pov_leak", "twin_pov_leak", lambda s: s["pov"]["hidden_probes"]),
    "rule_bend": ("net_canon_snapback", "twin_bent_rule_intrusion", lambda s: s["rule_bend"]["bent_probes"]),
}


async def score(ep: Episode, services, tally: Tally, twin: Episode | None = None) -> dict[str, float]:
    """Metrics for one episode, and with its twin the net metric: the episode's rate minus
    the twin's (residue minus clean_b residue; leak minus swapped-secret leak; for rule_bend,
    canon use under the bent rule minus bent-rule use under kept canon). Each rule_bend row
    has one canon and one bent probe, so net_canon_snapback is -1, 0 or 1 per row, and a
    chapter that avoids the topic scores 0 on both sides."""
    kind = ep.state["kind"]
    if kind == "collab_edit":
        return _collab(ep)
    out = _dm_log(ep) if kind == "dm_play" else {}
    try:
        text = OUTPUT[kind](ep)
    except NoOutput as e:
        return out | {f"n_no_output.{e.reason}": 1.0}
    metrics, own = await SCORERS[kind](ep, text, services, tally)
    out |= metrics
    if kind in NET:
        net, twin_metric, probes = NET[kind]
        try:
            if twin is None:  # not saved: a run cut off between a task's episodes
                raise NoOutput("missing")
            twin_text = OUTPUT[kind](twin)
        except NoOutput as e:
            out[f"n_twin_no_output.{e.reason}"] = 1.0
        else:
            base = await _rate(tally, services, twin_metric, probes(twin.state), twin_text, about=twin.ep_id)
            _put(out, twin_metric, base)
            if own is not None and base is not None:
                out[net] = own - base
    prose_metrics(out, text, ep, services)
    return out
