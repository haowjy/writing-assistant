"""`wa-bench view`: a local, read-only page for reading task rows and runs.

    uv run wa-bench view [--run runs/base-1a-qwen3.5-9b runs/base-1a-gemma-4-26b ...] [--port 8766] [--host 127.0.0.1]

Serves `view.html` and a small JSON API. The page asks for one thing at a time (a run's
report, its episode list, one task's episodes, one task across runs, one task row): an
episode's call records hold whole message histories, so a run is never shipped to the
browser at once, and the histories never are (each call contributes its reply, reasoning,
tool calls and their results).

Runs are read through the same loader, twin pairing and report aggregation as `wa-bench
report` (`build_report`), and re-read when a file in the run changes, so a run still being
written can be watched. Fields an older build did not record load as their empty defaults
and show as "not recorded". A task row is shown by running the suite's own `setup` and
`play` with a Turn that answers nothing (`_no_model`): the episodes, forks, twin swaps and
delivered messages are the ones a run would make. Held-out `test` rows are served only when
the page asks for them by name (`test=held-out`); `local_dev` rows never are.

Read-only: only GET is served and nothing is written. Bound to loopback by default, and
then only Host headers naming loopback are answered (a page elsewhere cannot rebind a name
onto this server and read book text); `--host` binds another address (the tailnet one),
which is refused while any run holds `local_dev` episodes. The page makes no external
requests (its CSP forbids them).
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from writing_assistant.cli import row_meta, twins_of
from writing_assistant.cli.report import build_report, guardrails_line
from writing_assistant.corpus import Corpus
from writing_assistant.episode import Episode, strings
from writing_assistant.episode.store import LOCAL_DEV, RUN_FILE, SPLITS, Task, load_episodes, read_jsonl, task_rows
from writing_assistant.llm.client import LLMClient
from writing_assistant.suites import SUITES, Services

PAGE = Path(__file__).with_name("view.html")
LOOPBACK = ("127.0.0.1", "localhost", "::1")
# Metrics the page folds into a "prose profile" (length, readability and copy overlap of every
# prose output): the same for every task kind, so they would bury the outcome metrics.
PROSE = ("readability_", "overlap_")


def is_prose(metric: str) -> bool:
    return metric.startswith(PROSE) and metric != "overlap_flagged"


class Run:
    """One run directory; what is read from it is cached until a file in it changes."""

    def __init__(self, path: Path, passed: set[str]) -> None:
        self.path, self.name, self.passed = path, path.name, passed
        self._sig: tuple = ()
        self._cache: dict[Any, Any] = {}

    def _cached(self, key: Any, make: Any) -> Any:
        sig = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(self.path.iterdir()) if p.is_file())
        if sig != self._sig:
            self._sig, self._cache = sig, {}
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def info(self) -> dict[str, Any]:
        return self._cached("info", lambda: json.loads((self.path / RUN_FILE).read_text())
                            if (self.path / RUN_FILE).exists() else {})

    def profiles(self) -> list[str]:
        """Judge profiles with a scores file, `eval` first."""
        names = [p.name.removeprefix("scores-").removesuffix(".jsonl") for p in self.path.glob("scores-*.jsonl")]
        return sorted(names, key=lambda n: (n != "eval", n))

    def profile(self, asked: str | None) -> str:
        have = self.profiles()
        return asked if asked in have else (have[0] if have else "eval")

    def episodes(self) -> dict[str, Episode]:
        return self._cached("episodes", lambda: {ep.ep_id: ep for ep in load_episodes(self.path)})

    def twins(self) -> dict[str, Episode]:
        """Parent ep_id -> twin, as the scorer pairs them."""
        return self._cached("twins", lambda: twins_of(list(self.episodes().values()), SUITES[self.info()["suite"]]))

    def scores(self, profile: str) -> dict[str, dict]:
        return self._cached(("scores", profile),
                            lambda: {r["ep_id"]: r for r in read_jsonl(self.path / f"scores-{profile}.jsonl")})

    def judgments(self, profile: str) -> dict[str, list[dict]] | None:
        """Scored ep_id -> its judge items in the order asked; None when the run stored none."""
        def make() -> dict[str, list[dict]] | None:
            path = self.path / f"judgments-{profile}.jsonl"
            if not path.exists():
                return None
            out: dict[str, list[dict]] = defaultdict(list)
            for j in read_jsonl(path):
                out[j["scored"]].append(j)
            return out
        return self._cached(("judgments", profile), make)

    def report(self, profile: str) -> dict[str, Any]:
        return self._cached(("report", profile),
                            lambda: build_report(self.path, profile, ["kind", "condition"], None, None, self.passed))

    def summary(self) -> dict[str, Any]:
        info = self.info()
        if not info:
            return {"name": self.name, "guardrails": guardrails_line(self.path / "guardrails.json")}
        # The suite's judged metrics in its own order: the page's default metric per task kind.
        return {"name": self.name, "suite": info["suite"], "endpoint": info["endpoint"], "model": info["model"],
                "split": info["split"], "started": info.get("started"), "profiles": self.profiles(),
                "episodes": len(self.episodes()), "twins": len(self.twins()),
                "judged": list(SUITES[info["suite"]].judged_by)}


# -- what the page shows of an episode -----------------------------------------------------------

def model_actors(ep: Episode) -> set[str]:
    return {a for a, endpoint in ep.actors.items() if endpoint != "scripted"}


def written(ep: Episode) -> list[dict[str, str]]:
    """What the model wrote: the files a model actor wrote, else its reply to each delivery
    (the last call of each turn). Display only: scoring reads its own artefact per task kind."""
    models = model_actors(ep)
    paths = list(dict.fromkeys(e.data["path"] for e in ep.events if e.kind == "file_write" and e.actor in models))
    if paths:
        return [{"label": p, "text": ep.files.get(p, "")} for p in paths]
    turns = [(e.actor, e.data["turn_idx"]) for e in ep.events if e.kind == "deliver" and e.actor in models]
    out = []
    for actor, turn in turns:
        calls = [c for c in ep.calls if c.actor == actor and c.turn_idx == turn]
        if calls and calls[-1].response.strip():
            out.append({"label": f"turn {turn + 1}" if len(turns) > 1 else "reply", "text": calls[-1].response})
    return out


def _tool_result(content: Any) -> dict[str, Any]:
    """A tool result as {summary, text}: a read's file text apart from the rest of the JSON."""
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return {"summary": "", "text": content}
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        rest = {k: v for k, v in data.items() if k != "text"}
        return {"summary": json.dumps(rest, ensure_ascii=False), "text": data["text"]}
    return {"summary": json.dumps(data, ensure_ascii=False, indent=1), "text": ""}


def _turns(ep: Episode) -> list[dict[str, Any]]:
    """Each delivery with the calls that served it; calls outside a delivery come last."""
    by_id = {m.get("tool_call_id"): m.get("content") for c in ep.calls for m in c.messages if m.get("role") == "tool"}
    # A call's results reach the record in the next call's messages, so the last call's do not:
    # those come from the tool events, paired in order when every call has its event.
    events = defaultdict(list)
    for e in ep.events:
        if e.kind == "tool":
            events[e.actor].append(e.data.get("result"))
    made = defaultdict(int)
    for c in ep.calls:
        made[c.actor] += len(c.tool_calls)
    seen: dict[str, int] = defaultdict(int)

    def call_view(c: Any) -> dict[str, Any]:
        tools = []
        for t in c.tool_calls:
            k = seen[c.actor]
            seen[c.actor] += 1
            if t.get("id") in by_id:
                result = _tool_result(by_id[t["id"]])
            elif made[c.actor] == len(events[c.actor]):
                result = _tool_result(events[c.actor][k])
            else:
                result = None
            tools.append({"name": t.get("name"), "arguments": t.get("arguments"), "result": result})
        return {"actor": c.actor, "endpoint": c.endpoint, "finish_reason": c.finish_reason, "usage": c.usage,
                "reasoning": c.reasoning, "response": c.response, "tool_calls": tools,
                "served": c.ext.get("served")}

    calls = [call_view(c) for c in ep.calls]
    out = []
    for e in ep.events:
        if e.kind == "deliver":
            idx = e.data["turn_idx"]
            out.append({"actor": e.actor, "turn_idx": idx, "message": e.data["message"],
                        "calls": [v for c, v in zip(ep.calls, calls, strict=True) if c.actor == e.actor and c.turn_idx == idx]})
    if loose := [v for c, v in zip(ep.calls, calls, strict=True) if c.turn_idx is None]:
        out.append({"actor": "", "turn_idx": None, "message": None, "calls": loose})
    return out


def _system(ep: Episode) -> dict[str, str]:
    """Each model actor's system prompt as sent (the first call's), else as set up."""
    out = {}
    for actor in sorted(model_actors(ep)):
        first = next((c for c in ep.calls if c.actor == actor and c.messages), None)
        if first and first.messages[0].get("role") == "system":
            out[actor] = first.messages[0].get("content") or ""
        elif prompt := ep.state.get("system_prompts", {}).get(actor):
            out[actor] = prompt
    return out


def _event_text(e: Any) -> str:
    d = e.data
    if e.kind == "file_write":
        return f"writes {d.get('path')} (rev {d.get('rev')})"
    if e.kind == "tool":
        return f"{d.get('name')} {json.dumps(d.get('args'), ensure_ascii=False)[:200]}"
    if e.kind == "deliver":
        return d.get("message", "")
    return d.get("text") or json.dumps(d, ensure_ascii=False)[:300]


def _files(ep: Episode) -> list[dict[str, Any]]:
    """Every file: the ones written in play with their text before the first write and after the
    last; the ones nobody wrote by path and size (a model that read one has it in a tool result)."""
    out = []
    for path, text in ep.files.items():
        writes = [e for e in ep.events if e.kind == "file_write" and e.data.get("path") == path]
        if not writes:
            out.append({"path": path, "chars": len(text)})
            continue
        before = writes[0].data.get("before")
        out.append({"path": path, "chars": len(text), "final": text, "initial": before,
                    "writers": sorted({e.actor for e in writes}),
                    "diff": diff_lines(before, text) if before is not None else None})
    return out


def diff_lines(a: str, b: str, context: int = 1) -> list[list[str]]:
    """[op, line] for a unified line diff; op is " ", "-", "+" or "@" (a gap between hunks)."""
    out = []
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), n=context, lineterm=""):
        if line.startswith(("---", "+++")):
            continue
        out.append(["@", ""] if line.startswith("@@") else [line[0], line[1:]])
    return out


def _prompt(ep: Episode) -> str:
    """What the model was shown up front: system prompts and every delivered message."""
    parts = [p for p in _system(ep).values()]
    parts += [e.data["message"] for e in ep.events if e.kind == "deliver"]
    return "\n\n".join(parts)


def _mounted(ep: Episode) -> dict[str, str]:
    """Each file as the benchmark mounted it: its text before the first write, else as it is."""
    out = dict(ep.files)
    for e in reversed(ep.events):
        if e.kind == "file_write" and e.data.get("before") is not None:
            out[e.data["path"]] = e.data["before"]
    return out


def _shown(ep: Episode, against: Episode | None) -> dict[str, Any]:
    """The whole prompt as [op, line], with the lines that differ from `against`'s marked, and
    the mounted files whose text differs from `against`'s (a twin can differ in a file only)."""
    prompt = _prompt(ep)
    # whole-prompt context: one hunk, so its "@" header marks no gap
    lines = [d for d in diff_lines(_prompt(against), prompt, sys.maxsize) if d[0] != "@"] if against else []
    files = []
    if against:
        mine, theirs = _mounted(ep), _mounted(against)
        files = [{"path": p, "lines": diff_lines(theirs.get(p, ""), mine.get(p, ""), 2)}
                 for p in sorted(set(mine) | set(theirs)) if mine.get(p) != theirs.get(p)]
    return {"against": against.ep_id if against else None,
            "lines": lines or [[" ", line] for line in prompt.splitlines()], "files": files}


def _judgments(items: list[dict], about: dict[str, tuple[Episode, str]], scene: dict[str, Any]) -> list[dict]:
    """The judge items behind one scores row. `about` maps each judged episode to its name on
    the page. Evidence that is a judged episode's own output is not shipped again; A/B sides
    are named when they are that output, the real passage or the impostor's."""
    texts = {w["text"].strip(): label for ep, label in about.values() for w in written(ep)}
    for name, label in (("y_star", "the real passage"), ("impostor", "another author's passage")):
        if isinstance(scene.get(name), dict):
            texts.setdefault(scene[name].get("text", "").strip(), label)
    out = []
    for j in items:
        v = {k: j.get(k) for k in ("id", "metric", "question", "model", "unscored", "judge", "replies")}
        v["about"] = j.get("ep_id")
        if "evidence" in j:
            src = texts.get(j["evidence"].strip())
            v["evidence_is"] = src
            v["evidence"] = None if src else j["evidence"]
        if "a" in j:
            ctx = j.get("context")
            v["context"] = ctx if isinstance(ctx, list) else [ctx] if ctx else []
            for side in ("a", "b"):
                v[side] = j[side]
                v[f"{side}_is"] = texts.get(j[side].strip())
        out.append(v)
    return out


def _outcome(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if not k.startswith("n_") and not is_prose(k)}


def _score_view(row: dict | None) -> dict[str, Any]:
    """A scores row as the page shows it: why there are no metrics, or the metrics in two parts."""
    if row is None:
        return {"not_scored": "no scores row"}
    if "error" in row:
        return {"not_scored": f"scorer error: {row['error']}"}
    if "skipped" in row:
        return {"not_scored": f"not scored: the episode failed ({row['skipped']})"}
    if "baseline_of" in row or "metrics" not in row:
        return {}
    m = row["metrics"]
    return {"metrics": _outcome(m), "prose": {k: v for k, v in m.items() if is_prose(k)},
            "counts": {k: v for k, v in m.items() if k.startswith("n_")}, "code": row.get("code")}


def _row_fields(ep: Episode) -> dict[str, Any]:
    """The task row's short fields: what distinguishes one task from another in a list."""
    return {k: v for k, v in ep.meta.get("row", {}).items()
            if k not in ("kind", "split", "work_ids") and isinstance(v, str | int | float) and len(str(v)) <= 60}


def _kind(ep: Episode) -> str:
    return ep.state.get("kind") or ep.meta.get("row", {}).get("kind") or "?"


def episode_list(run: Run, profile: str) -> list[dict[str, Any]]:
    twin_ids = {t.ep_id for t in run.twins().values()}
    scores = run.scores(profile)
    out = []
    for ep in run.episodes().values():
        if ep.ep_id in twin_ids:
            continue
        sc = _score_view(scores.get(ep.ep_id))
        counts = sc.get("counts", {})
        out.append({"ep_id": ep.ep_id, "task_id": ep.task_id, "kind": _kind(ep), "condition": ep.condition,
                    "status": ep.status, "failure": ep.failure_reason, "row": _row_fields(ep),
                    "metrics": sc.get("metrics", {}), "not_scored": sc.get("not_scored"),
                    "no_output": [k.split(".", 1)[1] for k in counts if k.startswith("n_no_output.")],
                    "words": sum(len(w["text"].split()) for w in written(ep))})
    return out


def episode_family(run: Run, ep_id: str, profile: str, runs: dict[str, Run]) -> dict[str, Any] | None:
    """The episode's task as played in this run: every arm with its twin, each with what it was
    shown, what it wrote, how (turns and calls), its scores and the judge items behind them."""
    eps = run.episodes()
    if ep_id not in eps:
        return None
    task_id = eps[ep_id].task_id
    family = [ep for ep in eps.values() if ep.task_id == task_id]
    twins = run.twins()
    twin_of = {t.ep_id: p for p, t in twins.items()}
    ordered = []
    for ep in family:
        if ep.ep_id not in twin_of:
            ordered.append(ep)
            if ep.ep_id in twins and twins[ep.ep_id] in family:
                ordered.append(twins[ep.ep_id])
    ordered += [ep for ep in family if ep not in ordered]
    scores, judgments = run.scores(profile), run.judgments(profile)
    arms = [ep for ep in ordered if ep.ep_id not in twin_of]
    out = []
    for ep in ordered:
        scene = ep.state.get("scene", {}) if isinstance(ep.state.get("scene"), dict) else {}
        parent = twin_of.get(ep.ep_id)
        # What its prompt is compared with: a twin with its parent, another arm with the first.
        against = parent or (arms[0].ep_id if ep is not arms[0] and ep in arms else None)
        about = {ep.ep_id: (ep, "this episode's output")}
        if ep.ep_id in twins:
            about[twins[ep.ep_id].ep_id] = (twins[ep.ep_id], "the twin's output")
        v = {"ep_id": ep.ep_id, "condition": ep.condition, "twin_of": parent, "status": ep.status,
             "failure": ep.failure_reason, "actors": ep.actors, "meta": {k: ep.meta.get(k) for k in
                                                                        ("temperature", "max_tokens", "code", "harness")},
             "system": _system(ep), "written": written(ep), "turns": _turns(ep), "files": _files(ep),
             "events": [{"idx": e.idx, "actor": e.actor, "kind": e.kind, "text": _event_text(e),
                         "seen_by": sorted(e.visible_to)} for e in ep.events],
             "score": {} if parent else _score_view(scores.get(ep.ep_id)),
             "judgments": None if judgments is None or parent else _judgments(judgments.get(ep.ep_id, []), about, scene),
             "shown": _shown(ep, eps[against] if against else None)}
        if scene:
            v["scene"] = {k: scene[k] for k in ("mode", "k", "exemplars", "y_star", "beat_sheet") if k in scene}
        if isinstance(edit := ep.state.get("edit"), dict) and edit.get("path") in ep.files:
            v["edit"] = {"path": edit["path"], "objection": edit.get("objection"),
                         "diff": diff_lines(edit.get("original", ""), ep.files[edit["path"]])}
        out.append(v)
    first = eps[ep_id]
    return {"run": run.summary(), "profile": profile, "profiles": run.profiles(), "task_id": task_id,
            "kind": _kind(first), "row": _row_fields(first), "split": first.split, "episodes": out,
            "elsewhere": [n for n, r in runs.items() if n != run.name and r.info() and
                          any(e.task_id == task_id for e in r.episodes().values())]}


def compare(runs: dict[str, Run], task_id: str, asked: dict[str, str]) -> list[dict[str, Any]]:
    """One task across runs: each run's episodes of it with what they wrote, their outcome
    metrics and the judges' answers (no reasoning, prompts or call records)."""
    out = []
    for name, run in runs.items():
        if not run.info():
            continue
        family = [ep for ep in run.episodes().values() if ep.task_id == task_id]
        if not family:
            continue
        profile = run.profile(asked.get(name))
        scores, judgments = run.scores(profile), run.judgments(profile) or {}
        twin_of = {t.ep_id: p for p, t in run.twins().items()}
        eps = []
        for ep in family:
            sc = {} if ep.ep_id in twin_of else _score_view(scores.get(ep.ep_id))
            eps.append({"ep_id": ep.ep_id, "condition": ep.condition, "twin_of": twin_of.get(ep.ep_id),
                        "status": ep.status, "failure": ep.failure_reason, "written": written(ep),
                        "metrics": sc.get("metrics", {}), "not_scored": sc.get("not_scored"),
                        "judged": [{"metric": j["metric"], "question": j["question"], "model": j.get("model"),
                                    "unscored": j.get("unscored"), "about": j.get("ep_id")}
                                   for j in judgments.get(ep.ep_id, [])]})
        info = run.info()
        out.append({"run": name, "endpoint": info["endpoint"], "model": info["model"], "profile": profile,
                    "episodes": eps})
    return out


# -- task rows, with no run --------------------------------------------------------------------

class Tasks:
    """Public task rows per (suite, split), read once. `local_dev` rows are never listed: they
    are read only under `local_only()`, which a page has no business entering."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], list[Task]] = {}

    def rows(self, suite: str, split: str) -> list[Task]:
        if (suite, split) not in self._rows:
            self._rows[suite, split] = task_rows(suite, split)
        return self._rows[suite, split]

    def find(self, suite: str, task_id: str) -> Task | None:
        return next((t for split in SPLITS for t in self.rows(suite, split) if t.task_id == task_id), None)


def task_list(tasks: Tasks, suite: str, splits: list[str]) -> list[dict[str, Any]]:
    out = []
    for split in splits:
        for t in tasks.rows(suite, split):
            fields = {k: v for k, v in row_meta(t).items() if k not in ("kind", "split", "work_ids") and len(str(v)) <= 60}
            out.append({"task_id": t.task_id, "kind": t.kind, "split": t.split, "row": fields})
    return out


async def _no_model(ep: Episode, actor: str, message: str) -> str:
    """The Turn the page plays a task with: it records the delivery and answers nothing, so
    every message the benchmark itself sends is on the record as the model would get it."""
    ep.deliver(actor, message)
    return ""


def _leaves(value: Any) -> list[str]:
    """The texts in a payload field worth testing for: strings of three words or more."""
    return [s for s in strings(value) if len(s.split()) >= 3]


def _given(ep: Episode) -> str:
    """Everything a model actor of `ep` could read: its system prompt, its deliveries and every
    revision of the files the ACL lets it reach (a file the author edits mid-play was read
    before the edit too)."""
    models = model_actors(ep)
    readable = {p for p in ep.files if any(ep.readable(a, p) for a in models)}
    revisions = [e.data[k] for e in ep.events if e.kind == "file_write" and e.data["path"] in readable
                 for k in ("before", "after") if e.data.get(k)]
    return "\n".join([*(_system(ep).values()), *(e.data["message"] for e in ep.events if e.kind == "deliver"),
                      *(ep.files[p] for p in readable), *revisions])


def task_family(task: Task, runs: dict[str, Run]) -> dict[str, Any]:
    """A task played with no model: the episodes `setup` forks and the messages `play` delivers
    (the suite's own code, so this cannot drift from a run), which payload fields reach the
    model in each episode, and what loaded runs scored and asked of it."""
    suite = SUITES[task.suite]
    services = Services(llm=None, judges=None, search=None, refs=None, corpus=Corpus())  # type: ignore[arg-type]
    head = {"task_id": task.task_id, "suite": task.suite, "kind": task.kind, "split": task.split,
            "row": {k: v for k, v in row_meta(task).items() if k not in ("kind", "split")}}
    try:
        eps = suite.setup(task, services)
        asyncio.run(suite.play(eps, services, _no_model))
    except Exception as e:  # a row this build cannot set up: show why instead of failing the page
        return head | {"error": f"{type(e).__name__}: {e}"}
    by_id = {ep.ep_id: ep for ep in eps}
    twin_of = {t.ep_id: p for p, t in twins_of(eps, suite).items()}
    arms = [ep for ep in eps if ep.ep_id not in twin_of]
    given = {ep.ep_id: _given(ep) for ep in eps}
    fields = []
    for name, value in task.payload.items():
        leaves = _leaves(value)
        if not leaves:
            continue
        seen = {ep.ep_id: sum(leaf in given[ep.ep_id] for leaf in leaves) / len(leaves) for ep in eps}
        fields.append({"field": name, "value": value, "reaches": seen})
    out = []
    for ep in eps:
        parent = twin_of.get(ep.ep_id) or (ep.parent_id if ep.parent_id in by_id else None)
        against = parent or (arms[0].ep_id if ep is not arms[0] else None)
        out.append({"ep_id": ep.ep_id, "condition": ep.condition, "twin_of": twin_of.get(ep.ep_id), "parent": parent,
                    "actors": ep.actors, "tools": ep.actor_tools, "system": _system(ep),
                    "shown": _shown(ep, by_id[against] if against else None),
                    "files": [{"path": p, "chars": len(t), "text": t,
                               "readable": sorted(a for a in model_actors(ep) if ep.readable(a, p))} for p, t in ep.files.items()],
                    "events": [{"idx": e.idx, "actor": e.actor, "kind": e.kind, "text": _event_text(e),
                                "seen_by": sorted(e.visible_to)} for e in ep.events]})
    return head | {"episodes": out, "fields": fields, "judged_by": suite.judged_by, "runs": _scored_in(task, runs),
                   "kind_scored": _kind_scored(task, runs)}


def _kind_scored(task: Task, runs: dict[str, Run]) -> dict[str, Any] | None:
    """The metrics the first loaded run that played this task's kind scored, per condition: how
    a task of the kind is scored when no loaded run played this task."""
    for name, run in runs.items():
        if not run.info() or run.info()["suite"] != task.suite:
            continue
        scores = run.scores(run.profile(None))
        by_condition: dict[str, set[str]] = defaultdict(set)
        for ep in run.episodes().values():
            if _kind(ep) == task.kind:
                by_condition[ep.condition] |= set(_score_view(scores.get(ep.ep_id)).get("metrics", {}))
        if any(by_condition.values()):
            return {"run": name, "metrics": {c: sorted(m) for c, m in by_condition.items()}}
    return None


def _scored_in(task: Task, runs: dict[str, Run]) -> list[dict[str, Any]]:
    """Each loaded run that played the task: its episodes of it, the metrics they scored and
    the judge questions asked (the questions, not the answers)."""
    out = []
    for name, run in runs.items():
        if not run.info() or run.info()["suite"] != task.suite:
            continue
        mine = [ep for ep in run.episodes().values() if ep.task_id == task.task_id]
        if not mine:
            continue
        profile = run.profile(None)
        scores, judgments = run.scores(profile), run.judgments(profile) or {}
        eps = []
        for ep in mine:
            asked = {(j["metric"], j["question"], j.get("ep_id") != ep.ep_id): None for j in judgments.get(ep.ep_id, [])}
            eps.append({"ep_id": ep.ep_id, "condition": ep.condition,
                        "metrics": sorted(_score_view(scores.get(ep.ep_id)).get("metrics", {})),
                        "asked": [{"metric": m, "question": q, "of_twin": t} for m, q, t in asked]})
        out.append({"run": name, "endpoint": run.info()["endpoint"], "profile": profile, "episodes": eps})
    return out


# -- the server -------------------------------------------------------------------------------

def make_handler(runs: dict[str, Run], loopback: bool) -> type[BaseHTTPRequestHandler]:
    tasks = Tasks()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if loopback and host not in LOOPBACK:
                return self._send(403, b"this server answers only to a loopback host name", "text/plain")
            url = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/runs":
                return self._json([r.summary() for r in runs.values()])
            if url.path in ("/api/tasks", "/api/task"):
                suite = q.get("suite", "")
                if suite not in SUITES:
                    return self._send(404, b"no such suite", "text/plain")
                # Reading held-out test rows is a use of the test split: only on an explicit ask.
                splits = [s for s in SPLITS if s != "test" or q.get("test") == "held-out"]
                if url.path == "/api/tasks":
                    return self._json({"suites": sorted(SUITES), "tasks": task_list(tasks, suite, splits)})
                task = tasks.find(suite, q.get("id", ""))
                if task is None or task.split not in splits:
                    return self._send(404, b"no such task (test rows need the held-out switch)", "text/plain")
                return self._json(task_family(task, runs))
            if url.path == "/api/compare":
                return self._json(compare(runs, q.get("task", ""), json.loads(q.get("profiles", "{}"))))
            run = runs.get(q.get("run", ""))
            if run is None or not run.info():
                return self._send(404, b"no such run", "text/plain")
            profile = run.profile(q.get("judges"))
            if url.path == "/api/report":
                return self._json(run.report(profile) | {"profile": profile, "profiles": run.profiles()})
            if url.path == "/api/episodes":
                return self._json({"profile": profile, "episodes": episode_list(run, profile)})
            if url.path == "/api/episode":
                fam = episode_family(run, q.get("ep", ""), profile, runs)
                return self._json(fam) if fam else self._send(404, b"no such episode", "text/plain")
            self._send(404, b"not found", "text/plain")

        def _json(self, data: Any) -> None:
            self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _send(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; "
                             "style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def load_runs(paths: list[Path], passed: set[str]) -> dict[str, Run]:
    """Run directories by name; a directory that is neither a run nor a guardrails result is
    skipped with a note, so `--run runs/*` works."""
    runs: dict[str, Run] = {}
    for path in paths:
        if not (path / RUN_FILE).exists() and not (path / "guardrails.json").exists():
            print(f"skipping {path}: no {RUN_FILE} or guardrails.json", file=sys.stderr)
            continue
        if path.name in runs:
            raise SystemExit(f"two runs are named {path.name}: {runs[path.name].path} and {path}")
        runs[path.name] = Run(path, passed)
    return runs


def cmd_view(args: argparse.Namespace) -> int:
    passed = set(LLMClient(args.models).config.get("gates", {}).get("passed", []))
    runs = load_runs([Path(p) for p in args.run], passed)
    loopback = args.host in LOOPBACK
    if not loopback:
        private = [r.name for r in runs.values() if r.info() and (r.info().get("split") == LOCAL_DEV or
                                                                  any(e.split == LOCAL_DEV for e in r.episodes().values()))]
        if private:
            raise SystemExit(f"{', '.join(private)} hold local_dev text: serve them on loopback only")
    server = HTTPServer((args.host, args.port), make_handler(runs, loopback))
    shown = "localhost" if loopback else args.host
    print(f"viewing {len(runs)} run{'s' * (len(runs) > 1)} at http://{shown}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
