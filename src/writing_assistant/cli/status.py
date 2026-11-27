"""`wa-bench status --run DIR | --world NAME`: what is running or last ran, how far it got,
what failed and what the same command would do next, from the run log (`run.Log`) and the
store. Read-only; it names no endpoint and calls none."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from writing_assistant.episode.library import Library
from writing_assistant.episode.segments import STORE_FILE, Store, has_store
from writing_assistant.episode.store import read_jsonl
from writing_assistant.run import LOG_FILE
from writing_assistant.run.directory import Refused, services_for
from writing_assistant.run.scoring import to_score


def cmd_status(args: argparse.Namespace) -> int:
    try:
        lines = _world(args) if args.world else _run(Path(args.run), args)
    except (LookupError, sqlite3.Error) as e:
        sys.exit(str(e))
    print("\n".join(lines))
    return 0


def _last(events: list[dict], cmds: tuple[str, ...]) -> list[dict]:
    """The events of the last invocation of one of `cmds`: its `started` line and what the
    same process logged after it."""
    starts = [i for i, e in enumerate(events) if e["event"] == "started" and e["cmd"] in cmds]
    if not starts:
        return []
    return [e for e in events[starts[-1]:] if e["pid"] == events[starts[-1]]["pid"]]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _said(mine: list[dict]) -> tuple[list[str], bool]:
    """An invocation in lines: when it started and how it ended, what it did, what failed.
    Also whether it is still running."""
    start, last = mine[0], mine[-1]
    running = last["event"] != "finished" and _alive(start["pid"])
    end = (f"finished in {last['seconds']} s" if last["event"] == "finished"
           else f"running (pid {start['pid']})" if running else "stopped without finishing (killed?)")
    asked = ", ".join(f"{k} {start[k]}" for k in ("endpoint", "judges", "rubric") if start.get(k))
    did = Counter(e["event"] for e in mine[1:] if e["event"] != "finished")
    results = Counter(e["result"] for e in mine if e["event"] == "judged")
    out = [f"`{start['cmd']}`{' (' + asked + ')' if asked else ''} started {start['at']}: {end}",
           "  " + (", ".join(f"{n} {event}" for event, n in did.items()) or "no event")
           + (f" ({', '.join(f'{n} {r}' for r, n in results.items())})" if results else "")
           + (f"; last: {last['event']} {last.get('ep_id') or last.get('name') or ''} at {last['at']}"
              if last["event"] not in ("started", "finished") else "")]
    failed = Counter(f"{e['name']} in {e['phase']}: {e['error']}" for e in mine if e["event"] == "failed")
    out += [f"  failed{f' {n} times' if n > 1 else ''}: {what}" for what, n in failed.items()]
    unjudged = [e for e in mine if e["event"] == "judged" and e["result"] in ("error", "unscored")]
    for (result, why), n in Counter((e["result"], e.get("why", "")) for e in unjudged).items():
        first = next(e for e in unjudged if (e["result"], e.get("why", "")) == (result, why))
        out.append(f"  {n} rows {result} ({why}), the first {first['ep_id']} under {first['judging']}")
    return out, running


def _run(run_dir: Path, args: argparse.Namespace) -> list[str]:
    if not has_store(run_dir):
        raise LookupError(f"{run_dir}: no run directory with a {STORE_FILE} there")
    events = read_jsonl(run_dir / LOG_FILE)
    out = [f"{run_dir}"]
    with Store(run_dir / STORE_FILE, readonly=True) as store:
        open_ = store.names(closed=False)
        out.append(f"  store: {len(store.names())} closed episodes, {len(open_)} open" + (f" ({', '.join(open_)})" if open_ else ""))
        played, scored = _last(events, ("run", "run --world")), _last(events, ("score",))
        if not played and not scored:
            return [*out, f"  no {LOG_FILE}: nothing has run here since runs were logged"]
        for mine in (played, scored):
            if not mine:
                continue
            said, running = _said(mine)
            out += said
            if running:
                continue
            start = mine[0]
            if start["cmd"] == "run --world":
                todo = sum(g["size"] - sum(store.is_closed(n) for n in store.rollouts(g["source"], 0, group))
                           for group, g in start["groups"].items())
                out.append(f"  next: the same command plays {todo} rollouts ({len(open_)} of them open, continued from their last turn)"
                           if todo else "  next: nothing to play; `wa-bench score --world` judges the world")
            elif start["cmd"] == "run":
                out.append(f"  next: the same command continues {len(open_)} open episodes and plays the task rows not yet stored"
                           if open_ else "  next: the same command plays the task rows not yet stored, if any")
            else:
                out.append(_to_score(store, args, start))
    return out


def _to_score(store: Store, args: argparse.Namespace, start: dict[str, Any]) -> str:
    """What the last `score` would do if run again, by the rule it would use."""
    try:
        want, _, _ = to_score(store, start["judges"], services_for(args.models, start["judges"]).judges, rubric=start.get("rubric"))
    except (Refused, KeyError) as e:
        return f"  next: {e}"
    rows = sum(len(rubrics) for wanted in want.values() for rubrics in wanted.values())
    return f"  next: the same command scores {rows} rows" if rows else "  next: nothing to score"


def _world(args: argparse.Namespace) -> list[str]:
    lib = Library()
    events = read_jsonl(lib.log_path(args.world))
    with lib.open(args.world) as store:
        groups = store.groups()
        rows = ", ".join(f"{name} {len(store.scores(name))}" for name in store.judgings()) or "none"
        out = [f"world {args.world}: {len(store.situations())} situations, {len(store.names())} closed rollouts, "
               f"{sum(g['played'] for g in groups.values())}/{len(groups)} groups played; score rows: {rows}"]
        scored = _last(events, ("score --world",))
        if scored:
            said, running = _said(scored)
            out += said + ([] if running else [_to_score(store, args, scored[0])])
    played = _last(events, ("run --world",))
    if played:  # the rollouts are played in a run directory: its log and store say how far it got
        out += _run(Path(played[0]["run_dir"]), args)
    if not played and not scored:
        out.append("  no log: nothing has run on this world since runs were logged")
    return out
