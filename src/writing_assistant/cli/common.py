"""What more than one `wa-bench` command shares: how an episode is matched by `--where`,
reading a run directory some run has played in, ending a command, and `score`'s selection
and closing lines for a run directory and a world alike."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from writing_assistant.episode import Episode
from writing_assistant.episode.segments import Store, Unreadable, has_store, judging
from writing_assistant.episode.store import RUN_FILE
from writing_assistant.run.scoring import to_score
from writing_assistant.suites import Services


def episode_row(ep: Episode) -> dict[str, Any]:
    """What `--where` and `--group-by` test of a saved episode: its row's fields, its condition
    and, of a rollout, the checkpoint it was forked from (`<ep_id>@<n>`) and its group."""
    return (ep.meta.get("row", {}) | {"condition": ep.condition}
            | ({"rollout_of": "{}@{}".format(*ep.rollout_of), "group": ep.group} if ep.rollout_of else {}))


def folder_files(flag: str, folder: str) -> dict[str, str]:
    """path -> text of every file under `folder`, as an episode holds files."""
    root = Path(folder)
    if not root.is_dir():
        sys.exit(f"{flag} {root}: no folder there")
    try:
        return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}
    except UnicodeDecodeError as e:
        sys.exit(f"{flag} {root}: an episode's files are text, and one there is not UTF-8 ({e})")


def played_here(run_dir: Path) -> dict[str, Any]:
    """run.json of a directory some run has played in, for the commands that read a run.
    One that holds only what `situate` stored, or nothing, is refused in a sentence."""
    if (run_dir / RUN_FILE).exists():
        return json.loads((run_dir / RUN_FILE).read_text())
    if has_store(run_dir):
        raise Unreadable(f"{run_dir}: no run has played here. It holds situations and the positions they were set up at "
                         f"(`situate`), which are never scored or reported themselves. `run --from '<situation>@0'` plays "
                         f"one, here (`--out {run_dir}`) or in another directory (`--from-run {run_dir}`), and `score` "
                         "and `report` read the directory it played in")
    raise Unreadable(f"{run_dir}: no run directory there (no {RUN_FILE})")


async def close(services: Services) -> None:
    """End a command: cancel what is still in flight (a gather whose sibling raised leaves
    its other calls running), then close the client, so no request goes out after a stop."""
    stragglers = asyncio.all_tasks() - {asyncio.current_task()}
    for task in stragglers:
        task.cancel()
    await asyncio.gather(*stragglers, return_exceptions=True)
    await services.llm.aclose()


def selected(args: argparse.Namespace, store: Store, services: Services, **kw: Any) -> tuple[dict[str, dict[str, list[str]]], dict[str, int], set[str]]:
    """What this `score` has to do in `store`, by the one rule both of its forms share
    (`run.scoring.to_score`): the flags are `--judges`, `--rubric`, `--pending` and `--all`."""
    try:
        judging(args.judges, args.rubric or "")
    except ValueError as e:
        sys.exit(str(e))
    return to_score(store, args.judges, services.judges, rubric=args.rubric, pending=args.pending, again=args.all, **kw)


def scored(where: str, args: argparse.Namespace, suite: str, rows: list[dict], unknown: dict[str, int], services: Services) -> int:
    """`score`'s closing lines and exit code. A row whose every item went unscored (no
    thinking was recorded, the judge refused) is a row, and is not counted as scored."""
    errors = sum("error" in r for r in rows)
    rubrics = sorted({r["rubric"] for r in rows if "rubric" in r})
    empty = [r["metrics"] for r in rows if "metrics" in r and all(k.startswith("n_unscored.") for k in r["metrics"])]
    why = sorted({k.rsplit(".", 1)[1] for metrics in empty for k in metrics})
    print(f"{where}, judges {args.judges}{', rubrics ' + ', '.join(rubrics) if rubrics else ''}: "
          f"{len({r['ep_id'] for r in rows})} episodes, {len(rows)} rows: {sum('metrics' in r for r in rows) - len(empty)} scored, "
          f"{len(empty)} with nothing scored{' (' + ', '.join(why) + ')' if why else ''}, "
          f"{errors} scorer errors; spent ${sum(services.llm.spent.values()):.4f}")
    for name, n in sorted(unknown.items()):
        print(f"  rubric {name!r}: listed by {n} episodes, and suite {suite} has no rubric of that name; nothing was scored under it")
    return 1 if errors else 0
