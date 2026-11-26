"""`wa-bench`: run, score, report, guardrails, collect, export, sync.

A run directory (`runs/<suite>-<endpoint>-<UTC time>/` unless `--out` names one):

    run.json            what was run: suite, driver, endpoint, model, split, max_tokens; one
                        entry per invocation (time, code version, where, limit); per judge
                        profile the judges' stamps and the code that scored; the USD the hosts
                        reported. Each episode's meta has its own temperature, max_tokens and
                        code version, so a run resumed after a code change says which is which
    episodes.jsonl      every saved episode, appended one task at a time; never rewritten
    errors.jsonl        tasks that raised in setup or play (not saved; a resume reruns them)
    scores-<profile>.jsonl  one row per episode: metrics, or why it was not scored; a twin's
                        row says only whose baseline it is (`baseline_of`). `wa-bench score`
                        replaces the rows of the episodes it scored, never episodes.jsonl
    judgments-<profile>.jsonl  every labelable judge item behind the scores, one per line, in
                        label_ui.py's shape (question, evidence or a/b/context, `model` the
                        judge's answer or null) plus `scored` (the row it counts in), `ep_id`
                        (whose text was judged), `metric`, `unscored`, `judge` and `replies`
    guardrails.json     from `wa-bench guardrails --run`

`run` resumes a directory that already has a run.json for the same suite, driver, endpoint,
model, split and max_tokens (anything else is refused): tasks with saved episodes are
skipped (`--where` may differ between invocations) and saved episodes with no scores row are
scored first. One task that raises, or one episode whose scoring raises, is recorded and the
run goes on, but the command then exits 1. An account-level refusal from a host (HTTP 401,
402, 403: `llm.client.AccountError`) stops the run at once, exit 1; what was saved stays
resumable. Every append cuts off a line a killed run left unfinished first. `local_dev` rows are set up, played and
scored inside `episode.store.local_only()`, and only against a `local = true` endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from writing_assistant.corpus import Corpus
from writing_assistant.corpus.collect import collect
from writing_assistant.corpus.release import PERMISSIVE, export_public, stage_sync
from writing_assistant.drivers import DRIVERS, Driver, Session
from writing_assistant.episode import Episode
from writing_assistant.episode.store import (LOCAL_DEV, RUN_FILE, TEST_USES, Task, append_jsonl, load_episodes, local_only, matches,
                                             read_jsonl, save_episodes)
from writing_assistant.judges import Judges, judges_from
from writing_assistant.llm.client import AccountError, LLMClient
from writing_assistant.suites import SUBJECT, SUITES, Services, Suite, Tally

ROOT = Path(__file__).resolve().parents[3]
ERRORS_FILE = "errors.jsonl"
# task kind -> payload fields that carry work text (the licence rule in corpus/release.py)
TEXT_FIELDS = {kind: fields for suite in SUITES.values() for kind, fields in suite.text_fields.items()}


# -- the Turn binding ----------------------------------------------------------------------

class Binding:
    """The `Turn` a suite's `play()` calls: (episode, actor, message) -> reply. Logs the
    delivery with `Episode.deliver`, then runs the driver in the actor's session, opening one
    on first use of each (episode, actor); a fork has its own ep_id, so its own session.
    Refuses an actor still mapped to the `SUBJECT` placeholder or to `scripted`."""

    def __init__(self, driver: Driver) -> None:
        self.driver = driver
        self.sessions: dict[tuple[str, str], Session] = {}

    async def __call__(self, ep: Episode, actor: str, message: str) -> str:
        endpoint = ep.actors[actor]
        if endpoint in (SUBJECT, "scripted"):
            raise RuntimeError(f"{ep.ep_id}: actor {actor!r} is {endpoint!r}, not bound to an endpoint")
        key = (ep.ep_id, actor)
        if key not in self.sessions:
            self.sessions[key] = await self.driver.open(ep, actor, endpoint)
        ep.deliver(actor, message)
        return await self.driver.turn(self.sessions[key], message)

    async def close(self, ep_ids: set[str]) -> None:
        for key in [k for k in self.sessions if k[0] in ep_ids]:
            await self.driver.close(self.sessions.pop(key))


def bind_subject(eps: list[Episode], endpoint: str) -> None:
    for ep in eps:
        ep.actors = {a: endpoint if e == SUBJECT else e for a, e in ep.actors.items()}


# -- .env and shared setup -----------------------------------------------------------------

def load_dotenv(path: Path) -> None:
    """KEY=VALUE lines into the environment, never overriding what is already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def services_for(models: Path, profile: str) -> Services:
    client = LLMClient(models)
    judges = judges_from(client, profile)
    return Services(llm=client, judges=judges, search=None, refs=None, corpus=Corpus(judges=judges))


def judge_stamps(judges: Judges) -> dict[str, dict]:
    return {name: getattr(judges, name).stamp for name in judges.__dataclass_fields__}


def code_version() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def row_meta(task: Task) -> dict[str, Any]:
    """The row's scalar fields, stamped on each episode for `--where` and `--group-by`; a
    field marked as work text is not one of them."""
    text = {f.split(".")[0].removesuffix("[]") for f in TEXT_FIELDS.get(task.kind, [])}
    scalars = {k: v for k, v in task.payload.items() if isinstance(v, str | int | float | bool) and k not in text}
    return scalars | {"kind": task.kind, "split": task.split, "work_ids": ",".join(task.work_ids)}


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


# -- scoring -------------------------------------------------------------------------------

def twins_of(episodes: list[Episode], suite: Suite) -> dict[str, Episode]:
    """Parent ep_id -> its twin: the saved episode whose parent_id is the parent's ep_id and
    whose condition is the one `suite.twins` names for the parent's condition. A twin is a
    baseline, judged only through its parent's `score()`."""
    by_id = {ep.ep_id: ep for ep in episodes}
    return {ep.parent_id: ep for ep in episodes
            if ep.parent_id in by_id and suite.twins.get(by_id[ep.parent_id].condition) == ep.condition}


async def score_episode(suite: Suite, ep: Episode, twin: Episode | None, services: Services) -> tuple[dict, list[dict]]:
    """(scores row, the judgments behind it)."""
    row = {"ep_id": ep.ep_id, "task_id": ep.task_id, "condition": ep.condition, "status": ep.status,
           "twin": twin.ep_id if twin else None}
    if ep.status == "failed":
        return row | {"skipped": ep.failure_reason}, []
    tally = Tally()
    try:
        with local_only(ep.split == LOCAL_DEV):
            metrics = await suite.score(ep, services, tally, twin)
    except AccountError:
        raise
    except Exception as e:  # one episode's scorer failing must not stop the others
        return row | {"error": f"{type(e).__name__}: {e}"}, []
    judgments = [{"id": f"{ep.ep_id}#{n}", "scored": ep.ep_id, "ep_id": ep.ep_id} | j
                 for n, j in enumerate(tally.judgments)]
    return row | {"metrics": tally.into(metrics)}, judgments


async def score_all(suite: Suite, episodes: list[Episode], services: Services, concurrency: int,
                    only: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Rows for every episode, or for the ep_ids in `only` (a twin's row says whose baseline
    it is), and all their judgments. Twins are paired over all of `episodes`."""
    twins = twins_of(episodes, suite)
    baseline_of = {twin.ep_id: parent for parent, twin in twins.items()}
    sem = asyncio.Semaphore(concurrency)

    async def one(ep: Episode) -> tuple[dict, list[dict]]:
        if ep.ep_id in baseline_of:
            return {"ep_id": ep.ep_id, "task_id": ep.task_id, "condition": ep.condition, "status": ep.status,
                    "baseline_of": baseline_of[ep.ep_id]}, []
        async with sem:
            return await score_episode(suite, ep, twins.get(ep.ep_id), services)

    scored = await asyncio.gather(*(one(ep) for ep in episodes if only is None or ep.ep_id in only))
    return [row for row, _ in scored], [j for _, js in scored for j in js]


def record(run_dir: Path, services: Services, profile: str | None) -> None:
    """Add the USD the hosts reported to run.json and, when episodes were scored, which
    judges scored them."""
    info = json.loads((run_dir / RUN_FILE).read_text())
    spent = info.setdefault("spent_usd", {})
    for endpoint, usd in services.llm.spent.items():
        spent[endpoint] = round(spent.get(endpoint, 0.0) + usd, 6)
    if profile:
        info.setdefault("scoring", {})[profile] = {"judges": judge_stamps(services.judges), "code": code_version(), "at": _now()}
    _write_json(run_dir / RUN_FILE, info)


async def close(services: Services) -> None:
    """End a command: cancel what is still in flight (a gather whose sibling raised leaves
    its other calls running), then close the client, so no request goes out after a stop."""
    stragglers = asyncio.all_tasks() - {asyncio.current_task()}
    for task in stragglers:
        task.cancel()
    await asyncio.gather(*stragglers, return_exceptions=True)
    await services.llm.aclose()


# -- run -----------------------------------------------------------------------------------

async def cmd_run(args: argparse.Namespace) -> int:
    suite = SUITES[args.suite]
    services = services_for(args.models, args.judges)
    if args.endpoint not in services.llm.endpoints:
        sys.exit(f"unknown endpoint {args.endpoint!r}; models.toml has {sorted(services.llm.endpoints)}")
    if args.split == "test" and not args.held_out_test:
        sys.exit("--split test is the held-out split, never used to choose anything; pass --held-out-test to run "
                 f"it, and the run is logged in {TEST_USES}")
    if args.split == LOCAL_DEV and not services.llm.endpoints[args.endpoint].get("local"):
        sys.exit(f"--split local_dev plays personal text, which may reach only an endpoint with `local = true`; "
                 f"{args.endpoint!r} is hosted. Local endpoints: "
                 f"{sorted(n for n, e in services.llm.endpoints.items() if e.get('local'))}")
    run_dir = Path(args.out or f"runs/{args.suite}-{args.endpoint}-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"suite": args.suite, "driver": args.driver, "endpoint": args.endpoint,
                "model": services.llm.endpoints[args.endpoint]["model"], "split": args.split,
                "max_tokens": args.max_tokens}
    code = code_version()
    if (run_dir / RUN_FILE).exists():
        info = json.loads((run_dir / RUN_FILE).read_text())
        if {k: info.get(k) for k in identity} != identity:
            sys.exit(f"{run_dir} holds a different run: {({k: info.get(k) for k in identity})}; this one is {identity}")
    else:
        info = identity | {"started": _now(), "reported_apart": args.driver == "direct"}
    info.setdefault("invocations", []).append({"at": _now(), "code": code, "where": args.where, "limit": args.limit})
    _write_json(run_dir / RUN_FILE, info)
    if args.split == "test":
        append_jsonl(TEST_USES, [identity | {"at": _now(), "run": str(run_dir.resolve()), "code": code,
                                             "where": args.where, "limit": args.limit}])

    saved = load_episodes(run_dir)
    done = {ep.task_id for ep in saved}
    with local_only(args.split == LOCAL_DEV):
        tasks = [t for t in suite.tasks(args.split, args.where) if t.task_id not in done][: args.limit]
    scores_path, judgments_path = run_dir / f"scores-{args.judges}.jsonl", run_dir / f"judgments-{args.judges}.jsonl"
    # Rows about episodes no longer saved (a task cut off mid-write, which plays again) go.
    ids = {ep.ep_id for ep in saved}
    for path, key in ((scores_path, "ep_id"), (judgments_path, "scored")):
        rows = read_jsonl(path)
        if any(r[key] not in ids for r in rows):
            _write_jsonl(path, [r for r in rows if r[key] in ids])
    unscored = set() if args.no_score else {ep.ep_id for ep in saved} - {r["ep_id"] for r in read_jsonl(scores_path)}
    print(f"{run_dir}: {len(tasks)} tasks to run ({len(done)} already saved"
          + (f", {len(unscored)} saved episodes to score first)" if unscored else ")"), flush=True)
    driver = DRIVERS[args.driver](services.llm, suite.tools, max_tokens=args.max_tokens)
    turn = Binding(driver)
    meta = {"driver": args.driver, "harness": f"{args.driver}@{code}", "code": code, "endpoint": args.endpoint,
            "max_tokens": args.max_tokens, "run": run_dir.name}
    sem = asyncio.Semaphore(args.concurrency)
    counts: Counter[str] = Counter()

    async def score(eps: list[Episode], only: set[str] | None = None) -> None:
        rows, judgments = await score_all(suite, eps, services, len(eps) or 1, only)
        append_jsonl(judgments_path, judgments)
        append_jsonl(scores_path, rows)
        counts["score_errors"] += sum("error" in row for row in rows)

    async def one(task: Task) -> None:
        async with sem:
            eps: list[Episode] = []
            phase = "setup"
            try:
                with local_only(task.split == LOCAL_DEV):
                    eps = suite.setup(task, services)
                    bind_subject(eps, args.endpoint)
                    for ep in eps:
                        ep.meta |= meta | {"row": row_meta(task)}
                    phase = "play"
                    eps = await suite.play(eps, services, turn)
            except Exception as e:  # one task failing must not stop the run; resume reruns it
                counts["task_errors"] += 1
                append_jsonl(run_dir / ERRORS_FILE, [{"task_id": task.task_id, "phase": phase, "at": _now(),
                                                      "error": f"{type(e).__name__}: {e}",
                                                      "trace": traceback.format_exc(limit=-4)}])
                print(f"  {task.task_id}: {phase} error: {type(e).__name__}: {str(e)[:200]}", flush=True)
                if isinstance(e, AccountError):
                    raise
                return
            finally:
                await turn.close({ep.ep_id for ep in eps})
            for ep in eps:
                ep.complete()
                counts[ep.status] += 1
            save_episodes(eps, run_dir)
            if not args.no_score:
                await score(eps)
            print(f"  {task.task_id}: " + ", ".join(f"{ep.condition}={ep.status}" for ep in eps), flush=True)

    stopped: AccountError | None = None
    try:
        async with asyncio.TaskGroup() as tg:  # an AccountError in one task cancels the rest
            if unscored:
                tg.create_task(score(saved, unscored))
            for t in tasks:
                tg.create_task(one(t))
    except* AccountError as group:
        stopped = group.exceptions[0]
    finally:
        record(run_dir, services, None if args.no_score else args.judges)
        await close(services)
    print(f"{run_dir}: {dict(counts)}; spent ${sum(services.llm.spent.values()):.4f} "
          f"({', '.join(f'{k} ${v:.4f}' for k, v in services.llm.spent.items())})")
    if stopped:
        print(f"stopped: the host refused the account ({stopped}); nothing more can run until that is fixed. "
              f"Saved episodes stay; rerun the same command to resume.", file=sys.stderr)
    if stopped or counts["task_errors"] or counts["score_errors"]:
        return 1
    return 0


# -- score ---------------------------------------------------------------------------------

async def cmd_score(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    info = json.loads((run_dir / RUN_FILE).read_text())
    suite = SUITES[info["suite"]]
    episodes = load_episodes(run_dir)
    if args.where:
        episodes = [ep for ep in episodes if matches(ep.meta.get("row", {}), args.where)]
    services = services_for(args.models, args.judges)
    try:
        rows, judgments = await score_all(suite, episodes, services, args.concurrency)
    except* AccountError as group:
        sys.exit(f"stopped: the host refused the account ({group.exceptions[0]}); no scores were written")
    finally:
        await close(services)
    # Merged by ep_id: the rows and judgments of episodes `--where` did not select stay.
    path = run_dir / f"scores-{args.judges}.jsonl"
    _write_jsonl(path, list(({r["ep_id"]: r for r in read_jsonl(path)} | {r["ep_id"]: r for r in rows}).values()))
    rescored = {r["ep_id"] for r in rows}
    jpath = run_dir / f"judgments-{args.judges}.jsonl"
    _write_jsonl(jpath, [j for j in read_jsonl(jpath) if j["scored"] not in rescored] + judgments)
    record(run_dir, services, args.judges)
    errors = sum("error" in r for r in rows)
    print(f"{path}: {len(rows)} episodes, {sum('metrics' in r for r in rows)} scored, {errors} scorer errors; "
          f"spent ${sum(services.llm.spent.values()):.4f}")
    return 1 if errors else 0


# -- guardrails, collect, export, sync -----------------------------------------------------

def cmd_guardrails(args: argparse.Namespace) -> int:
    """The script reads the endpoint from `--models` and, as `.env` is already loaded here, the
    key from the environment; it creates nothing when the endpoint refuses a test request."""
    out = Path(args.run) if args.run else Path(f"runs/guardrails-{args.endpoint}-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}")
    done = subprocess.run([str(ROOT / "scripts" / "guardrails.sh"), args.endpoint, str(out), str(args.models.resolve())])
    if done.returncode != 0:
        sys.exit(f"guardrails.sh exited {done.returncode}; no guardrails.json stored")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    query: dict[str, Any] = {}
    if args.ids:
        query["ids"] = [int(i) for i in args.ids.split(",")]
    if args.search:
        query["search"] = args.search
    if args.limit:
        query["limit"] = args.limit
    since = dt.date.fromisoformat(args.since) if args.since else None
    works = collect(Corpus(), {args.source: query}, since=since)
    print(f"{len(works)} works written or refreshed: " + ", ".join(w.work_id for w in works[:20]))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    out = Path(args.public)
    export_public(Corpus(), out, TEXT_FIELDS)
    print(f"{out}: works.jsonl and tasks/ written (work text inline only for {sorted(PERMISSIVE)})")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    bucket = LLMClient(args.models).config.get("storage", {}).get("bucket", "")
    stage = Path(args.stage)
    if stage.exists():
        shutil.rmtree(stage)
    counts = stage_sync(stage, Path(args.runs), Corpus(), TEXT_FIELDS)
    print(f"staged {stage}: {dict(counts)}")
    if args.dry_run:
        return 0
    if not bucket:
        sys.exit("models.toml [storage] bucket is empty: no gs:// bucket is set up; staged only")
    if shutil.which("gcloud") is None:
        sys.exit("gcloud is not installed; staged only")
    return subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(stage), bucket]).returncode


# -- entry point ---------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wa-bench", description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, default=Path("models.toml"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="set up, play, save and score a suite's tasks against one endpoint")
    r.add_argument("--suite", required=True, choices=sorted(SUITES))
    r.add_argument("--endpoint", required=True, help="models.toml endpoint name of the subject")
    r.add_argument("--driver", default="direct", choices=sorted(DRIVERS))
    r.add_argument("--split", default="dev", choices=["train", "dev", "test", LOCAL_DEV])
    r.add_argument("--held-out-test", action="store_true",
                   help="required with --split test; each such run is appended to tasks/test-split-uses.jsonl")
    r.add_argument("--where", help="key=value[,key=value]: task row fields, then payload fields; a comma not followed by key= stays in the value")
    r.add_argument("--limit", type=int, help="run at most this many tasks")
    r.add_argument("--out", help="run directory (resumed when it exists)")
    r.add_argument("--judges", default="eval", choices=["eval", "train"])
    r.add_argument("--no-score", action="store_true", help="play and save only; score later")
    r.add_argument("--concurrency", type=int, default=8, help="tasks in flight")
    r.add_argument("--max-tokens", type=int, default=16384, help="per model call, thinking included")

    s = sub.add_parser("score", help="re-score a saved run's episodes (twins re-paired)")
    s.add_argument("--run", required=True)
    s.add_argument("--judges", default="eval", choices=["eval", "train"])
    s.add_argument("--where")
    s.add_argument("--concurrency", type=int, default=8)

    rp = sub.add_parser("report", help="metrics per group, with gates, refusals and baseline deltas")
    rp.add_argument("--run", required=True)
    rp.add_argument("--judges", default="eval", choices=["eval", "train"])
    rp.add_argument("--group-by", help="comma-separated row fields (kind, k, mode, author, condition, ...)")
    rp.add_argument("--where")
    rp.add_argument("--baseline")

    g = sub.add_parser("guardrails", help="IFEval, HumanEval+ and BFCL for one endpoint (scripts/guardrails.sh)")
    g.add_argument("--endpoint", required=True)
    g.add_argument("--run", help="store guardrails.json in this run directory")

    c = sub.add_parser("collect", help="import works into the corpus")
    c.add_argument("--source", default="gutenberg")
    c.add_argument("--since")
    c.add_argument("--ids", help="comma-separated source ids")
    c.add_argument("--search")
    c.add_argument("--limit", type=int)

    e = sub.add_parser("export", help="public release: permissive text inline, the rest hashed")
    e.add_argument("--public", required=True)

    y = sub.add_parser("sync", help="stage and rsync to the private gs:// bucket")
    y.add_argument("--runs", default="runs")
    y.add_argument("--stage", default="runs/.sync-stage")
    y.add_argument("--dry-run", action="store_true")
    return p


from writing_assistant.cli.report import cmd_report  # noqa: E402  (needs the helpers above)

COMMANDS = {"run": cmd_run, "score": cmd_score, "report": cmd_report, "guardrails": cmd_guardrails,
            "collect": cmd_collect, "export": cmd_export, "sync": cmd_sync}


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(".env"))
    args = parser().parse_args(argv)
    result = COMMANDS[args.cmd](args)
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


if __name__ == "__main__":
    sys.exit(main())
