"""`wa-bench`: run, score, report, view, guardrails, collect, export, sync.

The commands parse arguments, open a run directory and choose what is played; the play
itself (bind, deliver, save, close, score) is `run.Runner`, which other callers use too.

A run directory (`runs/<suite>-<endpoint>-<UTC time>/` unless `--out` names one):

    run.json            what was run: suite, driver, endpoint, model, split, max_tokens; one
                        entry per invocation (time, code version, where, limit); the USD the
                        hosts reported; `served`, every build that answered each endpoint (the
                        response's model and system_fingerprint: on a local llama.cpp server,
                        the weights file and engine build). Each episode's meta has its own temperature, max_tokens
                        and code version, so a run resumed after a code change says which is which
    store.db            the episodes (`episode/segments.py`): each is saved before play, again
                        every time a delivery returns, and closed when its task's play ends.
                        Readers (`score`, `report`, `view`, `sync`) see closed episodes only
    errors.jsonl        tasks that raised in setup (a resume sets them up again) or in play
                        (their episodes stay open, and a resume continues them)
    run.lock            empty; every live `run` holds a lock on it, which is how one starting
                        up tells episodes a dead run left open from those another is playing
    scores-<profile>.jsonl  `<profile>` is `--judges`, a `[judges.<profile>]` table in
                        models.toml; `none` names no judge, so a run with no judge served
                        plays and scores what needs no model now and is re-scored later with
                        `score --judges eval`. One row per episode: metrics, or why it was not scored, and the
                        `code` version that scored it; a twin's row says only whose baseline it
                        is (`baseline_of`). `wa-bench score` replaces the rows of the episodes it
                        scored, never the store
    judgments-<profile>.jsonl  every labelable judge item behind the scores, one per line, in
                        label_ui.py's shape (question, evidence or a/b/context, `model` the
                        judge's answer or null) plus `scored` (the row it counts in), `ep_id`
                        (whose text was judged), `metric`, `unscored`, `judge` and `replies`
    guardrails.json     from `wa-bench guardrails --run`

`run` resumes a directory that already has a run.json for the same suite, driver, endpoint,
model, split and max_tokens (anything else is refused, and of two runs starting together in a
new directory one claims it: `run.run_info`): tasks with stored episodes are
skipped (`--where` may differ between invocations), closed episodes with no scores row are
scored first, and every open episode (a run killed mid-task, a task whose play raised) is
continued from its last turn checkpoint through the same loop as a fresh one, so a killed
run loses at most the deliveries in flight. A continued episode keeps the `meta` it started
with. Several `run`s may share a directory when they choose different tasks (`--where`,
`--from`): each appends to the one store, and only a `run` that finds no other alive there
continues the open episodes. One task that raises, or one episode whose scoring raises, is recorded and the
run goes on, but the command then exits 1. An account-level refusal from a host (HTTP 401,
402, 403: `llm.client.AccountError`) stops the run at once, exit 1; what was saved stays
resumable. `local_dev` rows are set up, played and
scored inside `episode.store.local_only()`, and only against a `local = true` endpoint.

`run --from <ep_id>@<n> --rollouts N` plays no task rows: it forks turn checkpoint `n` of a
stored episode N times and plays each fork on from there against `--endpoint`. `n` counts
that episode's own checkpoints (0: its start; a rollout's start is where it was forked), so
for an episode `run` set up it is the number of deliveries that had returned. The episode is read from this run's store, or from
`--from-run`'s, whose chain is first copied into this run's store so the store stays
self-contained. Rollout `i` is named `<ep_id>@<n>.<i>` and records what it is a rollout of
(`Episode.rollout_of`), so each samples with its own seed; one already stored is not made
again. A rollout has no twin: it is scored on its own metrics and gets no `net_*`. A twin
can be rolled out like any episode, and its rollouts are baselines of nothing. Only the
subject's actors (`meta["subject_actors"]`) are rebound to `--endpoint`. `report`,
`view` and calibration sampling keep rollouts out of the run's own numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from writing_assistant.corpus import Corpus
from writing_assistant.corpus.collect import collect
from writing_assistant.corpus.release import PERMISSIVE, export_public, stage_sync
from writing_assistant.drivers import DRIVERS
from writing_assistant.episode import Episode
from writing_assistant.episode.segments import STORE_FILE, Store, Unreadable, closed_episodes, has_store
from writing_assistant.episode.store import LOCAL_DEV, RUN_FILE, TEST_USES, append_jsonl, local_only, matches, read_jsonl
from writing_assistant.judges import judges_from
from writing_assistant.llm.client import AccountError, LLMClient
from writing_assistant.run import ROOT, Runner, code_version, run_info, score_all
from writing_assistant.run import now as _now
from writing_assistant.suites import SUITES, Services

LOCK_FILE = "run.lock"
# task kind -> payload fields that carry work text (the licence rule in corpus/release.py)
TEXT_FIELDS = {name: suite.text_fields for name, suite in SUITES.items()}  # what export and sync may release


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
    if profile not in client.config["judges"]:
        sys.exit(f"no [judges.{profile}] in {models}; it has {sorted(client.config['judges'])}")
    judges = judges_from(client, profile)
    return Services(llm=client, judges=judges, search=None, refs=None, corpus=Corpus(judges=judges))


def episode_row(ep: Episode) -> dict[str, Any]:
    """What `--where` and `--group-by` test of a saved episode: its row's fields, its condition
    and, of a rollout, the checkpoint it was forked from (`<ep_id>@<n>`)."""
    return (ep.meta.get("row", {}) | {"condition": ep.condition}
            | ({"rollout_of": "{}@{}".format(*ep.rollout_of)} if ep.rollout_of else {}))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


# -- run.json and the end of a command ---------------------------------------------------

def record_usage(run_dir: Path, services: Services) -> None:
    """Add the USD the hosts reported, and the builds that answered, to run.json."""
    with run_info(run_dir) as info:
        spent = info.setdefault("spent_usd", {})
        for endpoint, usd in services.llm.spent.items():
            spent[endpoint] = round(spent.get(endpoint, 0.0) + usd, 6)
        served = info.setdefault("served", {})
        for endpoint, builds in services.llm.served.items():
            served[endpoint] = sorted(set(served.get(endpoint, [])) | builds)


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
    if args.source and (args.where or args.limit):
        sys.exit("--from plays rollouts of one stored episode; --where and --limit choose task rows and do not apply")
    if not args.source and (args.from_run or args.rollouts != 1):
        sys.exit("--from-run and --rollouts go with --from <ep_id>@<n>")
    run_dir = Path(args.out or f"runs/{args.suite}-{args.endpoint}-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}")
    stored = has_store(run_dir)  # a directory written before the store is refused, not given a store
    # What --from names is looked at before the directory is made or its run.json touched.
    source: tuple[str, int] | None = None
    home: Store | None = None  # the store --from's episode is read from when it is not this run's
    if args.source:
        name, _, at = args.source.rpartition("@")
        if not name or not at.isdigit():
            sys.exit(f"--from takes <ep_id>@<n>, the episode and its turn checkpoint; got {args.source!r}")
        source = (name, int(at))
        if args.from_run:
            if not has_store(args.from_run):
                sys.exit(f"--from-run {args.from_run}: no run directory with a {STORE_FILE} there")
            home = Store(Path(args.from_run) / STORE_FILE, readonly=True)
        elif not stored:
            sys.exit(f"--from {args.source}: {run_dir} has no {STORE_FILE} to take the episode from; "
                     "--from-run names the run that holds it")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"suite": args.suite, "driver": args.driver, "endpoint": args.endpoint,
                "model": services.llm.endpoints[args.endpoint]["model"], "split": args.split,
                "max_tokens": args.max_tokens}
    code = code_version()
    # The directory is this run's, or becomes it, before anything is stored in it: of two
    # first invocations one claims the identity and the other is held to it.
    with run_info(run_dir) as info:
        if not info:
            info |= identity | {"started": _now(), "reported_apart": args.driver == "direct"}
        elif {k: info.get(k) for k in identity} != identity:
            sys.exit(f"{run_dir} holds a different run: {({k: info.get(k) for k in identity})}; this one is {identity}")
        store = Store(run_dir / STORE_FILE)  # one this build does not write is refused here, run.json unchanged
    if args.split == "test":
        append_jsonl(TEST_USES, [identity | {"at": _now(), "run": str(run_dir.resolve()), "code": code,
                                             "where": args.where, "limit": args.limit}])

    runner = Runner(suite, services, store, run_dir, endpoint=args.endpoint, driver=args.driver,
                    max_tokens=args.max_tokens, profile=args.judges, concurrency=args.concurrency,
                    score=not args.no_score)
    stopped: AccountError | None = None

    # Held shared for as long as this run lives (the kernel drops it when the process dies).
    # Getting it exclusively first means no other run is alive in this directory.
    lock = (run_dir / LOCK_FILE).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        alone = True
    except BlockingIOError:
        alone = False
    fcntl.flock(lock, fcntl.LOCK_SH)

    with store:
        closed = list(store.episodes())
        with local_only(args.split == LOCAL_DEV):
            # Open episodes, grouped as they were played: a task's episodes close together, a
            # rollout alone. With another run alive here they are that run's, in play, and are
            # left to it, as are the closed ones it has yet to score.
            unfinished: dict[str, list[Episode]] = {}
            for ep in store.episodes(closed=False):
                unfinished.setdefault(ep.ep_id if ep.rollout_of else ep.task_id, []).append(ep)
            # A rollout stores no task: a directory holding only rollouts of a task has yet to play it.
            done = {ep.task_id for ep in [*closed, *(ep for eps in unfinished.values() for ep in eps)] if not ep.rollout_of}
            if not alone:
                print(f"{run_dir}: another run is alive here; its {sum(map(len, unfinished.values()))} open episodes are left to it")
                unfinished = {}
            if source:
                tasks, rollouts = [], runner.bind(_rollouts(args, store, home, *source))
            else:
                tasks, rollouts = [t for t in suite.tasks(args.split, args.where) if t.task_id not in done][: args.limit], []
        with run_info(run_dir) as info:  # every refusal is behind: this invocation plays
            info.setdefault("invocations", []).append(
                {"at": _now(), "code": code, "where": args.where, "limit": args.limit}
                | ({"from": args.source, "from_run": args.from_run, "rollouts": args.rollouts} if args.source else {}))
        unscored = ({ep.ep_id for ep in closed} - {r["ep_id"] for r in read_jsonl(run_dir / f"scores-{args.judges}.jsonl")}
                    if alone and not args.no_score else set())
        print(f"{run_dir}: " + (f"{len(rollouts)} rollouts of {args.source}" if args.source else f"{len(tasks)} tasks") + " to run ("
              + ", ".join([f"{len(done)} tasks already stored"]
                          + [f"{sum(map(len, unfinished.values()))} open episodes to continue"] * bool(unfinished)
                          + [f"{len(unscored)} closed episodes to score first"] * bool(unscored)) + ")", flush=True)

        try:
            async with asyncio.TaskGroup() as tg:  # an AccountError in one task cancels the rest
                if unscored:
                    tg.create_task(runner.score(closed, unscored))
                for task_id, eps in unfinished.items():
                    tg.create_task(runner.play(task_id, lambda eps=eps: eps))
                for t in tasks:
                    tg.create_task(runner.play(t.task_id, lambda t=t: runner.fresh(t)))
                for ep in rollouts:
                    tg.create_task(runner.play(ep.ep_id, lambda ep=ep: [ep]))
        except* AccountError as group:
            stopped = group.exceptions[0]
        finally:
            record_usage(run_dir, services)
            await close(services)
    print(f"{run_dir}: {dict(runner.counts)}; spent ${sum(services.llm.spent.values()):.4f} "
          f"({', '.join(f'{k} ${v:.4f}' for k, v in services.llm.spent.items())})")
    if stopped:
        print(f"stopped: the host refused the account ({stopped}); nothing more can run until that is fixed. "
              f"Saved episodes stay; rerun the same command to resume.", file=sys.stderr)
    if stopped or runner.counts["task_errors"] or runner.counts["score_errors"]:
        return 1
    return 0


def _rollouts(args: argparse.Namespace, store: Store, home: Store | None, name: str, at: int) -> list[Episode]:
    """`--from <name>@<at> --rollouts N`: the rollouts of that turn checkpoint not yet in the
    run's store, for the run to bind to its endpoint. The episode comes from this store or,
    with `--from-run`, from `home`, its chain copied in first."""
    try:
        with home or nullcontext(store) as home:
            source = home.load(name, at=at)
            if (source.suite, source.split) != (args.suite, args.split):
                sys.exit(f"--from {args.source} is a suite {source.suite} episode of the {source.split} split; "
                         f"this run is suite {args.suite}, split {args.split}")
            rollouts = [source.rollout(at, i) for i in range(args.rollouts)]
            stored = set(store.names()) | set(store.names(closed=False))
            for ep in rollouts:  # two runs' episodes may share a name, never a segment
                if ep.ep_id in stored and store.chain(ep.ep_id)[-2:-1] != [source.seg]:
                    sys.exit(f"--from {args.source}: {store.path} already holds {ep.ep_id}, a rollout of another episode "
                             f"named {name} (another run's); give this run's rollouts their own --out")
            if home is not store:
                home.copy_chain(name, store)
    except (LookupError, ValueError, sqlite3.Error) as e:
        sys.exit(f"--from {args.source}: {e}")
    return [ep for ep in rollouts if ep.ep_id not in stored]


# -- score ---------------------------------------------------------------------------------

async def cmd_score(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    info = json.loads((run_dir / RUN_FILE).read_text())
    suite = SUITES[info["suite"]]
    episodes = closed_episodes(run_dir)
    only = {ep.ep_id for ep in episodes if matches(episode_row(ep), args.where)} if args.where else None
    services = services_for(args.models, args.judges)
    try:
        rows, judgments = await score_all(suite, episodes, services, args.concurrency, only)
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
    record_usage(run_dir, services)
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
                   help="required with --split test; each such run is appended to $WA_DATA/test-split-uses.jsonl")
    r.add_argument("--where", help="key=value[,key=value]: task row fields, then payload fields; a comma not followed by key= stays in the value")
    r.add_argument("--limit", type=int, help="run at most this many tasks")
    r.add_argument("--from", dest="source", metavar="EP_ID@N",
                   help="play rollouts of a stored episode's turn checkpoint N instead of task rows (see --rollouts)")
    r.add_argument("--from-run", help="the run directory whose store holds --from's episode (default: --out's own)")
    r.add_argument("--rollouts", type=int, default=1, help="with --from: how many forks of the checkpoint to play")
    r.add_argument("--out", help="run directory (resumed when it exists)")
    r.add_argument("--judges", default="eval", help="[judges.<profile>] in models.toml; `none` scores only what needs no model judge")
    r.add_argument("--no-score", action="store_true", help="play and save only; score later")
    r.add_argument("--concurrency", type=int, default=8, help="tasks in flight")
    r.add_argument("--max-tokens", type=int, default=16384, help="per model call, thinking included")

    s = sub.add_parser("score", help="re-score a saved run's episodes (twins re-paired)")
    s.add_argument("--run", required=True)
    s.add_argument("--judges", default="eval", help="[judges.<profile>] in models.toml; `none` scores only what needs no model judge")
    s.add_argument("--where", help="key=value[,key=value]: the episode's row fields and its condition, as in report")
    s.add_argument("--concurrency", type=int, default=8)

    rp = sub.add_parser("report", help="metrics per group, with gates, refusals and baseline deltas")
    rp.add_argument("--run", required=True)
    rp.add_argument("--judges", default="eval", help="[judges.<profile>] in models.toml; `none` scores only what needs no model judge")
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

    v = sub.add_parser("view", help="a local read-only page: task rows, and runs' scores, episodes and judges")
    v.add_argument("--run", nargs="+", action="extend", default=[], help="run directories (several to compare)")
    v.add_argument("--port", type=int, default=8766)
    v.add_argument("--host", default="127.0.0.1", help="address to bind; not loopback refuses runs with local_dev text")

    y = sub.add_parser("sync", help="stage and rsync to the private gs:// bucket")
    y.add_argument("--runs", default="runs")
    y.add_argument("--stage", default="runs/.sync-stage")
    y.add_argument("--dry-run", action="store_true")
    return p


from writing_assistant.cli.report import cmd_report  # noqa: E402  (needs the helpers above)
from writing_assistant.cli.view import cmd_view  # noqa: E402

COMMANDS = {"run": cmd_run, "score": cmd_score, "report": cmd_report, "guardrails": cmd_guardrails,
            "collect": cmd_collect, "export": cmd_export, "sync": cmd_sync, "view": cmd_view}


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(".env"))
    args = parser().parse_args(argv)
    try:
        result = COMMANDS[args.cmd](args)
        return asyncio.run(result) if asyncio.iscoroutine(result) else result
    except Unreadable as e:
        sys.exit(str(e))


if __name__ == "__main__":
    sys.exit(main())
