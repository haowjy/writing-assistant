"""`wa-bench`: run, situate, score, report, view, guardrails, collect, export, sync.

The commands parse arguments, open a run directory and choose what is played; the play
itself (bind, deliver, save, close, score) is `run.Runner`, which other callers use too.

A run directory (`runs/<suite>-<endpoint>-<UTC time>/` unless `--out` names one):

    run.json            what was run: suite, driver, endpoint, model, split, max_tokens, cast (each
                        cast name but `subject` with its endpoint and model); `id`,
                        random, minted when the directory is claimed (the run's identity in
                        every group it makes); one
                        entry per invocation (time, code version, where, limit); the USD the
                        hosts reported; `served`, every build that answered each endpoint (the
                        response's model and system_fingerprint: on a local llama.cpp server,
                        the weights file and engine build). Each episode's meta has its own temperature, max_tokens
                        and code version, so a run resumed after a code change says which is which
    store.db            the episodes and their score rows (`episode/segments.py`). Each episode
                        is saved before play, again every time a delivery returns, and closed
                        when its task's play ends. Readers (`score`, `report`, `view`, `sync`)
                        see closed episodes only. Score rows are kept per judge profile
                        (`--judges`, a `[judges.<profile>]` table in models.toml; `none` names
                        no judge, so a run with no judge served plays and scores what needs no
                        model now and is re-scored later with `score --judges eval`). One row
                        per episode: metrics, or why it was not scored (`skipped`: the episode
                        failed; `error`: the scorer raised), the `code` version that scored it,
                        when (`scored_at`) and by which judges (`judges`); a twin's row says
                        only whose baseline it is (`baseline_of`). With each row are the judge
                        items behind it, in label_ui.py's shape (question, evidence or
                        a/b/context, `model` the judge's answer or null) plus `scored` (the row
                        it counts in), `ep_id` (whose text was judged), `metric`, `unscored`,
                        `judge` and `replies`. Rows are only added: `wa-bench score` adds a
                        row for each episode it scores, as it is made. Every reader takes an
                        episode's newest row that is not an `error`, or its newest when all
                        are, so the row shown may be older than the last attempt
    errors.jsonl        tasks that raised in setup (a resume sets them up again) or in play
                        (their episodes stay open, and a resume continues them)
    run.lock            empty; every live `run` holds a lock on it, which is how one starting
                        up tells episodes a dead run left open from those another is playing
    guardrails.json     from `wa-bench guardrails --run`

`run` resumes a directory that already has a run.json for the same suite, driver, endpoint,
model, split, max_tokens and cast (anything else is refused, and of two runs starting together in a
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
self-contained. The N rollouts are a **group**: what this run made of this checkpoint in this
`--round` (default `0`). The group's id is 12 hex digits derived from run.json's `id`, the
round, the episode's name (its base's, for a twin: rollout `i` of each is then in one group)
and `n`, so a resumed run works it out again and makes only the members still missing.
Member `i` is named `<ep_id>@<n>.<group>.<i>` and records what it is a rollout of
(`Episode.rollout_of`, `Episode.group`), so each samples with its own seed and no two runs
make the same name. A group's size is fixed by the first command that asks for it (N per
episode rolled out; every member carries it in `meta["group_size"]`): asking again with
another N is refused, and more rollouts of the checkpoint are another `--round`. A rollout
has no twin: it is scored on its own metrics and gets no `net_*`. A twin
can be rolled out like any episode, and its rollouts are baselines of nothing. A rollout is
played by this run's cast, whatever played its source: a seat names a cast name (`subject`,
or one `--cast`'s table maps), so another `--endpoint` changes who plays the subject's seats
and nothing else. A seat whose name the cast lacks stops the command before play. `report`,
`view` and calibration sampling keep rollouts out of the run's own numbers; `report` lists
them per group and says which groups are settled (`Store.groups`).

`situate --run DIR --at <ep_id> --row task.json` sets a new task up on the files a closed
episode ended with and stores it in DIR's store as a **situation** (`run.situate`,
`Episode.begin`): the task takes the files and their revisions, not the transcript, and a
file some seats could not read stays closed to every new seat until the task's `setup`
grants it. The row file is one JSON object, `suite`, `kind` and `payload` (and `task_id` to
name the situation; otherwise `<root episode's name>/<12 random hex>`); the split and works
are the position's. `--from-run` names the run whose store holds the position when DIR's
does not. A situation is stored closed and is never played in place, scored or reported:
`run --from '<situation>@0' --rollouts N` plays a group of it, in a directory whose run is
of the situation's suite and split. `run` plays those rollouts and does not score them:
judging is its own pass (`score`), which nothing that plays or grows a world calls.

`score` re-scores a run's closed episodes into its store, row by row: one killed part-way
has added the rows it finished and left the other episodes with the rows they had.
`score --pending` re-scores only the episodes with no row but ones saying the scorer raised
(`error`), which is what a judge that was down leaves.

A suite's scoring is split into **rubrics**, and an episode has one row per **judging**: a
judge profile and a rubric (`episode.segments.judging`; suites 1a and 3 have one unnamed
rubric, stored under the bare profile). Commands take `--judges PROFILE [--rubric NAME]`,
never a joined name. With no `--rubric`, `score` scores each episode under the rubrics its
task lists (the kind's defaults, or the row's `payload["rubrics"]`), `score --pending`
covers every rubric the profile has rows for, and `report` prints every such rubric as a
block of its own.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from writing_assistant.corpus import Corpus
from writing_assistant.corpus.collect import collect
from writing_assistant.corpus.release import PERMISSIVE, export_public, stage_sync
from writing_assistant.drivers import DRIVERS
from writing_assistant.episode import Episode
from writing_assistant.episode.library import Library
from writing_assistant.episode.segments import STORE_FILE, Store, Unreadable, has_store, judging, split_judging
from writing_assistant.episode.store import LOCAL_DEV, RUN_FILE, TEST_USES, local_only, matches
from writing_assistant.llm.client import AccountError, LLMClient
from writing_assistant.run import ROOT, Runner, cast_from, code_version, run_info, situate, situated
from writing_assistant.run import now as _now
from writing_assistant.run.directory import Refused, open_run, record_usage, services_for
from writing_assistant.run.directory import identity as run_identity
from writing_assistant.run.scoring import score_into, wanted
from writing_assistant.suites import SUITES, Services

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


def episode_row(ep: Episode) -> dict[str, Any]:
    """What `--where` and `--group-by` test of a saved episode: its row's fields, its condition
    and, of a rollout, the checkpoint it was forked from (`<ep_id>@<n>`) and its group."""
    return (ep.meta.get("row", {}) | {"condition": ep.condition}
            | ({"rollout_of": "{}@{}".format(*ep.rollout_of), "group": ep.group} if ep.rollout_of else {}))


# -- run.json and the end of a command ---------------------------------------------------

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


# -- run -----------------------------------------------------------------------------------

async def cmd_run(args: argparse.Namespace) -> int:
    suite = SUITES[args.suite]
    services = services_for(args.models, args.judges)
    try:
        cast = cast_from(services.llm.config, args.cast) if args.cast else {}
    except ValueError as e:
        sys.exit(f"--cast {args.cast}: {e}")
    identity = run_identity(services, suite=args.suite, driver=args.driver, endpoint=args.endpoint, split=args.split,
                            max_tokens=args.max_tokens, cast=cast)
    if args.split == "test" and not args.held_out_test:  # `open_run` refuses it too; said here before anything else is looked at
        sys.exit(f"--split test is the held-out split, never used to choose anything; pass --held-out-test to run "
                 f"it, and the run is logged in {TEST_USES}")
    if args.source and (args.where or args.limit):
        sys.exit("--from plays rollouts of one stored episode; --where and --limit choose task rows and do not apply")
    if not args.source and (args.from_run or args.rollouts != 1 or args.round != "0"):
        sys.exit("--from-run, --rollouts and --round go with --from <ep_id>@<n>")
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
    code = code_version()
    # The directory is this run's, or becomes it, before anything is stored in it.
    opened = open_run(run_dir, identity, held_out_test=args.held_out_test, note={"where": args.where, "limit": args.limit})
    store, alone = opened.store, opened.alone
    runner = Runner(suite, services, store, run_dir, run_id=opened.run_id, endpoint=args.endpoint, driver=args.driver,
                    max_tokens=args.max_tokens, concurrency=args.concurrency, cast=cast, round=args.round)
    stopped: AccountError | None = None
    counts = runner.counts  # this command adds what its scoring did to what the runner played
    scoring = asyncio.Semaphore(args.concurrency)

    def listed(eps: list[Episode]) -> dict[str, list[str]]:
        """ep_id -> the rubrics this command scores it under: its task's listed ones. A
        rollout of a situation is left to `wa-bench score`."""
        return wanted(suite, [ep for ep in eps if not situated(ep)])[0]

    async def score(eps: list[Episode], want: dict[str, list[str]]) -> None:
        """Score closed episodes into the store (twins paired over all of `eps`)."""
        async with scoring:
            rows = await score_into(store, args.judges, suite, eps, services, len(eps) or 1, want)
        counts["score_errors"] += sum("error" in row for row in rows)

    async def play(name: str, begin: Callable[[], list[Episode]]) -> None:
        eps = await runner.play(name, begin)
        if eps and not args.no_score:
            await score(eps, listed(eps))

    with store:
        closed = list(store.episodes())
        with local_only(args.split == LOCAL_DEV):
            # With another run alive here the open episodes are that run's, in play, and are
            # left to it, as are the closed ones it has yet to score.
            unfinished = runner.unfinished()
            # A rollout stores no task: a directory holding only rollouts of a task has yet to play it.
            done = {ep.task_id for ep in [*closed, *(ep for eps in unfinished.values() for ep in eps)] if not ep.rollout_of}
            if not alone:
                print(f"{run_dir}: another run is alive here; its {sum(map(len, unfinished.values()))} open episodes are left to it")
                unfinished = {}
            if source:
                tasks, rollouts = [], _rollouts(args, runner, home, *source)
            else:
                tasks, rollouts = [t for t in suite.tasks(args.split, args.where) if t.task_id not in done][: args.limit], []
        with run_info(run_dir) as info:  # every refusal is behind: this invocation plays
            info.setdefault("invocations", []).append(
                {"at": _now(), "code": code, "where": args.where, "limit": args.limit}
                | ({"from": args.source, "from_run": args.from_run, "rollouts": args.rollouts, "round": args.round}
                   if args.source else {}))
        unscored: dict[str, list[str]] = {}  # closed episode -> its listed rubrics with no row yet
        if alone and not args.no_score:
            have = {split_judging(name)[1]: store.scores(name).keys() for name in store.judgings(args.judges)}
            unscored = {ep_id: missing for ep_id, rubrics in listed(closed).items()
                        if (missing := [r for r in rubrics if ep_id not in have.get(r, ())])}
        print(f"{run_dir}: " + (f"{len(rollouts)} rollouts of {args.source}" if args.source else f"{len(tasks)} tasks") + " to run ("
              + ", ".join([f"{len(done)} tasks already stored"]
                          + [f"{sum(map(len, unfinished.values()))} open episodes to continue"] * bool(unfinished)
                          + [f"{len(unscored)} closed episodes to score first"] * bool(unscored)) + ")", flush=True)

        try:
            async with asyncio.TaskGroup() as tg:  # an AccountError in one task cancels the rest
                if unscored:
                    tg.create_task(score(closed, unscored))
                for task_id, eps in unfinished.items():
                    tg.create_task(play(task_id, lambda eps=eps: eps))
                for t in tasks:
                    tg.create_task(play(t.task_id, lambda t=t: runner.fresh(t)))
                for ep in rollouts:
                    tg.create_task(play(ep.ep_id, lambda ep=ep: [ep]))
        except* AccountError as group:
            stopped = group.exceptions[0]
        finally:
            record_usage(run_dir, services)
            await close(services)
    print(f"{run_dir}: {dict(counts)}; spent ${sum(services.llm.spent.values()):.4f} "
          f"({', '.join(f'{k} ${v:.4f}' for k, v in services.llm.spent.items())})")
    if stopped:
        print(f"stopped: the host refused the account ({stopped}); nothing more can run until that is fixed. "
              f"Saved episodes stay; rerun the same command to resume.", file=sys.stderr)
    if stopped or counts["task_errors"] or counts["score_errors"]:
        return 1
    return 0


def _rollouts(args: argparse.Namespace, runner: Runner, home: Store | None, name: str, at: int) -> list[Episode]:
    """`--from <name>@<at> --rollouts N`: the members of that group the run's store does not
    hold yet. The episode comes from the run's store or, with `--from-run`, from `home`, its
    chain copied in first."""
    try:
        with home or nullcontext(runner.store) as home:
            source = home.load(name, at=at)
            if (source.suite, source.split) != (args.suite, args.split):
                sys.exit(f"--from {args.source} is a suite {source.suite} episode of the {source.split} split; "
                         f"this run is suite {args.suite}, split {args.split}")
            rollouts = runner.rollouts(source, at, args.rollouts)
            if home is not runner.store:
                home.copy_chain(name, runner.store)
    except (LookupError, ValueError, sqlite3.Error) as e:
        sys.exit(f"--from {args.source}: {e}")
    return rollouts


# -- situate -------------------------------------------------------------------------------

def cmd_situate(args: argparse.Namespace) -> int:
    row = json.loads(Path(args.row).read_text())
    if not isinstance(row, dict) or row.get("suite") not in SUITES or not isinstance(row.get("kind"), str):
        sys.exit(f"--row {args.row}: one JSON object with `suite` (one of {sorted(SUITES)}), `kind` and `payload`")
    if args.world:
        # In a world: the position and the situation are in one file, and no judge is reachable.
        if args.from_run:
            sys.exit("--from-run names a run directory that holds --at's episode; a world holds its own positions")
        services = services_for(args.models, "none")
        try:
            with Library().writing(args.world) as store:
                eps = situate(store, services, row["suite"], row["kind"], row.get("payload", {}), args.at, name=row.get("task_id"))
        except (LookupError, ValueError, sqlite3.Error) as e:
            sys.exit(f"--at {args.at}: {e}")
    else:
        run_dir = Path(args.run)
        if args.from_run and not has_store(args.from_run):
            sys.exit(f"--from-run {args.from_run}: no run directory with a {STORE_FILE} there")
        if not args.from_run and not has_store(run_dir):
            sys.exit(f"--at {args.at}: {run_dir} has no {STORE_FILE} to take the episode from; --from-run names the run that holds it")
        services = services_for(args.models, args.judges)
        run_dir.mkdir(parents=True, exist_ok=True)
        with Store(run_dir / STORE_FILE) as store, \
                (Store(Path(args.from_run) / STORE_FILE, readonly=True) if args.from_run else nullcontext()) as home:
            try:
                eps = situate(store, services, row["suite"], row["kind"], row.get("payload", {}), args.at,
                              name=row.get("task_id"), home=home)
            except (LookupError, ValueError, sqlite3.Error) as e:
                sys.exit(f"--at {args.at}: {e}")
    for ep in eps:
        print(ep.ep_id)
    return 0


# -- score ---------------------------------------------------------------------------------

async def cmd_score(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    info = played_here(run_dir)
    suite = SUITES[info["suite"]]
    if not has_store(run_dir):
        sys.exit(f"{run_dir}: no {STORE_FILE}; nothing has been played here")
    services = services_for(args.models, args.judges)
    with Store(run_dir / STORE_FILE) as store:
        episodes = list(store.episodes())
        chosen = [ep for ep in episodes if matches(episode_row(ep), args.where)] if args.where else episodes
        want, unknown = to_score(store, suite, chosen, args.judges, args.rubric, args.pending)
        try:
            rows = await score_into(store, args.judges, suite, episodes, services, args.concurrency, want)
        except* AccountError as group:
            sys.exit(f"stopped: the host refused the account ({group.exceptions[0]}); the rows scored before it are "
                     "stored, the others are as they were")
        finally:
            await close(services)
    record_usage(run_dir, services)
    return scored(str(run_dir), args, info["suite"], rows, unknown, services)


def to_score(store: Store, suite: Any, episodes: list[Episode], profile: str, rubric: str | None,
             pending: bool) -> tuple[dict[str, list[str]], dict[str, int]]:
    """What `score` scores (`run.scoring.wanted`): each episode's listed rubrics, or the one
    `--rubric` names. With `--pending`, only the rows saying the scorer raised, of every
    rubric the profile has rows for or of the one named."""
    try:
        judging(profile, rubric or "")
    except ValueError as e:
        sys.exit(str(e))
    if rubric is not None and rubric not in suite.rubrics:
        sys.exit(f"--rubric {rubric}: the suite has no rubric of that name; it has "
                 f"{', '.join(repr(r) for r in suite.rubrics)}")
    if not pending:
        return wanted(suite, episodes, rubric)
    raised: dict[str, list[str]] = {}
    for name in store.judgings(profile):
        _, r = split_judging(name)
        if r in suite.rubrics and rubric in (None, r):
            for ep_id, row in store.scores(name).items():
                if "error" in row:
                    raised.setdefault(ep_id, []).append(r)
    return {ep.ep_id: raised[ep.ep_id] for ep in episodes if ep.ep_id in raised}, {}


def scored(where: str, args: argparse.Namespace, suite: str, rows: list[dict], unknown: dict[str, int], services: Services) -> int:
    """`score`'s closing lines and exit code."""
    errors = sum("error" in r for r in rows)
    rubrics = sorted({r["rubric"] for r in rows if "rubric" in r})
    print(f"{where}, judges {args.judges}{', rubrics ' + ', '.join(rubrics) if rubrics else ''}: "
          f"{len({r['ep_id'] for r in rows})} episodes, {sum('metrics' in r for r in rows)} scored, "
          f"{errors} scorer errors; spent ${sum(services.llm.spent.values()):.4f}")
    for name, n in sorted(unknown.items()):
        print(f"  rubric {name!r}: listed by {n} episodes, and suite {suite} has no rubric of that name; nothing was scored under it")
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
    r.add_argument("--cast", help="[cast.<name>] in models.toml: the endpoint of every model seat that is not the subject's")
    r.add_argument("--driver", default="direct", choices=sorted(DRIVERS))
    r.add_argument("--split", default="dev", choices=["train", "dev", "test", LOCAL_DEV])
    r.add_argument("--held-out-test", action="store_true",
                   help="required with --split test; each such run is appended to $WA_DATA/test-split-uses.jsonl")
    r.add_argument("--where", help="key=value[,key=value]: task row fields, then payload fields; a comma not followed by key= stays in the value")
    r.add_argument("--limit", type=int, help="run at most this many tasks")
    r.add_argument("--from", dest="source", metavar="EP_ID@N",
                   help="play rollouts of a stored episode's turn checkpoint N instead of task rows (see --rollouts)")
    r.add_argument("--from-run", help="the run directory whose store holds --from's episode (default: --out's own)")
    r.add_argument("--rollouts", type=int, default=1,
                   help="with --from: how many forks of the checkpoint to play; they are one group, whose size is then fixed")
    r.add_argument("--round", default="0",
                   help="with --from: which pass over the checkpoint this is (a trainer's policy step); each round's "
                        "rollouts are a group of their own")
    r.add_argument("--out", help="run directory (resumed when it exists)")
    r.add_argument("--judges", default="eval", help="[judges.<profile>] in models.toml; `none` scores only what needs no model judge")
    r.add_argument("--no-score", action="store_true", help="play and save only; score later")
    r.add_argument("--concurrency", type=int, default=8, help="tasks in flight")
    r.add_argument("--max-tokens", type=int, default=16384, help="per model call, thinking included")

    st = sub.add_parser("situate", help="set a new task up on the files a closed episode ended with, and store it")
    where = st.add_mutually_exclusive_group(required=True)
    where.add_argument("--run", help="the run directory whose store the situation goes into")
    where.add_argument("--world", help="the library world the situation goes into; --at is then an episode of that world")
    st.add_argument("--at", required=True, metavar="EP_ID", help="the position: a closed episode's name")
    st.add_argument("--row", required=True, help="a JSON file: the task's suite, kind and payload")
    st.add_argument("--from-run", help="the run directory whose store holds --at's episode (default: --run's own)")
    st.add_argument("--judges", default="eval", help="[judges.<profile>] in models.toml, for what the suite's setup asks of the corpus")

    s = sub.add_parser("score", help="re-score a saved run's episodes (twins re-paired)")
    s.add_argument("--run", required=True)
    s.add_argument("--judges", default="eval", help="[judges.<profile>] in models.toml; `none` scores only what needs no model judge")
    s.add_argument("--rubric", help="one of the suite's rubrics; default: every rubric (score: each episode's listed "
                                             "ones; report: every one the profile has rows for)")
    s.add_argument("--where", help="key=value[,key=value]: the episode's row fields and its condition, as in report")
    s.add_argument("--pending", action="store_true",
                   help="only the rows saying the scorer raised (a judge that was down), of every rubric or the one named")
    s.add_argument("--concurrency", type=int, default=8)

    rp = sub.add_parser("report", help="metrics per group, with gates, refusals and baseline deltas")
    rp.add_argument("--run", required=True)
    rp.add_argument("--judges", default="eval", help="[judges.<profile>] in models.toml; `none` scores only what needs no model judge")
    rp.add_argument("--rubric", help="one of the suite's rubrics; default: every rubric (score: each episode's listed "
                                             "ones; report: every one the profile has rows for)")
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
    add_library_parser(sub)
    return p


from writing_assistant.cli.library import add_parser as add_library_parser  # noqa: E402
from writing_assistant.cli.library import cmd_library  # noqa: E402
from writing_assistant.cli.report import cmd_report  # noqa: E402  (needs the helpers above)
from writing_assistant.cli.view import cmd_view  # noqa: E402

COMMANDS = {"run": cmd_run, "situate": cmd_situate, "score": cmd_score, "report": cmd_report, "guardrails": cmd_guardrails,
            "collect": cmd_collect, "export": cmd_export, "sync": cmd_sync, "view": cmd_view, "library": cmd_library}


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(".env"))
    args = parser().parse_args(argv)
    try:
        result = COMMANDS[args.cmd](args)
        return asyncio.run(result) if asyncio.iscoroutine(result) else result
    except (Unreadable, Refused) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    sys.exit(main())
