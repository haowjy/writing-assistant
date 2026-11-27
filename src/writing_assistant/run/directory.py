"""A run directory and what a run plays with: `services_for`, `open_run`, `record_usage`.

A run directory is `run.json`, `store.db`, `errors.jsonl` and `run.lock`. It belongs to one
run, whose **identity** is in run.json: suite, driver, endpoint and model, split, token
limit, the cast, and the world it plays (None for a run of task rows or of another run's
episodes). `open_run` is the one way a directory is claimed, by the command line, the
controller or anything else that plays, so what it checks cannot be skipped:

- a directory that holds a different run is refused;
- the `test` split is played only on an explicit say-so, and every use is logged
  (`TEST_USES`). The split is the identity's, which for a world is the world's.
"""

from __future__ import annotations

import fcntl
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from writing_assistant.corpus import Corpus
from writing_assistant.episode.segments import STORE_FILE, Store
from writing_assistant.episode.store import LOCAL_DEV, TEST_USES, append_jsonl
from writing_assistant.judges import judges_from
from writing_assistant.llm.client import LLMClient
from writing_assistant.run import code_version, now, resolved, run_info
from writing_assistant.suites import Services

LOCK_FILE = "run.lock"


class Refused(ValueError):
    """A run that may not start as asked. The message is for the person at the command line."""


def services_for(models: Path, profile: str) -> Services:
    """What a run or a judging pass reaches models, judges and the corpus through. `none` is
    the profile with no judge at all: what anything that only plays is given."""
    client = LLMClient(models)
    if profile not in client.config["judges"]:
        raise Refused(f"no [judges.{profile}] in {models}; it has {sorted(client.config['judges'])}")
    judges = judges_from(client, profile)
    return Services(llm=client, judges=judges, search=None, refs=None, corpus=Corpus(judges=judges))


def identity(services: Services, *, suite: str, driver: str, endpoint: str, split: str, max_tokens: int,
             cast: dict[str, str], world: str | None = None) -> dict[str, Any]:
    """A run's identity as run.json holds it. Refuses an endpoint models.toml does not have,
    and a `local_dev` run with a hosted endpoint in any seat: personal text may reach only
    an endpoint with `local = true`."""
    endpoints = services.llm.endpoints
    if endpoint not in endpoints:
        raise Refused(f"unknown endpoint {endpoint!r}; models.toml has {sorted(endpoints)}")
    hosted = [e for e in (endpoint, *cast.values()) if not endpoints[e].get("local")]
    if split == LOCAL_DEV and hosted:
        raise Refused(f"local_dev text may reach only an endpoint with `local = true`; {', '.join(map(repr, hosted))} is "
                      f"hosted. Local endpoints: {sorted(n for n, e in endpoints.items() if e.get('local'))}")
    return {"suite": suite, "driver": driver, "endpoint": endpoint, "model": endpoints[endpoint]["model"], "split": split,
            "max_tokens": max_tokens, "cast": resolved(endpoints, cast), "world": world}


@dataclass
class Opened:
    """A claimed run directory: the run's id, its store (open for writing), and whether no
    other run was alive in it when it was opened. `lock` is `run.lock`, held (shared, or
    exclusively when asked) for as long as this object lives; the kernel drops it when the
    process dies."""

    run_id: str
    store: Store
    alone: bool
    lock: IO[str]


def open_run(run_dir: Path, identity: dict[str, Any], *, held_out_test: bool = False, note: dict[str, Any] | None = None,
             exclusive: bool = False) -> Opened:
    """Claim `run_dir` for the run `identity` describes, or find it already that run's, and
    open its store. Of two first invocations one claims the identity and the other is held
    to it; a store this build does not write is refused with run.json unchanged. `note` is
    added to the line a `test` run leaves in the log of the held-out split's uses.

    `exclusive`: the directory is this run's alone for as long as the result lives. Another
    run alive in it is refused, and none can start. Without it several runs may share a
    directory (`alone` says whether one was there), each holding the lock shared."""
    if identity["split"] == "test" and not held_out_test:
        raise Refused(f"this run plays the held-out test split: pass --held-out-test to run it (logged in {TEST_USES})")
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_info(run_dir) as info:
        if not info:
            info |= identity | {"started": now(), "reported_apart": identity["driver"] == "direct"}
        elif (was := {k: info.get(k) for k in identity} | {"cast": info.get("cast", {})}) != identity:  # no cast before runs had one
            raise Refused(f"{run_dir} holds a different run: {was}; this one is {identity}")
        run_id = info.setdefault("id", uuid.uuid4().hex)  # a directory claimed before runs had one gets it now
        store = Store(run_dir / STORE_FILE)
    # Getting the lock exclusively first means no other run is alive in this directory.
    lock = (run_dir / LOCK_FILE).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        alone = True
    except BlockingIOError:
        alone = False
    if exclusive and not alone:
        store.db.close()
        lock.close()
        raise Refused(f"{run_dir}: another run is alive here; run this again when it ends")
    if identity["split"] == "test":  # after every refusal: a call that plays nothing used nothing
        append_jsonl(TEST_USES, [identity | {"at": now(), "run": str(run_dir.resolve()), "code": code_version()} | (note or {})])
    if not exclusive:  # the change to shared is not atomic (flock(2)), which is why a run that must be alone keeps it
        fcntl.flock(lock, fcntl.LOCK_SH)
    return Opened(run_id, store, alone, lock)


def record_usage(run_dir: Path, services: Services) -> None:
    """Add the USD the hosts reported, and the builds that answered, to run.json."""
    with run_info(run_dir) as info:
        spent = info.setdefault("spent_usd", {})
        for endpoint, usd in services.llm.spent.items():
            spent[endpoint] = round(spent.get(endpoint, 0.0) + usd, 6)
        served = info.setdefault("served", {})
        for endpoint, builds in services.llm.served.items():
            served[endpoint] = sorted(set(served.get(endpoint, [])) | builds)
