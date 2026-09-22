"""Single inherited process lease with advisory progress; no elapsed/stall cutoff."""

import fcntl
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from writing_agent.catalog import save_json


@contextmanager
def writer_lease(run_dir):
    """Child inherits this open file description, keeping ownership if parent dies."""
    with (Path(run_dir) / "writer.lock").open("a+") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this full48 run") from exc
        yield guard.fileno()
        # Closing, not LOCK_UN: a surviving child must retain the inherited lease.


def verify_lease(run_dir, descriptor):
    """Require the inherited descriptor for this run, already holding its lock."""
    actual = os.fstat(descriptor)
    expected = (Path(run_dir) / "writer.lock").stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Worker lease does not belong to run")
    # Our own description must own the lock, not merely observe another writer.
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError("Worker descriptor is not the owning lease") from exc
    # A second open description must conflict with our inherited exclusive lease.
    with (Path(run_dir) / "writer.lock").open("a") as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    raise ValueError("Worker requires an already locked inherited lease")


def process_progress(pid, run_dir):
    """Linux /proc CPU/RSS/IO plus durable artifact progress; never infer a stall."""
    sample = {"time": time.time(), "pid": pid}
    for name in ("status", "stat", "io"):
        try:
            sample[name] = (Path("/proc") / str(pid) / name).read_text()
        except OSError:
            sample[name] = None
    trainer = Path(run_dir) / "trainer"
    sample["groups_started"] = len(list((trainer / "groups").glob("*/started.json")))
    sample["attempts_with_results"] = len(
        list((trainer / "groups").glob("*/attempt-*/result.json"))
    )
    sample["checkpoint_markers"] = [
        p.parent.name for p in trainer.glob("checkpoint-*/complete.json")
    ]
    return sample


def supervise(run_dir, command, *, admit, interval=15.0):
    """Launch one owned child, retain liveness even during quiet generation/training.

    Only a user/process interrupt signals the owned child. Poll intervals are not
    deadlines. Child exit/failure never triggers a retry or an unrelated signal.
    """
    run_dir = Path(run_dir).resolve()
    if interval <= 0:
        raise ValueError("Progress interval must be positive")
    with writer_lease(run_dir) as descriptor:
        admission = admit()  # Recheck while holding ownership, before spawning.
        destination = run_dir / "supervision" / uuid4().hex
        destination.mkdir(parents=True)
        started = time.monotonic()
        record = {
            "status": "starting",
            "elapsed_cutoff": None,
            "started_at": time.time(),
            "command": command,
            "admission": admission,
            "supervisor_pid": os.getpid(),
        }
        save_json(destination / "status.json", record)
        process = None
        try:
            with (
                (destination / "worker.log").open("w") as log,
                (destination / "progress.jsonl").open("w") as progress,
            ):
                process = subprocess.Popen(
                    [*command, "--_lease-fd", str(descriptor)],
                    pass_fds=(descriptor,),
                    start_new_session=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                record.update(status="running", child_pid=process.pid)
                save_json(destination / "status.json", record)
                while True:
                    progress.write(json.dumps(process_progress(process.pid, run_dir)) + "\n")
                    progress.flush()
                    try:
                        returncode = process.wait(timeout=interval)
                    except subprocess.TimeoutExpired:
                        continue
                    break
                record.update(
                    status="completed" if returncode == 0 else "failed", returncode=returncode
                )
                if returncode:
                    raise RuntimeError(f"Full48 worker exited {returncode}; inspect {destination}")
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if process is not None and process.poll() is None:
                # Exclusive new session is this launch's process group, never global pkill.
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()  # No wall-clock escalation, even on requested interruption.
            raise
        finally:
            record.update(ended_at=time.time(), elapsed_seconds=time.monotonic() - started)
            save_json(destination / "status.json", record)
        return record
