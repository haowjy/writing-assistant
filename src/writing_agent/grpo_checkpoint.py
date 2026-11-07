"""Integrity helpers for trainer checkpoints and exported adapters."""

import hashlib
import json

from writing_agent.catalog import save_json


def file_hashes(path, *, exclude=()):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name not in exclude
    }


def seal_directory(path, identity, kind, *, group_files=None):
    files = file_hashes(path, exclude=("complete.json",))
    marker = {"identity": identity, "kind": kind, "files": files}
    if kind == "trainer":
        marker["group_files"] = group_files or {}
    save_json(path / "complete.json", marker)


def verify_checkpoint(path, identity):
    try:
        marker = json.loads((path / "complete.json").read_text())
        required = {
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        if (
            marker["identity"] != identity
            or marker["kind"] != "trainer"
            or not required <= marker["files"].keys()
            or marker["files"] != file_hashes(path, exclude=("complete.json",))
        ):
            raise ValueError("Incomplete or changed trainer checkpoint")
        for relative, digest in marker["group_files"].items():
            item = path.parent / "groups" / relative
            if not item.resolve().is_relative_to((path.parent / "groups").resolve()):
                raise ValueError("Invalid checkpoint group path")
            if hashlib.sha256(item.read_bytes()).hexdigest() != digest:
                raise ValueError("Checkpoint rollout evidence changed")
        step = json.loads((path / "trainer_state.json").read_text())["global_step"]
        if type(step) is not int or step < 1:
            raise ValueError("Invalid checkpoint global step")
        return step
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing/truncated complete trainer checkpoint") from exc
