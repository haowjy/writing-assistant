"""Frozen, task-specific reference selections for saved-output comparisons."""

import json
from pathlib import Path

from writing_agent.catalog import fingerprint


def load_matched_references(path: Path, scenarios: dict) -> dict:
    """Reject stale scenario assignments, changed texts, and training references."""
    manifest = json.loads(path.read_text())
    references = {r["id"]: r for r in manifest["references"]}
    if len(references) != len(manifest["references"]):
        raise ValueError("Duplicate reference IDs")
    for reference in references.values():
        if reference["sha256"] != fingerprint(reference["text"]):
            raise ValueError("Reference text hash mismatch")
        if reference["provenance"] != "human" or reference["role"] != "development":
            raise ValueError("Matched references require human development text")
    for key, scenario in scenarios.items():
        assignment = manifest["assignments"].get(key)
        if assignment is None or assignment["visible_hash"] != scenario["visible_hash"]:
            raise ValueError(f"Missing or stale reference assignment: {key}")
        if set(assignment["reference_ids"]) - references.keys():
            raise ValueError("Unknown matched reference ID")
        if scenario["visible"]["prose"] and not assignment["reference_ids"]:
            raise ValueError(f"Prose task needs an explicit reference selection: {key}")
    return manifest
