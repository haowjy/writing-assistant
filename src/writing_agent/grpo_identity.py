"""Non-mutating admission and base identity checks for bounded GRPO experiments."""

import copy
import hashlib
import json

from writing_agent.catalog import fingerprint, overlap_audit, validate_catalog


def admission_identity(tasks, admission):
    """Validate against the supplied compiled release/catalog, or label fixtures explicitly.

    Catalog provenance is caller supplied, as in scenario compilation. This validates
    its internal hashes, roles and connected lineage, not the truth of its provenance.
    """
    if not isinstance(admission, dict):
        raise ValueError("Explicit production or engineered-fixture admission is required")
    if admission.get("mode") == "engineered-fixture":
        if set(admission) != {"mode", "label"} or not str(admission.get("label", "")).strip():
            raise ValueError("Engineered fixture admission requires a nonempty label only")
        return {**admission, "selected_task_ids": [t["id"] for t in tasks]}
    if admission.get("mode") != "production":
        raise ValueError("Unknown training admission mode")
    try:
        catalog = admission["catalog"]
        manifest = admission["release_manifest"]
        digest = fingerprint(catalog)
        if digest != admission["catalog_hash"] or digest != manifest["catalog_hash"]:
            raise ValueError("Release catalog hash mismatch")
        groups = validate_catalog(catalog)
        overlaps = overlap_audit(catalog)
        if any(pair["crosses_role"] for pair in overlaps):
            raise ValueError("Near-duplicate source lineage crosses research roles")
        sources = {s["id"]: s for s in catalog}
        excluded = set(admission["excluded_source_groups"])
        blocked = set()
        for source in catalog:
            identities = {source["id"], source["work_id"], groups[source["id"]]}
            identities.update(source.get(k) for k in ("author_id", "series_id"))
            if identities & excluded:
                blocked.add(groups[source["id"]])
            if "text" in source and fingerprint(source["text"]) != source.get(
                "text_sha256", source["sha256"]
            ):
                raise ValueError("Source content hash mismatch")
        metadata = {m["id"]: m for m in manifest["scenarios"]}
        if len(metadata) != len(manifest["scenarios"]):
            raise ValueError("Duplicate release task IDs")
        for task in tasks:
            recorded = metadata.get(task["id"])
            actual = {k: v for k, v in task.items() if k not in {"visible", "labels"}}
            if actual != recorded or task["role"] != "train":
                raise ValueError("Training task metadata differs from release")
            if (
                fingerprint(task["visible"]) != task["visible_hash"]
                or fingerprint(task["labels"]) != task["labels_hash"]
            ):
                raise ValueError("Training task content differs from release")
            ids = task["source_ids"]
            if not ids or len(set(ids)) != len(ids) or set(ids) - sources.keys():
                raise ValueError("Unknown or missing training source identity")
            expected = sorted({groups[key] for key in ids})
            if expected != task["source_groups"]:
                raise ValueError("Training task source-group metadata mismatch")
            if any(sources[key]["role"] != "train" for key in ids) or blocked & set(expected):
                raise ValueError("Held-out source or connected lineage in training task")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed production admission/catalog/task metadata") from exc
    return {
        **copy.deepcopy(admission),
        "excluded_source_groups": sorted(excluded),
        "connected_groups": groups,
        "overlap_audit": overlaps,
        "selected_task_ids": [t["id"] for t in tasks],
    }


def base_tensor_identity(model):
    """Stream actual named parameters AND buffers, including nonpersistent buffers.

    Reads bounded CPU byte chunks without changing tensors, device, dtype or RNG.
    Dense contiguous tensors are required to avoid a hidden full-model copy. Alias
    names are retained; metadata and raw bytes are length-framed in sorted order.
    """
    import torch

    digest = hashlib.sha256()
    count = 0
    tensors = [
        *(
            ("parameter", name, value)
            for name, value in model.named_parameters(remove_duplicate=False)
        ),
        *(("buffer", name, value) for name, value in model.named_buffers(remove_duplicate=False)),
    ]
    for kind, name, tensor in sorted(tensors, key=lambda item: (item[0], item[1])):
        if (
            tensor.device.type == "meta"
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
        ):
            raise ValueError("Caller base tensors must be materialized, dense and contiguous")
        header = json.dumps(
            [kind, name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        data = tensor.detach().reshape(-1).view(torch.uint8)
        digest.update(data.numel().to_bytes(8, "big"))
        for offset in range(0, data.numel(), 1024 * 1024):
            digest.update(data[offset : offset + 1024 * 1024].cpu().numpy().tobytes())
        count += 1
    if not count:
        raise ValueError("Caller base model has no tensors")
    return {
        "trust": "actual-caller-base-tensors-v1",
        "sha256": digest.hexdigest(),
        "named_tensors": count,
    }
