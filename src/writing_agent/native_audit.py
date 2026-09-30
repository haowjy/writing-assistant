"""Tokenizer-backed admission audit for native task-graph training batches.

The task-graph store and derives remain tokenizer-free. This adapter-side module
reconstructs native rendering from committed context and records the admission that
the trainer and offline inspector require.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from writing_agent.native_gemma import NativeGemmaRenderer
from writing_agent.native_protocol import parse_native_response
from writing_agent.task_graph import (
    EventV1,
    canonical_bytes,
    load_canonical_json,
    thaw,
    validate_hash,
)
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_errors import AdapterContractError, ProjectionError
from writing_agent.task_graph_gate import LineageGate, StoreArtifactReader
from writing_agent.task_graph_group_records import GroupDecisionV1, GroupMemberResultV1
from writing_agent.task_graph_native_contracts import NativeSamplingHistory
from writing_agent.task_graph_record_contracts import GroupSpecV1
from writing_agent.task_graph_records import (
    RuntimeManifestV2,
    TrainingAdmissionV1,
    WriterTurnV2,
    decode_runtime_manifest,
)
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_token_ledger import decode_u32_token_ids, encode_u32_token_ids
from writing_agent.task_graph_training_export import (
    TrainingBatchExportV1,
    TrainingExportError,
    export_training_batch,
)
from writing_agent.task_graph_training_records import TrainingBatchV1

AUDIT_VERSION = "gemma4-native-audit-v1"


class TrainingAuditError(AdapterContractError):
    """A persisted training admission refused one or more native members."""

    def __init__(self, admission: TrainingAdmissionV1) -> None:
        self.admission = admission
        super().__init__("native training audit refused one or more members")


def audit_training_batch(
    store: Any,
    spec: GroupSpecV1,
    decision: GroupDecisionV1,
    tokenizer: Any,
    *,
    adapter_hash_before: str,
    adapter_hash_after: str,
    tokenizer_root: Path | str | None = None,
) -> TrainingAdmissionV1:
    """Export, audit and persist one native batch and return its admission record.

    The group coordinator owns durable group receipts. A refusal is still returned;
    the caller must require an all-admitted result before exposing its batch to TRL.
    """
    reader = StoreArtifactReader(store)
    exported = export_training_batch(spec, decision, reader)
    batch_ref = _persist_export(store, exported)
    admission, _manifest = _derive_admission(
        store,
        spec,
        decision,
        exported,
        tokenizer,
        adapter_hash_before=adapter_hash_before,
        adapter_hash_after=adapter_hash_after,
        tokenizer_root=tokenizer_root,
    )
    if admission.batch_ref != batch_ref:
        raise TrainingExportError("batch_identity_mismatch", "persisted batch ref differs")
    return admission


def require_training_admission(admission: TrainingAdmissionV1) -> None:
    """Raise after a durable refusal; only an all-admitted record may reach a trainer."""
    if not isinstance(admission, TrainingAdmissionV1):
        raise TypeError("native training requires TrainingAdmissionV1")
    if any(member["status"] != "admitted" for member in admission.members):
        raise TrainingAuditError(admission)


def inspect_group_offline(store_root: Path | str, group_id: str) -> bytes:
    """Re-derive a batch and its tokenizer-backed admission without network access.

    The canonical report omits prompt, packet, context, token, and per-member data. Its
    bytes are returned and atomically written to ``groups/<group_id>/inspection.json``.
    """
    validate_hash(group_id)
    gate = LineageGate()
    store = TaskGraphStore(store_root, verifier=gate)
    group_dir = store.root / "groups" / group_id
    # GroupCoordinator's sealed spec receipt is the record body itself, not a ref wrapper.
    try:
        spec_body = load_canonical_json((group_dir / "spec.json").read_bytes())
    except OSError as exc:
        raise ProjectionError("offline group spec receipt is unavailable") from exc
    if not isinstance(spec_body, dict):
        raise ProjectionError("offline group spec receipt is malformed")
    spec = GroupSpecV1.from_dict(spec_body)
    if spec.group_id != group_id or store.get_artifact(spec.identity()) != spec.to_wire():
        raise ProjectionError("offline group spec differs from its sealed identity")

    admission_receipt = _read_receipt(
        group_dir / "training-admission.json", {"admission_ref": None}
    )
    admission_ref = admission_receipt["admission_ref"]
    stored_admission = TrainingAdmissionV1.from_dict(store.get_artifact(admission_ref))
    if stored_admission.group_id != group_id:
        raise ProjectionError("training admission belongs to another group")
    batch_receipt = _read_receipt(group_dir / "training-batch.json", {"batch_ref": None})
    if batch_receipt["batch_ref"] != stored_admission.batch_ref:
        raise ProjectionError("training batch receipt differs from admission")
    decision = GroupDecisionV1.from_dict(store.get_artifact(stored_admission.decision_ref))
    if decision.group_id != group_id:
        raise ProjectionError("training decision belongs to another group")
    manifest = _native_manifest(store, spec)
    tokenizer, tokenizer_root = _load_pinned_tokenizer(manifest)
    exported = export_training_batch(spec, decision, StoreArtifactReader(store))
    expected_batch_ref = exported.batch.identity()
    stored_batch = TrainingBatchV1.from_dict(store.get_artifact(stored_admission.batch_ref))
    if (
        stored_admission.batch_ref != expected_batch_ref
        or stored_batch.to_wire() != exported.batch.to_wire()
    ):
        raise ProjectionError("offline training batch differs from its re-derived export")
    for artifact in exported.artifacts:
        if StoreArtifactReader(store).bytes_artifact(artifact.ref) != artifact.value:
            raise ProjectionError("offline training export byte artifact differs")

    expected_admission, _ = _derive_admission(
        store,
        spec,
        decision,
        exported,
        tokenizer,
        adapter_hash_before=stored_admission.adapter_hash_before,
        adapter_hash_after=stored_admission.adapter_hash_after,
        tokenizer_root=tokenizer_root,
    )
    if (
        expected_admission.to_wire() != stored_admission.to_wire()
        or expected_admission.identity() != admission_ref
    ):
        raise ProjectionError("offline training admission differs from its re-derived audit")
    require_training_admission(stored_admission)

    report = {
        "schema": 1,
        "group_id": group_id,
        "decision_ref": stored_admission.decision_ref,
        "batch_ref": stored_admission.batch_ref,
        "admission_ref": admission_ref,
        "status": "admitted",
        "member_count": len(stored_admission.members),
        "admitted_count": sum(m["status"] == "admitted" for m in stored_admission.members),
        "refused_count": sum(m["status"] == "refused" for m in stored_admission.members),
        "mismatch_count": 0,
    }
    encoded = canonical_bytes(report)
    _write_report(group_dir / "inspection.json", encoded)
    return encoded


def _derive_admission(
    store: Any,
    spec: GroupSpecV1,
    decision: GroupDecisionV1,
    exported: TrainingBatchExportV1,
    tokenizer: Any,
    *,
    adapter_hash_before: str,
    adapter_hash_after: str,
    tokenizer_root: Path | str | None,
) -> tuple[TrainingAdmissionV1, RuntimeManifestV2]:
    manifest = _native_manifest(store, spec)
    renderer = None
    try:
        renderer = NativeGemmaRenderer(tokenizer, manifest.renderer)
    except Exception:
        # The failure is written as a check-1 refusal for every member.
        pass
    tokenizer_ok = _tokenizer_files_match(tokenizer, manifest, tokenizer_root)
    try:
        validate_hash(adapter_hash_before)
        validate_hash(adapter_hash_after)
    except (TypeError, ValueError):
        adapter_hash_before = "0" * 64
        adapter_hash_after = "0" * 64

    expected_members = tuple(member.member_id for member in spec.members)
    batch = exported.batch
    result_by_member = _results_by_member(store, spec, decision)
    statuses: list[dict[str, Any]] = []
    reader = StoreArtifactReader(store)
    gate = store.verifier if isinstance(store.verifier, LineageGate) else LineageGate()
    for member_spec, batch_member in zip(spec.members, batch.members, strict=True):
        failure = None
        result = result_by_member[member_spec.member_id]
        try:
            if renderer is None:
                raise _AuditCheckFailure("renderer_initial_context")
            turns, root_context = _member_turns(
                store,
                reader,
                gate,
                spec,
                result,
                member_spec.member_id,
            )
            if not turns:
                raise _AuditCheckFailure("renderer_initial_context")

            first = turns[0]
            _prompt, expected_input = renderer.render_initial(
                root_context.messages, thaw(root_context.tools)
            )
            if first.input_ids != expected_input:
                raise _AuditCheckFailure("renderer_initial_context")

            for previous, current in zip(turns, turns[1:], strict=False):
                try:
                    current_context = reader.context(current.turn.context_revision_ref)
                    history = NativeSamplingHistory(
                        previous.turn, previous.input_ids, previous.generated_ids
                    )
                    suffix = renderer.external_suffix(history, current_context.messages)
                except Exception as exc:
                    raise _AuditCheckFailure("external_suffix_and_context_limit") from exc
                if current.input_ids != previous.input_ids + previous.generated_ids + suffix:
                    raise _AuditCheckFailure("external_suffix_and_context_limit")
                if current.turn.termination["kind"] == "context_limit":
                    cap = batch.max_context_tokens
                    if (
                        current.generated_ids
                        or cap is None
                        or len(previous.input_ids + previous.generated_ids + suffix) < cap
                    ):
                        raise _AuditCheckFailure("external_suffix_and_context_limit")

            for evidence in turns:
                if not evidence.generated_ids:
                    if (
                        evidence.turn.raw_output_ref is not None
                        or evidence.turn.native_parse_failed is not None
                    ):
                        raise _AuditCheckFailure("raw_output_and_message")
                    continue
                raw_output = _raw_output(store, evidence.turn.raw_output_ref)
                decoded = tokenizer.decode(evidence.generated_ids, skip_special_tokens=False)
                if not isinstance(decoded, str) or raw_output != decoded.encode("utf-8"):
                    raise _AuditCheckFailure("raw_output_and_message")
                prefix = tokenizer.decode(evidence.input_ids, skip_special_tokens=False)
                parsed = parse_native_response(
                    tokenizer,
                    decoded,
                    prefix=prefix,
                    action_id=evidence.turn.action_id,
                    termination=evidence.turn.termination,
                )
                expected_claim = True if parsed.failed else None
                if evidence.turn.native_parse_failed is not expected_claim:
                    raise _AuditCheckFailure("raw_output_and_message")
                if parsed.failed and (
                    evidence.turn.message.content != decoded or evidence.turn.message.calls
                ):
                    raise _AuditCheckFailure("raw_output_and_message")
                if (
                    intake_message(dict(parsed.message)).to_wire()
                    != evidence.turn.message.to_wire()
                ):
                    raise _AuditCheckFailure("raw_output_and_message")

            expected_policy = spec.policy["behavior_policy_ref"]
            if (
                adapter_hash_before != expected_policy
                or adapter_hash_after != expected_policy
                or not turns
                or any(
                    evidence.turn.sampling_pins["behavior_policy_ref"] != expected_policy
                    or evidence.turn.sampling_pins["manifest_ref"] != spec.policy["adapter_ref"]
                    or evidence.turn.sampling_pins["decoding_ref"] != spec.policy["decoding_ref"]
                    or evidence.turn.sampling_pins["renderer_ref"] != manifest.renderer.identity()
                    or evidence.turn.sampling_pins["seed"] != member_spec.writer_seed
                    for evidence in turns
                )
            ):
                raise _AuditCheckFailure("policy_and_adapter_binding")

            if not tokenizer_ok:
                raise _AuditCheckFailure("tokenizer_files")
            if not _batch_member_matches(reader, batch, batch_member, turns):
                raise _AuditCheckFailure("exported_token_layout")
        except _AuditCheckFailure as exc:
            failure = exc.check
        except Exception:
            # Any unverifiable host or adapter evidence fails closed as infrastructure.
            failure = "audit_unexpected_exception"
        statuses.append(
            {
                "member_id": member_spec.member_id,
                "status": "refused" if failure else "admitted",
                "failed_check": failure,
            }
        )

    record = TrainingAdmissionV1(
        group_id=spec.group_id,
        decision_ref=decision.identity(),
        batch_ref=batch.identity(),
        audit_version=AUDIT_VERSION,
        renderer_ref=manifest.renderer.identity(),
        tokenizer_descriptor_ref=manifest.tokenizer.identity(),
        adapter_hash_before=adapter_hash_before,
        adapter_hash_after=adapter_hash_after,
        members=statuses,
    )
    if tuple(item["member_id"] for item in record.members) != expected_members:
        raise AssertionError("audit member order differs from sealed group order")
    return record, manifest


class _AuditCheckFailure(Exception):
    def __init__(self, check: str) -> None:
        self.check = check


class _TurnEvidence:
    __slots__ = ("ref", "turn", "input_ids", "generated_ids")

    def __init__(
        self,
        ref: str,
        turn: WriterTurnV2,
        input_ids: tuple[int, ...],
        generated_ids: tuple[int, ...],
    ) -> None:
        self.ref = ref
        self.turn = turn
        self.input_ids = input_ids
        self.generated_ids = generated_ids


def _member_turns(store, reader, gate, spec, result, member_id):
    if result.final_checkpoint_id is None:
        raise _AuditCheckFailure("renderer_initial_context")
    view = gate.view(store, result.final_checkpoint_id)
    if view.group != spec or view.state.position["lineage_id"] != member_id:
        raise _AuditCheckFailure("policy_and_adapter_binding")
    checkpoint = reader.checkpoint(result.final_checkpoint_id)
    ordinal = next(m.ordinal for m in spec.members if m.member_id == member_id)
    start_path = store.root / "groups" / spec.group_id / f"start-{ordinal}.json"
    start_receipt = load_canonical_json(start_path.read_bytes())
    if (
        not isinstance(start_receipt, Mapping)
        or start_receipt.get("member_id") != member_id
        or start_receipt.get("parent_checkpoint_id") != spec.environment["entry_checkpoint_id"]
    ):
        raise _AuditCheckFailure("renderer_initial_context")
    start_checkpoint = reader.checkpoint(start_receipt.get("start_checkpoint_id"))
    if start_checkpoint.state.position["lineage_id"] != member_id:
        raise _AuditCheckFailure("renderer_initial_context")
    root_context = reader.context(start_checkpoint.state.context_ref)

    refs: list[str] = []
    event_ref = checkpoint.event_head
    seen: set[str] = set()
    while event_ref is not None:
        if event_ref in seen:
            raise _AuditCheckFailure("external_suffix_and_context_limit")
        seen.add(event_ref)
        event = EventV1.from_dict(reader.artifact(event_ref, domain="event"))
        if event.lineage_id != member_id:
            break
        if event.kind == "budget_charged":
            raise _AuditCheckFailure("external_suffix_and_context_limit")
        if event.kind == "writer_action":
            body = reader.artifact(event.payload_ref)
            if not isinstance(body, Mapping) or body.get("record_type") != WriterTurnV2.RECORD_TYPE:
                raise _AuditCheckFailure("policy_and_adapter_binding")
            turn = WriterTurnV2.from_dict(body)
            refs.append(event.payload_ref)
        event_ref = event.previous
    ordered = []
    for ref in reversed(refs):
        turn = WriterTurnV2.from_dict(reader.artifact(ref))
        ordered.append(
            _TurnEvidence(
                ref,
                turn,
                decode_u32_token_ids(
                    reader.bytes_artifact(turn.input_token_ids_ref), turn.input_token_count
                ),
                decode_u32_token_ids(
                    reader.bytes_artifact(turn.generated_token_ids_ref),
                    turn.generated_token_count,
                ),
            )
        )
    return tuple(ordered), root_context


def _batch_member_matches(reader, batch, member, turns) -> bool:
    if not turns:
        return False

    generated_turns = [index for index, turn in enumerate(turns) if turn.generated_ids]
    if not generated_turns:
        return False

    last_generated = generated_turns[-1]
    exported_turns = turns[: last_generated + 1]
    trailing = turns[last_generated + 1 :]
    trailing_ref = None
    if trailing:
        if (
            len(trailing) != 1
            or trailing[0].generated_ids
            or trailing[0].turn.termination["kind"] != "context_limit"
        ):
            return False
        trailing_ref = trailing[0].ref
    if member.get("trailing_context_limit_turn_ref") != trailing_ref:
        return False

    prompt_ids = exported_turns[0].input_ids
    completion_ids: list[int] = []
    env_mask = bytearray()
    previous = None
    for evidence in exported_turns:
        if previous is not None:
            prior_boundary = previous.input_ids + previous.generated_ids
            if evidence.input_ids[: len(prior_boundary)] != prior_boundary:
                return False
            suffix = evidence.input_ids[len(prior_boundary) :]
            completion_ids.extend(suffix)
            env_mask.extend(b"\0" * len(suffix))

        completion_ids.extend(evidence.generated_ids)
        env_mask.extend(b"\1" * len(evidence.generated_ids))
        previous = evidence

    sequence_length = len(prompt_ids) + len(completion_ids)
    if batch.max_context_tokens is not None and sequence_length > batch.max_context_tokens:
        return False
    if (
        reader.bytes_artifact(member["prompt_ids_ref"]) != encode_u32_token_ids(prompt_ids)
        or reader.bytes_artifact(member["completion_ids_ref"])
        != encode_u32_token_ids(tuple(completion_ids))
        or reader.bytes_artifact(member["env_mask_ref"]) != bytes(env_mask)
    ):
        return False
    return True


def _results_by_member(store, spec, decision) -> dict[str, GroupMemberResultV1]:
    if decision.group_id != spec.group_id or decision.status not in {"ready", "tie"}:
        raise TrainingExportError("group_not_exportable", "group decision is not settled")
    if len(decision.member_result_refs) != len(spec.members):
        raise TrainingExportError("incomplete_group", "group decision has incomplete results")
    results = {}
    for member, ref in zip(spec.members, decision.member_result_refs, strict=True):
        if ref is None:
            raise TrainingExportError("incomplete_group", "group decision has a missing result")
        result = GroupMemberResultV1.from_dict(store.get_artifact(ref))
        if result.group_id != spec.group_id or result.member_id != member.member_id:
            raise TrainingExportError("misbound_group_result", "decision result is misbound")
        results[member.member_id] = result
        receipt_path = store.root / "groups" / spec.group_id / f"result-{member.ordinal}.json"
        receipt = load_canonical_json(receipt_path.read_bytes())
        if receipt != {"result_ref": ref}:
            raise TrainingExportError("group_receipt_mismatch", "group result receipt differs")
    return results


def _native_manifest(store, spec: GroupSpecV1) -> RuntimeManifestV2:
    manifest = decode_runtime_manifest(store.get_artifact(spec.policy["adapter_ref"]))
    if not isinstance(manifest, RuntimeManifestV2):
        raise TrainingExportError("v2_manifest_required", "native audit requires RuntimeManifestV2")
    if (
        spec.training_mode != "native"
        or spec.policy["tokenizer_ref"] != manifest.tokenizer.identity()
        or spec.policy["decoding_ref"] != manifest.decoding.identity()
        or spec.policy["template_ref"] != manifest.renderer.template_ref
    ):
        raise TrainingExportError(
            "manifest_pin_mismatch", "native manifest differs from group pins"
        )
    return manifest


def _persist_export(store: Any, exported: TrainingBatchExportV1) -> str:
    for artifact in exported.artifacts:
        if store.persist_artifact(artifact) != artifact.ref:
            raise TrainingExportError("export_artifact_mismatch", "export artifact ref differs")
    ref = store.put_artifact(exported.batch.to_wire())
    if ref != exported.batch.identity():
        raise TrainingExportError("batch_identity_mismatch", "batch ref differs from identity")
    return ref


def _raw_output(store: Any, ref: str | None) -> bytes:
    if ref is None:
        raise _AuditCheckFailure("raw_output_and_message")
    value = store.get_artifact(ref)
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise _AuditCheckFailure("raw_output_and_message")


def _tokenizer_files_match(tokenizer: Any, manifest: RuntimeManifestV2, root=None) -> bool:
    try:
        base = Path(root) if root is not None else Path(tokenizer.name_or_path)
        if not base.is_dir():
            return False
        observed = {}
        for name in manifest.tokenizer.files_sha256:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            path = base / relative
            observed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return observed == dict(manifest.tokenizer.files_sha256)
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _load_pinned_tokenizer(manifest: RuntimeManifestV2):
    """Load only an already-cached snapshot; all Hugging Face access is local-only."""
    from transformers import AutoTokenizer

    descriptor = manifest.tokenizer
    model_path = Path(descriptor.model_id)
    if model_path.is_dir():
        snapshot = model_path
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=descriptor.model_id,
                revision=descriptor.revision,
                local_files_only=True,
            )
        )
    # Keep file-integrity mismatches in the durable audit path. The offline inspector
    # must be able to load the local tokenizer and re-derive `tokenizer_files` refusal,
    # rather than failing before it can reproduce TrainingAdmissionV1.
    for name in descriptor.files_sha256:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise TrainingExportError("tokenizer_file_path", "tokenizer descriptor path is unsafe")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    return tokenizer, snapshot


def _read_receipt(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = load_canonical_json(path.read_bytes())
    except OSError as exc:
        raise ProjectionError(f"offline receipt unavailable: {path.name}") from exc
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ProjectionError(f"offline receipt schema differs: {path.name}")
    for key, required in expected.items():
        if required is not None and value.get(key) != required:
            raise ProjectionError(f"offline receipt binding differs: {path.name}.{key}")
    return value


def _write_report(path: Path, encoded: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".inspection.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AUDIT_VERSION",
    "TrainingAuditError",
    "audit_training_batch",
    "inspect_group_offline",
    "require_training_admission",
]
