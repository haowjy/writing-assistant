"""Deterministic, resumable offline GRPO group orchestration; no model or optimizer."""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

from writing_agent.task_graph import (
    EventV1,
    MessageV1,
    canonical_bytes,
    load_canonical_json,
    tree_hash,
    validate_hash,
)
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_group_contract import (
    POLICY_FIELDS,
    GroupAdvantageV1,
    GroupDecisionV1,
    GroupError,
    GroupMemberResultV1,
    GroupMemberSpecV1,
    GroupSegmentCreditV1,
    GroupSpecV1,
    _environment,
    _fraction,
    _hash,
    _required_policy,
    _seed,
)
from writing_agent.task_graph_projection import project_writer_context
from writing_agent.task_graph_sampling import (
    CURRENT_ELIGIBILITY,
    ProjectionError,
    SamplingEvidenceV1,
    bind_group_sampling_claims,
)
from writing_agent.task_graph_store import RuntimeHandle, TaskGraphStore


class GroupCoordinatorV1:
    """A serializable fake runner's admission, start, collection and finalization API."""

    def __init__(self, store: TaskGraphStore, workers_root: Path | str):
        self.store = store
        self.workers_root = Path(workers_root).resolve()
        self.workers_root.mkdir(parents=True, exist_ok=True)
        self.groups_root = store.root / "groups"
        self.groups_root.mkdir(mode=0o700, exist_ok=True)

    @contextmanager
    def _locked(self, group_id: str):
        directory = self.groups_root / group_id
        directory.mkdir(mode=0o700, exist_ok=True)
        with (directory / ".lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield directory
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def seal(
        self,
        entry_checkpoint_id: str,
        *,
        policy: dict[str, str],
        group_seed: int,
        group_sequence: int,
        member_count: int,
        runner_mode: str = "real",
    ) -> GroupSpecV1:
        environment = _environment(self.store, entry_checkpoint_id)
        rendering = dict(
            self.store.load_context(
                self.store.load_checkpoint(entry_checkpoint_id).state.context_ref
            ).rendering
        )
        policy = _required_policy(policy, rendering)
        for field in POLICY_FIELDS - {"rng_derivation_version"}:
            self.store.get_artifact(policy[field])
        ContextPolicyV1.from_dict(self.store.get_artifact(policy["context_policy_ref"]))
        if type(member_count) is not int or not 2 <= member_count <= 64:
            raise GroupError("group size must be 2..64")
        if type(group_sequence) is not int or group_sequence < 0:
            raise GroupError("group sequence must be nonnegative")
        group_id = _hash(
            [
                "GroupIdV1",
                group_sequence,
                environment,
                policy,
                group_seed,
                runner_mode,
                member_count,
            ]
        )
        members = tuple(
            GroupMemberSpecV1(
                member_id=f"grp-{group_id[:24]}-{ordinal:02d}",
                ordinal=ordinal,
                writer_seed=_seed(group_seed, "writer", ordinal),
                environment_seed=_seed(group_seed, "environment"),
            )
            for ordinal in range(member_count)
        )
        spec = GroupSpecV1(
            group_id=group_id,
            group_sequence=group_sequence,
            group_seed=group_seed,
            runner_mode=runner_mode,
            environment=environment,
            policy=policy,
            members=members,
        )
        with self._locked(group_id) as directory:
            self._receipt(directory / "spec.json", spec.to_dict())
        return spec

    @staticmethod
    def _receipt(path: Path, body: dict[str, Any]) -> None:
        data = canonical_bytes(body)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise GroupError(f"conflicting immutable receipt: {path.name}") from None
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.unlink(temporary)

    def resume(self, group_id: str) -> GroupSpecV1:
        validate_hash(group_id)
        path = self.groups_root / group_id / "spec.json"
        spec = GroupSpecV1.from_dict(load_canonical_json(path.read_bytes()))
        if spec.group_id != group_id:
            raise GroupError("group directory misbinds spec")
        for field in POLICY_FIELDS - {"rng_derivation_version"}:
            self.store.get_artifact(spec.policy[field])
        ContextPolicyV1.from_dict(self.store.get_artifact(spec.policy["context_policy_ref"]))
        if _environment(self.store, spec.environment["entry_checkpoint_id"]) != spec.environment:
            raise GroupError("sealed entry contract drifted")
        return spec

    def assert_start_contract(
        self, spec: GroupSpecV1, checkpoint_id: str, policy: dict[str, str]
    ) -> None:
        candidate = _environment(self.store, checkpoint_id)
        if canonical_bytes(candidate) != canonical_bytes(spec.environment):
            raise GroupError("member entry differs from full sealed environment contract")
        rendering = dict(
            self.store.load_context(
                self.store.load_checkpoint(checkpoint_id).state.context_ref
            ).rendering
        )
        if canonical_bytes(_required_policy(policy, rendering)) != canonical_bytes(spec.policy):
            raise GroupError("member policy contract drifted")

    def start(self, spec: GroupSpecV1, ordinal: int, *, policy: dict[str, str]) -> RuntimeHandle:
        """Idempotently publish one fresh branch and restore its private workspace."""
        if type(ordinal) is not int or not 0 <= ordinal < len(spec.members):
            raise GroupError("invalid member ordinal")
        self.resume(spec.group_id)
        member = spec.members[ordinal]
        self.assert_start_contract(spec, spec.environment["entry_checkpoint_id"], policy)
        parent_id = spec.environment["entry_checkpoint_id"]
        parent = self.store.load_checkpoint(parent_id)
        parent_bytes = (self.store.root / "checkpoints" / f"{parent_id}.json").read_bytes()
        seed_ref = self.store.put_artifact(
            {
                "record_type": "GroupMemberSeedsV1",
                "group_id": spec.group_id,
                "member_id": member.member_id,
                "derivation": member.seed_provenance,
                "writer_seed": member.writer_seed,
                "environment_seed": member.environment_seed,
                "parent_rng_ref": parent.state.rng_ref,
            }
        )
        position = {
            **parent.state.to_dict()["position"],
            "lineage_id": member.member_id,
            "start_checkpoint": parent_id,
        }
        effect = {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": parent.state.identity(),
            "file_delta": {},
            "set": {"position": position, "rng_ref": seed_ref},
            "history_set": {"branch_base": parent.event_head},
        }
        effect_ref = self.store.put_artifact(effect)
        event = EventV1(
            previous=parent.event_head,
            seq=parent.state.history["seq"] + 1,
            lineage_id=member.member_id,
            rollout_id=member.member_id,
            node_visit_id=parent.state.position["visit_id"],
            kind="rollout_started",
            actor="environment",
            audience=("controller", "trainer"),
            payload_ref=effect_ref,
            versions_ref=parent.state.versions_ref,
            provenance_ref=parent.state.provenance_ref,
        )
        next_state = self.store._apply_recorded_effect_body(parent.state, event, effect)
        commit = self.store.branch(
            parent_id, member.member_id, (event,), next_state, artifact_refs=(effect_ref, seed_ref)
        )
        child_id = self.store.load_commit(commit).checkpoint
        if (self.store.root / "checkpoints" / f"{parent_id}.json").read_bytes() != parent_bytes:
            raise GroupError("sealed parent checkpoint changed during branch")
        if self.store.load_checkpoint(child_id).state.files != parent.state.files:
            raise GroupError("member branch did not restore identical files")
        with self._locked(spec.group_id) as directory:
            self._receipt(
                directory / f"start-{ordinal}.json",
                {
                    "member_id": member.member_id,
                    "parent_checkpoint_id": parent_id,
                    "start_checkpoint_id": child_id,
                    "commit_id": commit,
                },
            )
        destination = self.workers_root / member.member_id
        if destination.exists():
            # A resume materialization is disposable, but a caller must not reuse a
            # possibly modified workspace. The caller removes it explicitly first.
            raise GroupError("member workspace exists; remove disposable copy before restoring")
        runtime = self.store.restore(child_id, destination)
        if (
            tree_hash(
                {
                    p.relative_to(destination).as_posix(): p.read_text(encoding="utf-8")
                    for p in destination.rglob("*")
                    if p.is_file()
                }
            )
            != parent.state.tree_hash
        ):
            raise GroupError("materialized member workspace differs from sealed parent")
        return runtime

    def restore_member(self, spec: GroupSpecV1, ordinal: int, destination: Path) -> RuntimeHandle:
        if type(ordinal) is not int or not 0 <= ordinal < len(spec.members):
            raise GroupError("invalid member ordinal")
        receipt = self._start_receipt(spec, ordinal)
        head = self.store.read_head(spec.members[ordinal].member_id)
        if head is None:
            raise GroupError("member branch has no authoritative head")
        checkpoint_id = self.store.load_commit(head).checkpoint
        # The current head may be a partially completed rollout, but must still
        # descend from the exact sealed member start.
        cursor = self.store.load_checkpoint(checkpoint_id)
        while cursor.identity() != receipt["start_checkpoint_id"]:
            if not cursor.parents:
                raise GroupError("member head escaped its sealed branch")
            cursor = self.store.load_checkpoint(cursor.parents[0])
        return self.store.restore(checkpoint_id, destination)

    def _start_receipt(self, spec: GroupSpecV1, ordinal: int) -> dict:
        body = load_canonical_json(
            (self.groups_root / spec.group_id / f"start-{ordinal}.json").read_bytes()
        )
        if (
            not isinstance(body, dict)
            or set(body)
            != {"member_id", "parent_checkpoint_id", "start_checkpoint_id", "commit_id"}
            or not all(isinstance(body[key], str) for key in body)
            or body["member_id"] != spec.members[ordinal].member_id
            or body["parent_checkpoint_id"] != spec.environment["entry_checkpoint_id"]
        ):
            raise GroupError("member start receipt is misbound")
        child = self.store.load_checkpoint(body["start_checkpoint_id"])
        if child.parents != (body["parent_checkpoint_id"],):
            raise GroupError("member start is not a child of sealed entry")
        commit = self.store.load_commit(body["commit_id"])
        if commit.checkpoint != body["start_checkpoint_id"] or commit.parent_commit is not None:
            raise GroupError("member start receipt misbinds branch commit")
        member = spec.members[ordinal]
        parent = self.store.load_checkpoint(body["parent_checkpoint_id"])
        seeds = self.store.get_artifact(child.state.rng_ref)
        if (
            seeds
            != {
                "record_type": "GroupMemberSeedsV1",
                "group_id": spec.group_id,
                "member_id": member.member_id,
                "derivation": member.seed_provenance,
                "writer_seed": member.writer_seed,
                "environment_seed": member.environment_seed,
                "parent_rng_ref": parent.state.rng_ref,
            }
            or child.state.files != parent.state.files
            or child.state.context_ref != parent.state.context_ref
        ):
            raise GroupError("member start is not an exact isolated sealed branch")
        project_writer_context(
            self.store, body["parent_checkpoint_id"], body["start_checkpoint_id"]
        )
        return body

    def collect(self, spec: GroupSpecV1, result: GroupMemberResultV1) -> str:
        self._assert_sealed_spec(spec)
        ordinal = self._admit_result(spec, result)
        ref = self.store.put_artifact(result.to_dict())
        with self._locked(spec.group_id) as directory:
            path = directory / f"result-{ordinal}.json"
            if path.exists():
                previous_ref = load_canonical_json(path.read_bytes())["result_ref"]
                if previous_ref == ref:
                    return ref
                previous = GroupMemberResultV1.from_dict(self.store.get_artifact(previous_ref))
                self._admit_result(spec, previous, ordinal)
                resolution = (
                    previous.execution_status == "valid"
                    and result.execution_status == "valid"
                    and previous.terminal_outcome_ref == result.terminal_outcome_ref
                    and bool(previous.fixture_ref) == bool(result.fixture_ref)
                    and self._reward_status(previous) != "available"
                )
                first_terminal = (
                    previous.execution_status == "pending"
                    and result.execution_status in {"valid", "infrastructure_invalid"}
                )
                if (
                    previous.group_id != result.group_id
                    or previous.member_id != result.member_id
                    or previous.start_checkpoint_id != result.start_checkpoint_id
                    or not (resolution or first_terminal)
                ):
                    raise GroupError("result replacement is not a monotonic reward resolution")
                temporary = directory / f".result-{ordinal}-{ref}.tmp"
                self._receipt(temporary, {"result_ref": ref})
                os.replace(temporary, path)
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            else:
                self._receipt(path, {"result_ref": ref})
        return ref

    def collect_scripted(
        self,
        spec: GroupSpecV1,
        ordinal: int,
        *,
        reward: Fraction | None = None,
        execution_status: str = "valid",
        reward_status: str = "available",
    ) -> str:
        """Collect a hermetic fake result; it has no writer actions or native targets."""
        if spec.runner_mode != "fixture":
            raise GroupError("scripted results require a fixture-mode group")
        if type(ordinal) is not int or not 0 <= ordinal < len(spec.members):
            raise GroupError("invalid member ordinal")
        if execution_status not in {"valid", "infrastructure_invalid"}:
            raise GroupError("invalid scripted execution status")
        if execution_status == "infrastructure_invalid":
            if reward is not None:
                raise GroupError("infrastructure failure cannot carry a writer reward")
            reward_status = "unavailable"
        if reward_status not in {"available", "pending", "unavailable"}:
            raise GroupError("invalid scripted reward status")
        if (
            execution_status == "valid"
            and reward_status == "available"
            and not isinstance(reward, Fraction)
        ):
            raise GroupError("available scripted reward must be an exact Fraction")
        if reward is not None and not isinstance(reward, Fraction):
            raise GroupError("scripted reward must be exact, never float")
        start = self._start_receipt(spec, ordinal)["start_checkpoint_id"]
        member = spec.members[ordinal]
        fixture = {
            "record_type": "GroupScriptedTerminalV1",
            "schema": 1,
            "group_id": spec.group_id,
            "member_id": member.member_id,
            "start_checkpoint_id": start,
            "execution_status": execution_status,
            "reward_status": reward_status,
            "reward": _fraction(reward)
            if reward_status == "available" and reward is not None
            else None,
            "native_optimizer_eligible": False,
        }
        fixture_ref = self.store.put_artifact(fixture)
        result = GroupMemberResultV1(
            group_id=spec.group_id,
            member_id=member.member_id,
            start_checkpoint_id=start,
            fixture_ref=fixture_ref,
            execution_status=execution_status,
        )
        return self.collect(spec, result)

    def collect_invalid(
        self, spec: GroupSpecV1, ordinal: int, *, reason: str, evidence_ref: str | None = None
    ) -> str:
        """Record a real infrastructure interruption without manufacturing reward."""
        if spec.runner_mode != "real":
            raise GroupError("real interruption records require a real-mode group")
        if type(ordinal) is not int or not 0 <= ordinal < len(spec.members):
            raise GroupError("invalid member ordinal")
        if not isinstance(reason, str) or not reason:
            raise GroupError("infrastructure failure needs a cause")
        validate_hash(evidence_ref, optional=True)
        if evidence_ref is not None:
            self.store.get_artifact(evidence_ref)
        start = self._start_receipt(spec, ordinal)["start_checkpoint_id"]
        member = spec.members[ordinal]
        failure_ref = self.store.put_artifact(
            {
                "record_type": "GroupExecutionFailureV1",
                "schema": 1,
                "group_id": spec.group_id,
                "member_id": member.member_id,
                "start_checkpoint_id": start,
                "reason": reason,
                "evidence_ref": evidence_ref,
            }
        )
        return self.collect(
            spec,
            GroupMemberResultV1(
                group_id=spec.group_id,
                member_id=member.member_id,
                start_checkpoint_id=start,
                failure_ref=failure_ref,
                execution_status="infrastructure_invalid",
            ),
        )

    def _reward_status(self, result: GroupMemberResultV1) -> str:
        if result.fixture_ref:
            return self.store.get_artifact(result.fixture_ref)["reward_status"]
        if result.availability_ref:
            return self.store.get_artifact(result.availability_ref)["reward_status"]
        return "pending"

    def _assert_sealed_spec(self, spec: GroupSpecV1) -> None:
        if canonical_bytes(self.resume(spec.group_id).to_dict()) != canonical_bytes(spec.to_dict()):
            raise GroupError("group spec differs from its sealed receipt")

    def _sampled_trace_ref(self, entry: dict) -> str | None:
        """Only known sampled log records may carry a trace; unknown carriers fail closed."""
        ref = entry["record_ref"]
        private = self.store.artifact_visibilities(ref) == frozenset({"private"})
        record = self.store.get_artifact(ref, private=private)
        if not isinstance(record, dict):
            raise GroupError("runtime log record is not an object")
        expected = {
            "writer_action": "WriterActionV1",
            "budget_charged": "WriterSampledBudgetStopV1",
        }.get(entry["kind"])
        if expected is not None:
            if record.get("record_type") != expected or not isinstance(
                record.get("trace_ref"), str
            ):
                raise GroupError("sampled runtime log lacks its expected trace")
            return record["trace_ref"]
        if "trace_ref" in record:
            raise GroupError("unknown trace-bearing runtime log kind")
        return None

    def _admit_result(
        self, spec: GroupSpecV1, result: GroupMemberResultV1, expected_ordinal: int | None = None
    ) -> int:
        """One immutable admission boundary for live collection and offline recovery."""
        ordinal = next((m.ordinal for m in spec.members if m.member_id == result.member_id), None)
        if (
            result.group_id != spec.group_id
            or ordinal is None
            or (expected_ordinal is not None and ordinal != expected_ordinal)
        ):
            raise GroupError("result belongs to another group or member slot")
        if bool(result.fixture_ref) != (spec.runner_mode == "fixture"):
            raise GroupError("fixture and real terminal results cannot share a group")
        start = self._start_receipt(spec, ordinal)
        if result.start_checkpoint_id != start["start_checkpoint_id"]:
            raise GroupError("result start checkpoint is misbound")
        if result.fixture_ref:
            fixture = self.store.get_artifact(result.fixture_ref)
            if (
                fixture.get("record_type") != "GroupScriptedTerminalV1"
                or fixture.get("schema") != 1
                or fixture.get("group_id") != spec.group_id
                or fixture.get("member_id") != result.member_id
                or fixture.get("start_checkpoint_id") != result.start_checkpoint_id
                or fixture.get("execution_status") != result.execution_status
                or fixture.get("reward_status") not in {"available", "pending", "unavailable"}
                or fixture.get("native_optimizer_eligible") is not False
            ):
                raise GroupError("scripted terminal artifact is misbound")
            if fixture["reward_status"] == "available":
                if result.execution_status != "valid":
                    raise GroupError("invalid scripted execution cannot carry available reward")
                value = fixture.get("reward")
                if (
                    not isinstance(value, dict)
                    or set(value) != {"numerator", "denominator"}
                    or type(value["numerator"]) is not int
                    or type(value["denominator"]) is not int
                    or value["denominator"] <= 0
                    or _fraction(Fraction(value["numerator"], value["denominator"])) != value
                ):
                    raise GroupError("scripted reward is not a canonical exact fraction")
            elif fixture.get("reward") is not None:
                raise GroupError("unavailable scripted reward includes a number")
            return ordinal
        if result.execution_status == "pending":
            return ordinal
        if result.execution_status == "infrastructure_invalid":
            failure = self.store.get_artifact(result.failure_ref)
            if (
                failure.get("record_type") != "GroupExecutionFailureV1"
                or failure.get("schema") != 1
                or failure.get("group_id") != spec.group_id
                or failure.get("member_id") != result.member_id
                or failure.get("start_checkpoint_id") != result.start_checkpoint_id
                or not isinstance(failure.get("reason"), str)
                or not failure["reason"]
                or (
                    failure.get("evidence_ref") is not None
                    and not isinstance(failure["evidence_ref"], str)
                )
            ):
                raise GroupError("infrastructure failure record is misbound")
            if failure["evidence_ref"] is not None:
                self.store.get_artifact(failure["evidence_ref"])
            return ordinal
        final = self.store.load_checkpoint(result.final_checkpoint_id)
        if final.state.position["lineage_id"] != result.member_id:
            raise GroupError("terminal checkpoint belongs to another member")
        cursor = final
        while cursor.identity() != result.start_checkpoint_id:
            if not cursor.parents:
                raise GroupError("terminal checkpoint is not descended from member start")
            cursor = self.store.load_checkpoint(cursor.parents[0])
        project_writer_context(self.store, result.start_checkpoint_id, result.final_checkpoint_id)
        # Phase 6 context operations carry their own immutable policy witness.
        # A worker cannot silently swap that recipe after group admission.
        log = self.store.get_artifact(final.state.external_inputs_ref)
        if isinstance(log, dict) and log.get("record_type") == "WriterRuntimeLogV1":
            member = spec.members[ordinal]
            model = self.store.get_artifact(spec.policy["model_ref"])
            for entry in log["entries"]:
                trace_ref = self._sampled_trace_ref(entry)
                if trace_ref is not None:
                    trace = self.store.get_artifact(trace_ref)
                    try:
                        SamplingEvidenceV1.from_wire(trace)
                    except ProjectionError as exc:
                        raise GroupError(str(exc)) from exc
                    if trace.get("seed") is not None and (
                        type(trace["seed"]) is not int or trace["seed"] != member.writer_seed
                    ):
                        raise GroupError("writer sample used a different sampling stream")
                    if trace.get("model") is not None and (
                        not isinstance(model, dict) or trace["model"] != model.get("model_id")
                    ):
                        raise GroupError("writer sample used a different model")
                    try:
                        bind_group_sampling_claims(
                            spec.policy, member.writer_seed, trace, trace.get("adapter_trace")
                        )
                    except ProjectionError as exc:
                        raise GroupError(str(exc)) from exc
                    if trace.get("exact_request_ref") is not None:
                        try:
                            bind_group_sampling_claims(
                                spec.policy,
                                member.writer_seed,
                                trace,
                                self.store.get_artifact(trace["exact_request_ref"]),
                            )
                        except ProjectionError as exc:
                            raise GroupError(str(exc)) from exc
                if entry["kind"] != "context_changed":
                    continue
                operation = self.store.get_artifact(entry["record_ref"])
                if (
                    operation.get("record_type") == "ContextOperationV1"
                    and operation.get("policy_ref") != spec.policy["context_policy_ref"]
                ):
                    raise GroupError("member context policy drifted after start")
        published_outcome = result.availability_ref or result.terminal_outcome_ref
        if final.state.outcome_ref != published_outcome:
            raise GroupError("terminal/reward artifact is not published by final checkpoint")
        outcome = self.store.get_artifact(result.terminal_outcome_ref)
        if (
            outcome.get("record_type") != "TerminalOutcomeV1"
            or outcome.get("execution_status") != "valid"
            or final.state.position["phase"] != "terminal"
        ):
            raise GroupError("result lacks valid immutable terminal outcome")
        if result.availability_ref:
            availability = self.store.get_artifact(result.availability_ref)
            if (
                availability.get("record_type") != "RewardAvailabilityV1"
                or availability.get("terminal_outcome_ref") != result.terminal_outcome_ref
                or availability.get("reward_status") not in {"available", "pending", "unavailable"}
            ):
                raise GroupError("reward availability is misbound")
            if availability.get("reward_status") == "available":
                reward = self.store.get_artifact(availability["reward_ref"])
                eligibility = self.store.get_artifact(availability["eligibility_ref"])
                if (
                    reward.get("record_type") != "RewardV1"
                    or reward.get("terminal_outcome_ref") != result.terminal_outcome_ref
                    or reward.get("eligibility_ref") != availability["eligibility_ref"]
                    or reward.get("reward_contract_ref") != spec.environment["reward_contract_hash"]
                    or reward.get("candidate_checkpoint") != outcome.get("candidate_checkpoint")
                    or reward.get("check_result_refs") != outcome.get("check_result_refs")
                    or eligibility.get("record_type") != "TrainingEligibilityV1"
                    or eligibility.get("terminal_outcome_ref") != result.terminal_outcome_ref
                    or eligibility.get("status") != "ineligible"
                    or eligibility.get("reason") != CURRENT_ELIGIBILITY.training_reason
                    or type(reward.get("numerator")) is not int
                    or type(reward.get("normalization")) is not int
                    or reward["normalization"] <= 0
                ):
                    raise GroupError("reward/eligibility contract or arithmetic is misbound")
        return ordinal

    def finalize(self, spec: GroupSpecV1) -> GroupDecisionV1:
        self._assert_sealed_spec(spec)
        result_refs: list[str | None] = []
        results: list[GroupMemberResultV1 | None] = []
        directory = self.groups_root / spec.group_id
        for member in spec.members:
            path = directory / f"result-{member.ordinal}.json"
            if not path.exists():
                result_refs.append(None)
                results.append(None)
                continue
            receipt = load_canonical_json(path.read_bytes())
            if not isinstance(receipt, dict) or set(receipt) != {"result_ref"}:
                raise GroupError("result receipt has invalid schema")
            ref = receipt["result_ref"]
            result = GroupMemberResultV1.from_dict(self.store.get_artifact(ref))
            self._admit_result(spec, result, member.ordinal)
            result_refs.append(ref)
            results.append(result)
        if any(r is not None and r.execution_status == "infrastructure_invalid" for r in results):
            status, reason = "invalid", "infrastructure_invalid_member"
        elif any(r is None or r.execution_status == "pending" for r in results):
            status, reason = "pending", "members_pending"
        else:
            rewards = []
            for result in results:
                if result.fixture_ref:
                    fixture = self.store.get_artifact(result.fixture_ref)
                    if fixture["reward_status"] != "available":
                        break
                    value = fixture["reward"]
                    rewards.append(Fraction(value["numerator"], value["denominator"]))
                    continue
                if result.availability_ref is None:
                    break
                availability = self.store.get_artifact(result.availability_ref)
                if availability.get("reward_status") != "available":
                    break
                reward = self.store.get_artifact(availability["reward_ref"])
                rewards.append(Fraction(reward["numerator"], reward["normalization"]))
            if len(rewards) != len(results):
                status, reason = "pending", "reward_pending_or_unavailable"
            else:
                status = "tie" if len(set(rewards)) == 1 else "ready"
                reason = "zero_variance" if status == "tie" else "group_relative"
                mean = sum(rewards, Fraction()) / len(rewards)
                variance = sum(((r - mean) ** 2 for r in rewards), Fraction()) / len(rewards)
                advantage_refs = []
                credit_refs = []
                for ordinal, (result, reward) in enumerate(zip(results, rewards, strict=True)):
                    advantage = GroupAdvantageV1(
                        group_id=spec.group_id,
                        member_id=result.member_id,
                        result_ref=result_refs[ordinal],
                        reward=_fraction(reward),
                        mean=_fraction(mean),
                        variance=_fraction(variance),
                        centered=_fraction(reward - mean),
                        expression="zero"
                        if variance == 0
                        else "centered / sqrt(population_variance)",
                        advantage=_fraction(Fraction()) if variance == 0 else None,
                        zero_variance=variance == 0,
                    )
                    advantage_ref = self.store.put_artifact(advantage.to_dict())
                    advantage_refs.append(advantage_ref)
                    if result.fixture_ref is None:
                        credit_refs.extend(self._segment_credits(spec, result, advantage_ref))
                decision = GroupDecisionV1(
                    group_id=spec.group_id,
                    status=status,
                    reason=reason,
                    member_result_refs=tuple(result_refs),
                    advantage_refs=tuple(advantage_refs),
                    segment_credit_refs=tuple(credit_refs),
                )
                self.store.put_artifact(decision.to_dict())
                return decision
        decision = GroupDecisionV1(
            group_id=spec.group_id,
            status=status,
            reason=reason,
            member_result_refs=tuple(result_refs),
        )
        self.store.put_artifact(decision.to_dict())
        return decision

    def _segment_credits(
        self, spec: GroupSpecV1, result: GroupMemberResultV1, advantage_ref: str
    ) -> list[str]:
        """Bind credit to original sampled action traces, never the compacted tail."""
        final = self.store.load_checkpoint(result.final_checkpoint_id)
        start = self.store.load_checkpoint(result.start_checkpoint_id)
        events = []
        cursor = final.event_head
        while cursor != start.event_head:
            event = self.store.load_event(cursor)
            events.append(event)
            cursor = event.previous
            if cursor is None:
                raise GroupError("action history does not reach member start")
        refs = []
        action_index = 0
        for event in reversed(events):
            if event.kind != "writer_action":
                continue
            if event.rollout_id != result.member_id:
                raise GroupError("writer action belongs to another member")
            action_index += 1
            log = self.store.get_artifact(final.state.external_inputs_ref)
            entry = next((e for e in log.get("entries", []) if e.get("seq") == event.seq), None)
            if entry is None:
                raise GroupError("writer action missing immutable log entry")
            action = self.store.get_artifact(entry["record_ref"])
            trace = self.store.get_artifact(action["trace_ref"])
            try:
                sampled = SamplingEvidenceV1.from_wire(trace)
            except ProjectionError as exc:
                raise GroupError("invalid writer action trace") from exc
            if action.get("record_type") != "WriterActionV1" or sampled.action_id != action.get(
                "action_id"
            ):
                raise GroupError("invalid writer action trace")
            message = MessageV1.from_dict(
                self.store.get_artifact(entry["message_ref"], expected_domain="message")
            )
            if message.role != "assistant" or message.origin != action["action_id"]:
                raise GroupError("credit target is not the owning writer message")

            segments = []
            for index, part in enumerate(message.content):
                kind = {"text": "assistant_text", "tool_call": "tool_syntax"}.get(part["type"])
                if kind is not None and action["loss_eligibility"].get(kind) is True:
                    segments.append((kind, index, _hash(part)))
            if action["loss_eligibility"].get("assistant_ending") is True:
                segments.append(("assistant_ending", None, None))
            for kind, part_index, content_hash in segments:
                credit = GroupSegmentCreditV1(
                    group_id=spec.group_id,
                    member_id=result.member_id,
                    action_id=action["action_id"],
                    action_ref=entry["record_ref"],
                    message_ref=entry["message_ref"],
                    trace_ref=action["trace_ref"],
                    original_context_ref=trace["context_revision_ref"],
                    original_context_content_hash=trace["context_content_hash"],
                    advantage_ref=advantage_ref,
                    segment_kind=kind,
                    part_index=part_index,
                    segment_content_hash=content_hash,
                )
                refs.append(self.store.put_artifact(credit.to_dict()))
        if action_index != len(final.state.history["action_ids"]) - len(
            start.state.history["action_ids"]
        ):
            raise GroupError("writer action history count differs from immutable events")
        return refs
