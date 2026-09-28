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
    validate_hash,
)
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_group_contract import (
    GroupAdvantageV1,
    GroupDecisionV1,
    GroupMemberResultV1,
    GroupSegmentCreditV1,
    derive_group_seed,
    fraction_wire,
    payload_hash,
    resolve_group_environment,
    validate_group_policy,
)
from writing_agent.task_graph_operation import operation_scoped
from writing_agent.task_graph_record_contracts import (
    POLICY_FIELDS,
    ContextPolicyV1,
    GroupError,
    GroupMemberSpecV1,
    GroupSpecV1,
)
from writing_agent.task_graph_records import ContextOperationInputV1, MemberStartV1, WriterTurnV1
from writing_agent.task_graph_rollout_env import RolloutEnvironment
from writing_agent.task_graph_rollout_env import RuntimeHandle as RolloutRuntimeHandle
from writing_agent.task_graph_sampling import (
    bind_group_sampling_claims,
)
from writing_agent.task_graph_store import RuntimeHandle as LegacyRuntimeHandle
from writing_agent.task_graph_store import TaskGraphStore


class GroupCoordinatorV1:
    """A serializable fake runner's admission, start, collection and finalization API."""

    def __init__(
        self,
        store_or_environment: TaskGraphStore | RolloutEnvironment,
        workers_root: Path | str,
        *,
        session=None,
    ):
        self.environment = (
            store_or_environment if isinstance(store_or_environment, RolloutEnvironment) else None
        )
        self.store = (
            self.environment.store if self.environment is not None else store_or_environment
        )
        self.session = session or (None if self.environment is None else self.environment.session)
        self.workers_root = Path(workers_root).resolve()
        self.workers_root.mkdir(parents=True, exist_ok=True)
        self.groups_root = self.store.root / "groups"
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

    @operation_scoped
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
        environment, rendering = self._entry_contract(entry_checkpoint_id)
        policy = validate_group_policy(policy, rendering)
        if self.session is not None:
            self.session.require_seal(policy["adapter_ref"])
        for field in POLICY_FIELDS - {"rng_derivation_version"}:
            self.store.get_artifact(policy[field])
        ContextPolicyV1.from_dict(self.store.get_artifact(policy["context_policy_ref"]))
        if type(member_count) is not int or not 2 <= member_count <= 64:
            raise GroupError("group size must be 2..64")
        if type(group_sequence) is not int or group_sequence < 0:
            raise GroupError("group sequence must be nonnegative")
        group_id = payload_hash(
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
                writer_seed=derive_group_seed(group_seed, "writer", ordinal),
                environment_seed=derive_group_seed(group_seed, "environment"),
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
        # Keep the sealed contract addressable to the pure member-start derive.
        # Its payload identity is exactly GroupSpecV1.identity().
        self.store.put_artifact(spec.to_wire())
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

    def _entry_contract(self, checkpoint_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.environment is None:
            state = self.store.load_checkpoint(checkpoint_id).state
            rendering = self.store.load_context(state.context_ref).rendering
            return resolve_group_environment(self.store, checkpoint_id), dict(rendering)
        view = self.environment.verify(self.environment.open(checkpoint_id))
        return (
            resolve_group_environment(
                self.store,
                checkpoint_id,
                view=view,
                instance=self.environment.graph.instance,
            ),
            dict(view.context.rendering),
        )

    @operation_scoped
    def resume(self, group_id: str) -> GroupSpecV1:
        validate_hash(group_id)
        path = self.groups_root / group_id / "spec.json"
        spec = GroupSpecV1.from_dict(load_canonical_json(path.read_bytes()))
        if spec.group_id != group_id:
            raise GroupError("group directory misbinds spec")
        for field in POLICY_FIELDS - {"rng_derivation_version"}:
            self.store.get_artifact(spec.policy[field])
        ContextPolicyV1.from_dict(self.store.get_artifact(spec.policy["context_policy_ref"]))
        current, _ = self._entry_contract(spec.environment["entry_checkpoint_id"])
        if current != spec.environment:
            raise GroupError("sealed entry contract drifted")
        return spec

    def assert_start_contract(
        self, spec: GroupSpecV1, checkpoint_id: str, policy: dict[str, str]
    ) -> None:
        if self.session is not None:
            self.session.require_seal(spec.policy["adapter_ref"])
        candidate, rendering = self._entry_contract(checkpoint_id)
        if canonical_bytes(candidate) != canonical_bytes(spec.environment):
            raise GroupError("member entry differs from full sealed environment contract")
        if canonical_bytes(validate_group_policy(policy, rendering)) != canonical_bytes(
            spec.policy
        ):
            raise GroupError("member policy contract drifted")

    @operation_scoped
    def start(
        self, spec: GroupSpecV1, ordinal: int, *, policy: dict[str, str]
    ) -> RolloutRuntimeHandle | LegacyRuntimeHandle:
        """Start a member from its sealed slot, resuming a published head on retry."""
        if type(ordinal) is not int or not 0 <= ordinal < len(spec.members):
            raise GroupError("invalid member ordinal")
        if self.environment is None:
            return self._start_legacy(spec, ordinal, policy=policy)

        self.resume(spec.group_id)
        member = spec.members[ordinal]
        self.assert_start_contract(spec, spec.environment["entry_checkpoint_id"], policy)
        parent_id = spec.environment["entry_checkpoint_id"]
        head = self.store.read_head(member.member_id)
        receipt_path = self.groups_root / spec.group_id / f"start-{ordinal}.json"
        if head is not None and receipt_path.exists():
            receipt = self._start_receipt(spec, ordinal)
            child_id, commit = receipt["start_checkpoint_id"], receipt["commit_id"]
            runtime = self.environment.open_head(member.member_id)
        elif head is None:
            runtime = self.environment.start_member(
                parent_id, MemberStartV1(spec.identity(), ordinal)
            )
            child_id = runtime.checkpoint_id
            commit = self.store.read_head(member.member_id)
        else:
            runtime = self.environment.open_head(member.member_id)
            child_id = runtime.checkpoint_id
            commit = head
        view = self.environment.verify(runtime)
        self._assert_member_view(spec, member, view)
        if commit is None:
            raise GroupError("member start was not published")
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
        return runtime

    def _start_legacy(
        self, spec: GroupSpecV1, ordinal: int, *, policy: dict[str, str]
    ) -> LegacyRuntimeHandle:
        """Temporary old-runtime bridge; S7.2 retargets its remaining caller."""
        self.resume(spec.group_id)
        member = spec.members[ordinal]
        self.assert_start_contract(spec, spec.environment["entry_checkpoint_id"], policy)
        parent_id = spec.environment["entry_checkpoint_id"]
        parent = self.store.load_checkpoint(parent_id)
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
            raise GroupError("member workspace exists; remove disposable copy before restoring")
        return self.store.restore(child_id, destination)

    def _assert_member_view(self, spec, member, view) -> None:
        if view.state.position["lineage_id"] != member.member_id or view.group != spec:
            raise GroupError("verified member view differs from its sealed start")

    def _start_receipt(self, spec: GroupSpecV1, ordinal: int) -> dict:
        if self.environment is None:
            raise GroupError("member receipt verification requires a RolloutEnvironment")
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
        runtime = self.environment.open_head(member.member_id)
        view = self.environment.verify(runtime)
        self._assert_member_view(spec, member, view)
        chain = view.ancestry
        while chain is not None and chain.checkpoint_id != body["start_checkpoint_id"]:
            chain = chain.parent
        if chain is None:
            raise GroupError("member start receipt differs from verified lineage")
        return body

    @operation_scoped
    def collect(self, spec: GroupSpecV1, result: GroupMemberResultV1) -> str:
        if self.session is not None:
            self.session.require_seal(spec.policy["adapter_ref"])
        self._assert_sealed_spec(spec)
        ordinal, _ = self._admit_result(spec, result)
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

    @operation_scoped
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
            "reward": fraction_wire(reward)
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

    @operation_scoped
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
            reward = self.store.get_artifact(result.availability_ref)
            if reward.get("record_type") != "RewardV1":
                raise GroupError("member reward reference is not RewardV1")
            return "available"
        return "pending"

    def _assert_sealed_spec(self, spec: GroupSpecV1) -> None:
        if canonical_bytes(self.resume(spec.group_id).to_dict()) != canonical_bytes(spec.to_dict()):
            raise GroupError("group spec differs from its sealed receipt")

    def _admit_result(
        self, spec: GroupSpecV1, result: GroupMemberResultV1, expected_ordinal: int | None = None
    ) -> tuple[int, Any | None]:
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
                    or fraction_wire(Fraction(value["numerator"], value["denominator"])) != value
                ):
                    raise GroupError("scripted reward is not a canonical exact fraction")
            elif fixture.get("reward") is not None:
                raise GroupError("unavailable scripted reward includes a number")
            return ordinal, None
        if result.execution_status == "pending":
            return ordinal, None
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
            return ordinal, None

        view = self._verified_member_view(spec, ordinal)
        final = self.store.load_checkpoint(result.final_checkpoint_id)
        if final.state.position["lineage_id"] != result.member_id:
            raise GroupError("terminal checkpoint belongs to another member")
        self._check_member_policy_bindings(spec, ordinal, view)
        outcome = view.outcome
        if outcome.execution_status != "valid" or view.state.position["phase"] != "terminal":
            raise GroupError("result lacks a valid terminal outcome")
        if outcome.reward_status == "available":
            if not result.availability_ref or outcome.reward_ref != result.availability_ref:
                raise GroupError("result reward differs from the verified outcome")
            reward = self.store.get_artifact(result.availability_ref)
            if (
                reward.get("record_type") != "RewardV1"
                or result.terminal_outcome_ref != reward.get("terminal_outcome_ref")
                or reward.get("reward_contract_ref") != spec.environment["reward_contract_hash"]
                or type(reward.get("numerator")) is not int
                or type(reward.get("normalization")) is not int
                or reward["normalization"] <= 0
                or reward.get("eligibility_ref") != outcome.eligibility_ref
            ):
                raise GroupError("reward is not bound to the verified outcome")
            terminal = self.store.get_artifact(result.terminal_outcome_ref)
            eligibility = self.store.get_artifact(outcome.eligibility_ref)
            if (
                terminal.get("record_type") != "OutcomeV1"
                or terminal.get("reward_status") != "pending"
                or terminal.get("execution_status") != "valid"
                or eligibility.get("record_type") != "TrainingEligibilityV1"
                or eligibility.get("terminal_outcome_ref") != result.terminal_outcome_ref
                or eligibility.get("status") != outcome.training_eligibility
            ):
                raise GroupError("training eligibility is not bound to the reward outcome")
        elif (
            result.availability_ref is not None
            or result.terminal_outcome_ref != view.state.outcome_ref
        ):
            raise GroupError("result outcome differs from the verified head view")
        return ordinal, view

    def _verified_member_view(self, spec: GroupSpecV1, ordinal: int):
        if self.environment is None:
            raise GroupError("real group collection requires a RolloutEnvironment")
        member = spec.members[ordinal]
        runtime = self.environment.open_head(member.member_id)
        view = self.environment.verify(runtime)
        self._assert_member_view(spec, member, view)
        return view

    def _check_member_policy_bindings(self, spec: GroupSpecV1, ordinal: int, view) -> None:
        member = spec.members[ordinal]
        model = self.store.get_artifact(spec.policy["model_ref"])
        if not isinstance(model, dict) or not isinstance(model.get("model_id"), str):
            raise ProjectionError("sealed group model has no model ID")
        for sample in view.samples:
            turn = WriterTurnV1.from_dict(self.store.get_artifact(sample.turn_ref))
            context = self._context_for_revision(view, turn.context_revision_ref)
            trace = turn.adapter_trace or {}
            bind_group_sampling_claims(
                spec.policy,
                member.writer_seed,
                trace,
                trace,
                model_id=model["model_id"],
                context_content_hash=context.content_ref,
                context_revision_ref=context.revision_ref,
                rendering=context.rendering,
            )

        event_id = view.head_event_id
        while event_id is not None:
            event = self.store.load_event(event_id)
            if event.kind == "context_changed":
                operation = ContextOperationInputV1.from_dict(
                    self.store.get_artifact(event.payload_ref)
                )
                if operation.policy_ref != spec.policy["context_policy_ref"]:
                    raise ProjectionError("member context policy differs from the group spec")
            event_id = event.previous

    @staticmethod
    def _context_for_revision(view, revision_ref: str):
        chain = view.ancestry
        while chain is not None:
            if chain.context.revision_ref == revision_ref:
                return chain.context
            chain = chain.parent
        raise ProjectionError("sample context is absent from the verified member view")

    @operation_scoped
    def finalize(self, spec: GroupSpecV1) -> GroupDecisionV1:
        if self.session is not None:
            self.session.require_seal(spec.policy["adapter_ref"])
        self._assert_sealed_spec(spec)
        result_refs: list[str | None] = []
        results: list[GroupMemberResultV1 | None] = []
        views = []
        directory = self.groups_root / spec.group_id
        for member in spec.members:
            path = directory / f"result-{member.ordinal}.json"
            if not path.exists():
                result_refs.append(None)
                results.append(None)
                views.append(None)
                continue
            receipt = load_canonical_json(path.read_bytes())
            if not isinstance(receipt, dict) or set(receipt) != {"result_ref"}:
                raise GroupError("result receipt has invalid schema")
            ref = receipt["result_ref"]
            result = GroupMemberResultV1.from_dict(self.store.get_artifact(ref))
            _, view = self._admit_result(spec, result, member.ordinal)
            result_refs.append(ref)
            results.append(result)
            views.append(view)
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
                reward = self.store.get_artifact(result.availability_ref)
                if reward.get("record_type") != "RewardV1":
                    raise GroupError("member reward reference is not RewardV1")
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
                        reward=fraction_wire(reward),
                        mean=fraction_wire(mean),
                        variance=fraction_wire(variance),
                        centered=fraction_wire(reward - mean),
                        expression="zero"
                        if variance == 0
                        else "centered / sqrt(population_variance)",
                        advantage=fraction_wire(Fraction()) if variance == 0 else None,
                        zero_variance=variance == 0,
                    )
                    advantage_ref = self.store.put_artifact(advantage.to_dict())
                    advantage_refs.append(advantage_ref)
                    if result.fixture_ref is None:
                        credit_refs.extend(
                            self._segment_credits(spec, result, advantage_ref, views[ordinal])
                        )
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
        self,
        spec: GroupSpecV1,
        result: GroupMemberResultV1,
        advantage_ref: str,
        view,
    ) -> list[str]:
        """Emit segment targets from the gate-verified sample index, not runtime logs."""
        if view is None:
            raise GroupError("real segment credit requires a verified member view")
        refs = []
        for sample in view.samples:
            if sample.outcome != "action":
                continue
            turn = WriterTurnV1.from_dict(self.store.get_artifact(sample.turn_ref))
            message = self._message_for_sample(view, sample.event_id)
            if message.role != "assistant" or message.origin != sample.action_id:
                raise GroupError("sample does not own its derived assistant message")
            message_ref = self.store.persist(message)
            context = self._context_for_revision(view, turn.context_revision_ref)

            segments = []
            for index, part in enumerate(message.content):
                kind = {
                    "text": "assistant_text",
                    "tool_call": "tool_syntax",
                    "invalid_tool_call": "tool_syntax",
                }.get(part["type"])
                if kind is not None:
                    segments.append((kind, index, payload_hash(part)))
            segments.append(("assistant_ending", None, None))
            for kind, part_index, content_hash in segments:
                credit = GroupSegmentCreditV1(
                    group_id=spec.group_id,
                    member_id=result.member_id,
                    action_id=sample.action_id,
                    action_ref=sample.event_id,
                    message_ref=message_ref,
                    trace_ref=sample.turn_ref,
                    original_context_ref=turn.context_revision_ref,
                    original_context_content_hash=context.content_ref,
                    advantage_ref=advantage_ref,
                    segment_kind=kind,
                    part_index=part_index,
                    segment_content_hash=content_hash,
                )
                refs.append(self.store.put_artifact(credit.to_dict()))
        return refs

    @staticmethod
    def _message_for_sample(view, event_id: str) -> MessageV1:
        chain = view.ancestry
        while chain is not None:
            for source, message in zip(chain.context.sources, chain.context.messages, strict=True):
                if source == event_id:
                    return message
            chain = chain.parent
        raise GroupError("sample event has no verified assistant message")
