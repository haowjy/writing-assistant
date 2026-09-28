"""Producer, persistence, privacy, and recovery contracts for the new environment."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

from tests.task_graph_fixtures import make_entry_fixture
from tests.test_task_graph_derive_author import _answer_rule, _answered_reply
from tests.test_task_graph_derive_outcome import make_outcome_fixture
from tests.test_task_graph_derive_writer import call, make_turn, with_budget
from writing_agent.task_graph import EventV1, domain_hash, load_canonical_json
from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_contracts import (
    AuthorPacketV1,
    DecisionBindingsV1,
    InteractionContractV1,
    InteractionPolicyV1,
    ScriptedAuthorV1,
)
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_errors import (
    AdapterContractError,
    MissingReferenceError,
    ProjectionError,
)
from writing_agent.task_graph_gate import DERIVE, LineageGate
from writing_agent.task_graph_group import POLICY_FIELDS, derive_group_seed
from writing_agent.task_graph_record_contracts import GroupMemberSpecV1, GroupSpecV1
from writing_agent.task_graph_records import (
    AdmissionPolicyV1 as AdmissionPolicyRecord,
)
from writing_agent.task_graph_records import (
    ContextOperationInputV1,
    EnvironmentStepV1,
    MemberStartV1,
    ToolObservationV1,
)
from writing_agent.task_graph_rollout_env import (
    AuthorInput,
    CheckInput,
    RolloutEnvironment,
    SamplerInput,
    ToolInput,
)
from writing_agent.task_graph_store import TaskGraphStore

CANARY = "PORT_PRIVACY_CANARY_9814"
FAULT_STAGES = (
    "before_immutable_writes",
    "after_immutable_writes",
    "before_head_publication",
    "after_head_publication",
)
EVENT_KINDS = (
    "rollout_started",
    "writer_action",
    "budget_charged",
    "tool_result",
    "external_requested",
    "author_turn",
    "termination_recorded",
    "check_recorded",
    "transition_committed",
    "reward_recorded",
    "context_changed",
)
# `budget_charged` cannot be produced by an admitted entry: derive_entry has no token limits.


def _persist_fixture(store, fixture, *, checkpoint=True):
    for body in fixture.reader.public.values():
        store.put_artifact(body)
    for body in fixture.reader.private.values():
        store.put_artifact(body, private=True)
    store.persist(fixture.graph.instance)
    for artifact in fixture.artifacts:
        if artifact.kind in {"context_node", "context_revision"}:
            store.persist(artifact.value)
        elif artifact.value_kind == "bytes":
            store.put_bytes_artifact(artifact.value, private=artifact.kind == "private")
        else:
            body = (
                load_canonical_json(artifact.value)
                if artifact.value_kind == "canonical_json"
                else artifact.value.to_wire()
            )
            store.put_artifact(body, private=artifact.kind == "private")
    if checkpoint:
        return store.save_checkpoint(fixture.state)
    return None


def _author_check_fixture():
    fixture = make_outcome_fixture()
    reader = fixture.reader
    old_node = fixture.graph.node(fixture.node_id)
    old_check = old_node.checks["nonempty"]
    check = replace(old_check, id=CANARY, spec={**old_check.spec, "id": CANARY})
    reader.private[check.identity()] = check.to_dict()
    reward = replace(
        old_node.reward_contract,
        components={CANARY: old_node.reward_contract.components["nonempty"]},
    )
    reader.private[reward.identity()] = reward.to_dict()
    packet = replace(
        old_node.evaluator_packet,
        reward_contract_ref=reward.identity(),
        check_ids=(CANARY,),
    )
    reader.private[packet.identity()] = packet.to_dict()
    base_contract = old_node.contract
    completion = replace(
        base_contract.completion_contract,
        required_check_ids=(CANARY,),
        evaluation_packet_ref=packet.identity(),
    )
    base_contract = replace(
        base_contract, completion=completion, mandatory_checks=(check.identity(),)
    )
    reader.public[base_contract.identity()] = base_contract.to_dict()
    base_spec = replace(old_node.spec, entry_contract=base_contract.identity())
    base_instance = replace(fixture.graph.instance, nodes=(base_spec,))
    base_graph = admit_graph(
        base_instance,
        MappingArtifactResolver(reader.public, reader.private),
        policy=fixture.graph.policy,
    )
    old_node = base_graph.node(fixture.node_id)
    packet = AuthorPacketV1(preferences={"choice": "blue"}, requirements={"r1": CANARY})
    packet_ref = reader.add(packet.to_dict(), private=True)
    script = ScriptedAuthorV1(answers={"pick": _answer_rule()})
    script_ref = reader.add(script.to_dict(), private=True)
    policy = InteractionPolicyV1(public_decisions=({"id": "pick", "label": "Select option"},))
    policy_ref = reader.add(policy.to_dict())
    bindings = DecisionBindingsV1(bindings={"pick": "choice"})
    bindings_ref = reader.add(bindings.to_dict(), private=True)
    interaction = InteractionContractV1(
        mode="scripted_author",
        script_ref=script_ref,
        author_packet_ref=packet_ref,
        interaction_policy_ref=policy_ref,
        decision_bindings_ref=bindings_ref,
    )
    entry = replace(
        old_node.contract.entry_contract,
        tool_allowlist=(*old_node.contract.entry_contract.tool_allowlist, "ask_author"),
    )
    budget = replace(old_node.contract.budget_contract, max_author_calls=2)
    contract = replace(old_node.contract, entry=entry, interaction=interaction, budgets=budget)
    reader.public[contract.identity()] = contract.to_dict()
    spec = replace(old_node.spec, entry_contract=contract.identity())
    instance = replace(fixture.graph.instance, nodes=(spec,))
    graph = admit_graph(
        instance,
        MappingArtifactResolver(reader.public, reader.private),
        policy=fixture.graph.policy,
    )
    entry_value = derive_entry(graph, fixture.node_id, fixture.params, reader)
    return replace(fixture, graph=graph, state=entry_value.state, artifacts=entry_value.artifacts)


def _group_spec(fixture, entry_checkpoint, store):
    state = fixture.state

    def marker(field):
        return domain_hash("payload", {"fixture": field})

    environment = {
        name: marker(name)
        for name in "entry_tree_hash graph_hash controller_contract_hash check_contracts_hash "
        "source_refs_hash request_refs_hash visible_prefix_hash context_messages_hash "
        "rendering_hash tool_schemas_hash budget_hash versions_hash continuation_hash".split()
    }
    environment.update(
        entry_state_hash=state.identity(),
        instance_hash=state.instance_ref,
        node_contract_hash=state.position["entry_contract"],
        budget_ref=state.budgets_ref,
        versions_ref=state.versions_ref,
        decisions_ref=state.decisions_ref,
        disclosures_ref=state.disclosures_ref,
        external_inputs_ref=state.external_inputs_ref,
        rng_ref=state.rng_ref,
        outcome_ref=state.outcome_ref,
        provenance_ref=state.provenance_ref,
        entry_checkpoint_id=entry_checkpoint,
        node_id=state.position["node_id"],
        node_visit_id=state.position["visit_id"],
        reward_contract_hash=None,
        simulator_contract_hash=None,
        context_revision_ref=state.context_ref,
        author_packet_ref=None,
        requirements_ref=state.requirements_ref,
        horizon="node_exit",
    )
    policy = {name: fixture.params.rng_ref for name in POLICY_FIELDS - {"rng_derivation_version"}}
    policy["context_policy_ref"] = store.put_artifact(ContextPolicyV1("drop").to_wire())
    policy["rng_derivation_version"] = "sha256-domain-v1"
    sequence, seed, mode, count = 0, 771, "fixture", 2
    group_id = domain_hash(
        "payload", ["GroupIdV1", sequence, environment, policy, seed, mode, count]
    )
    members = tuple(
        GroupMemberSpecV1(
            member_id=f"grp-{group_id[:24]}-{ordinal:02d}",
            ordinal=ordinal,
            writer_seed=derive_group_seed(seed, "writer", ordinal),
            environment_seed=derive_group_seed(seed, "environment"),
        )
        for ordinal in range(count)
    )
    spec = GroupSpecV1(group_id, sequence, seed, mode, environment, policy, members)
    store.put_artifact(spec.to_wire())
    return spec


def _assert_fault_matrix(
    test,
    environment,
    runtime,
    input_record,
    root,
    *,
    event_kind,
    entry_checkpoint=None,
    lineage=None,
):
    """Run the same valid one-event commit against every store crash hook."""
    lineage = lineage or runtime.state.position["lineage_id"]
    retry_ids = []
    for stage in FAULT_STAGES:
        clone_root = root / f"matrix-{event_kind}-{stage}"
        clone_root.mkdir()
        clone_root.chmod(0o700)
        try:
            shutil.copytree(environment.store.root, clone_root / "store")
            gate = LineageGate()
            store = TaskGraphStore(clone_root / "store", verifier=gate)
            env = RolloutEnvironment(
                store, environment.graph, None, gate, environment.admission_policy
            )
            cloned_runtime = type(runtime)(
                runtime.checkpoint_id, runtime.state, runtime.context, runtime.workspace
            )
            old_head = store.read_head(lineage)
            publish = store.publish
            captured = {}

            def crash(
                *args,
                _stage=stage,
                _publish=publish,
                _captured=captured,
                **kwargs,
            ):
                _captured["args"] = args
                _captured["kwargs"] = dict(kwargs)

                def fail(current, expected=_stage):
                    if current == expected:
                        raise RuntimeError("injected crash")

                kwargs["fault"] = fail
                return _publish(*args, **kwargs)

            with test.subTest(event=event_kind, stage=stage):
                with patch.object(store, "publish", side_effect=crash):
                    with test.assertRaisesRegex(RuntimeError, "injected crash"):
                        if entry_checkpoint is None:
                            env.commit(cloned_runtime, input_record)
                        else:
                            env.start_member(
                                entry_checkpoint,
                                input_record,
                                clone_root / "member-workspace",
                            )
                head = store.read_head(lineage)
                if stage == "after_head_publication":
                    test.assertNotEqual(head, old_head)
                else:
                    test.assertEqual(head, old_head)
                    test.assertEqual(
                        env.verify(cloned_runtime).checkpoint_id, runtime.checkpoint_id
                    )

                args = captured["args"]
                retry_kwargs = captured["kwargs"]
                if stage == "after_head_publication":
                    published = publish(*args, **retry_kwargs)
                    commit_id = published[0] if isinstance(published, tuple) else published
                    test.assertEqual(commit_id, head)
                elif entry_checkpoint is None:
                    commit_id = env.commit(cloned_runtime, input_record).commit_id
                else:
                    env.start_member(entry_checkpoint, input_record, clone_root / "retry-workspace")
                    commit_id = store.read_head(lineage)
                test.assertEqual(store.read_head(lineage), commit_id)
                retry_ids.append(commit_id)
                commit = store.load_commit(commit_id)
                event = store.load_event(commit.events[0])
                test.assertEqual(event.kind, event_kind)
                test.assertEqual(
                    store.load_checkpoint(commit.checkpoint).artifact_refs,
                    (),
                )
                if stage == "after_head_publication":
                    opened = env.open(commit.checkpoint, clone_root / "recovered-workspace")
                    test.assertEqual(env.verify(opened).checkpoint_id, commit.checkpoint)
        finally:
            shutil.rmtree(clone_root, ignore_errors=True)
    test.assertEqual(len(set(retry_ids)), 1)


class RolloutEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_entry_fixture()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gate = LineageGate()
        self.store = TaskGraphStore(self.root / "store", verifier=self.gate)
        self.entry = _persist_fixture(self.store, self.fixture)
        self.environment = RolloutEnvironment(
            self.store, self.fixture.graph, None, self.gate, self.fixture.graph.policy
        )
        self.runtime = self.environment.open(self.entry, self.root / "entry-workspace")

    def _alternate_policy_entry(self, lineage: str):
        policy = replace(
            self.fixture.graph.policy,
            allowed_tools=self.fixture.graph.policy.allowed_tools - {"ask_author"},
        )
        graph = admit_graph(
            self.fixture.graph.instance,
            MappingArtifactResolver(self.fixture.reader.public, self.fixture.reader.private),
            policy=policy,
        )
        policy_ref = self.store.put_artifact(
            AdmissionPolicyRecord.from_admission_policy(policy).to_wire()
        )
        versions = self.store.get_artifact(self.fixture.state.versions_ref)
        versions["admission_policy_ref"] = policy_ref
        versions_ref = self.store.put_artifact(versions)
        params = replace(
            self.fixture.params,
            lineage_id=lineage,
            visit_id=lineage,
            versions_ref=versions_ref,
        )
        return graph, policy, params

    def _enter_alternate_policy(self, lineage: str):
        graph, policy, params = self._alternate_policy_entry(lineage)
        environment = RolloutEnvironment(self.store, graph, None, self.gate, policy)
        runtime = environment.enter(
            self.fixture.node_id,
            params,
            self.root / f"{lineage}-workspace",
        )
        return environment, runtime

    def test_enter_rejects_entry_params_with_a_different_admission_policy(self):
        _, _, params = self._alternate_policy_entry("policy-enter")
        with self.assertRaises(ProjectionError):
            self.environment.enter(
                self.fixture.node_id,
                params,
                self.root / "mismatched-entry-workspace",
            )
        self.assertIsNone(self.store.read_head("policy-enter"))

    def test_open_rejects_a_lineage_pinned_to_a_different_admission_policy(self):
        _, runtime = self._enter_alternate_policy("policy-open")
        with self.assertRaises(ProjectionError):
            self.environment.open(
                runtime.checkpoint_id,
                self.root / "mismatched-open-workspace",
            )

    def test_verify_rejects_a_lineage_pinned_to_a_different_admission_policy(self):
        _, runtime = self._enter_alternate_policy("policy-verify")
        with self.assertRaises(ProjectionError):
            self.environment.verify(runtime)

    def test_writer_and_tool_commits_close_without_artifact_refs_and_fold_once(self):
        view = self.environment.verify(self.runtime)
        self.assertEqual(next_step(view).kind, "sample_writer")
        counts = {"writer": 0, "tool": 0}
        writer, tool = DERIVE["WriterTurnV1"], DERIVE["ToolObservationV1"]

        def count_writer(*args):
            counts["writer"] += 1
            return writer(*args)

        def count_tool(*args):
            counts["tool"] += 1
            return tool(*args)

        with patch.dict(DERIVE, {"WriterTurnV1": count_writer, "ToolObservationV1": count_tool}):
            self.environment.verify(self.runtime)
            self.assertEqual(counts["writer"], 0)
            turn = make_turn(
                view,
                content="",
                calls=(call("read_file", {"path": "draft.txt"}, "read-first"),),
            )
            _assert_fault_matrix(
                self,
                self.environment,
                self.runtime,
                turn,
                self.root,
                event_kind="writer_action",
            )
            counts["writer"] = 0
            first = self.environment.commit(self.runtime, turn)
            self.assertEqual(counts["writer"], 2)
            first_checkpoint = self.store.load_checkpoint(first.runtime.checkpoint_id)
            self.assertEqual(first_checkpoint.artifact_refs, ())

            before_verify = counts.copy()
            verified = self.environment.verify(first.runtime)
            self.assertEqual(counts, before_verify)
            port = self.environment.port_input(verified, first.directive)
            self.assertIsInstance(port, ToolInput)
            queued = port.queue_entry
            observation = ToolObservationV1(
                call_id=queued["call_id"],
                dispatch={
                    "spec": {
                        "max_file_bytes": port.tool_spec.max_file_bytes,
                        "max_workspace_bytes": port.tool_spec.max_workspace_bytes,
                    },
                    "observation": {"ok": True, "valid": True, "result": "alpha"},
                    "effect": {},
                },
            )
            _assert_fault_matrix(
                self,
                self.environment,
                first.runtime,
                observation,
                self.root,
                event_kind="tool_result",
            )
            counts["tool"] = 0
            second = self.environment.commit(first.runtime, observation)
            self.assertEqual(counts["tool"], 2)
            second_checkpoint = self.store.load_checkpoint(second.runtime.checkpoint_id)
            self.assertEqual(second_checkpoint.artifact_refs, ())
            self.assertEqual(
                self.environment.verify(second.runtime).checkpoint_id,
                second.runtime.checkpoint_id,
            )

    def test_context_operation_commits_through_gate(self):
        policy = ContextPolicyV1("drop")
        policy_ref = self.store.put_artifact(policy.to_wire())
        operation = ContextOperationInputV1(policy_ref=policy_ref)
        _assert_fault_matrix(
            self,
            self.environment,
            self.runtime,
            operation,
            self.root,
            event_kind="context_changed",
        )
        result = self.environment.commit(self.runtime, operation)
        self.assertEqual(result.runtime.state.history["seq"], 1)
        self.assertEqual(self.store.load_checkpoint(result.runtime.checkpoint_id).artifact_refs, ())
        self.assertEqual(
            self.environment.verify(result.runtime).checkpoint_id,
            result.runtime.checkpoint_id,
        )

    def test_author_check_transition_and_reward_inputs_commit(self):
        from writing_agent.task_graph_evaluation import (
            FAMILIES,
            EvaluationRequestV1,
            produce_evaluation_evidence,
        )
        from writing_agent.task_graph_records import EvaluatorResultV1

        fixture = _author_check_fixture()
        root = self.root / "author-check"
        root.mkdir()
        root.chmod(0o700)
        gate = LineageGate()
        store = TaskGraphStore(root / "store", verifier=gate)
        entry = _persist_fixture(store, fixture)
        env = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        runtime = env.open(entry, root / "workspace")

        ask = call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["pick"],
                "proposals": [{"id": "p1", "text": "The blue door"}],
                "option_refs": ["p1"],
            },
            "ask-door",
        )
        ask_turn = make_turn(env.verify(runtime), content="", calls=(ask,))
        _assert_fault_matrix(self, env, runtime, ask_turn, root, event_kind="writer_action")
        writer_step = env.commit(runtime, ask_turn)
        self.assertEqual(store.load_event(writer_step.event_id).kind, "writer_action")
        request_author = EnvironmentStepV1(
            directive={"kind": "request_author", "source": "writer_request"}
        )
        _assert_fault_matrix(
            self,
            env,
            writer_step.runtime,
            request_author,
            root,
            event_kind="external_requested",
        )
        requested = env.commit(writer_step.runtime, request_author)
        author_view = env.verify(requested.runtime)
        author_port = env.port_input(author_view, requested.directive)
        self.assertIsInstance(author_port, AuthorInput)
        request_ref = requested.runtime.state.continuation["author_request"]
        fixture.reader.private[request_ref] = env.reader.artifact(request_ref, private=True)
        author_reply = _answered_reply(fixture, author_view)
        _assert_fault_matrix(
            self, env, requested.runtime, author_reply, root, event_kind="author_turn"
        )
        replied = env.commit(requested.runtime, author_reply)
        self.assertEqual(store.load_event(replied.event_id).kind, "author_turn")

        final_turn = make_turn(env.verify(replied.runtime), content="Done.")
        _assert_fault_matrix(
            self, env, replied.runtime, final_turn, root, event_kind="writer_action"
        )
        final = env.commit(replied.runtime, final_turn)
        self.assertEqual(final.directive.kind, "request_checks")
        request_checks = EnvironmentStepV1(directive={"kind": "request_checks"})
        _assert_fault_matrix(
            self,
            env,
            final.runtime,
            request_checks,
            root,
            event_kind="external_requested",
        )
        requested_checks = env.commit(final.runtime, request_checks)
        check_view = env.verify(requested_checks.runtime)
        check_port = env.port_input(check_view, requested_checks.directive)
        self.assertIsInstance(check_port, CheckInput)
        check = fixture.graph.node(fixture.node_id).checks[check_port.request["check_id"]]
        family = next(
            item for item in FAMILIES.values() if item.check_version == check.evaluator_version
        )
        request_ref = requested_checks.runtime.state.continuation["check_requests"][0]
        packet_ref = check_port.request["evaluator_packet_ref"]
        evaluation = EvaluationRequestV1.create(
            family.name,
            check_port.request["target_checkpoint"],
            check,
            packet_ref,
            check_port.files,
            evaluator_packet=check_port.evaluator_packet,
        )
        evidence = produce_evaluation_evidence(evaluation).to_wire(evaluation)
        evidence_ref = store.put_artifact(evidence)
        evaluation_result = EvaluatorResultV1(
            request_ref=request_ref,
            status="pass",
            evidence_ref=evidence_ref,
        )
        _assert_fault_matrix(
            self,
            env,
            requested_checks.runtime,
            evaluation_result,
            root,
            event_kind="check_recorded",
        )
        checked = env.commit(requested_checks.runtime, evaluation_result)
        self.assertEqual(store.load_event(checked.event_id).kind, "check_recorded")

        view = env.verify(checked.runtime)
        edge = next_step(view).edge_id
        transition_input = EnvironmentStepV1(
            directive={"kind": "commit_transition", "edge_id": edge}
        )
        _assert_fault_matrix(
            self,
            env,
            checked.runtime,
            transition_input,
            root,
            event_kind="transition_committed",
        )
        transitioned = env.commit(checked.runtime, transition_input)
        self.assertEqual(store.load_event(transitioned.event_id).kind, "transition_committed")
        seal_input = EnvironmentStepV1(
            directive={
                "kind": "seal_outcome",
                "task_status": "complete",
                "stop_reason": None,
            }
        )
        _assert_fault_matrix(
            self,
            env,
            transitioned.runtime,
            seal_input,
            root,
            event_kind="termination_recorded",
        )
        sealed = env.commit(transitioned.runtime, seal_input)
        self.assertEqual(store.load_event(sealed.event_id).kind, "termination_recorded")
        reward_input = EnvironmentStepV1(directive={"kind": "publish_reward"})
        _assert_fault_matrix(
            self,
            env,
            sealed.runtime,
            reward_input,
            root,
            event_kind="reward_recorded",
        )
        rewarded = env.commit(sealed.runtime, reward_input)
        self.assertEqual(store.load_event(rewarded.event_id).kind, "reward_recorded")
        self.assertEqual(env.verify(rewarded.runtime).checkpoint_id, rewarded.runtime.checkpoint_id)

    def test_port_allowlists_and_private_canaries(self):
        fixture = _author_check_fixture()
        root = self.root / "privacy"
        root.mkdir()
        root.chmod(0o700)
        gate = LineageGate()
        store = TaskGraphStore(root / "store", verifier=gate)
        entry = _persist_fixture(store, fixture)
        env = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        runtime = env.open(entry, root / "workspace")

        node = fixture.graph.node(fixture.node_id)
        self.assertIn(CANARY, repr(node.author_packet))
        self.assertIn(CANARY, repr(node.evaluator_packet))
        private_ledger = store.get_artifact(runtime.state.requirements_ref, private=True)
        self.assertIn(CANARY, repr(private_ledger))
        view = env.verify(runtime)
        sampler = env.port_input(view, next_step(view))
        self.assertIsInstance(sampler, SamplerInput)
        self.assertNotIn(CANARY, repr(sampler))

        turn = make_turn(
            view,
            content="",
            calls=(call("read_file", {"path": "draft.txt"}, "private-safe-read"),),
        )
        writer_step = env.commit(runtime, turn)
        tool_view = env.verify(writer_step.runtime)
        tool_input = env.port_input(tool_view, writer_step.directive)
        self.assertIsInstance(tool_input, ToolInput)
        self.assertNotIn(CANARY, repr(tool_input))
        self.assertEqual(
            {field.name for field in fields(SamplerInput)},
            {
                "messages",
                "tools",
                "rendering",
                "context_revision_ref",
                "action_id",
                "usage_requirements",
                "writer_seed",
                "model_ref",
                "behavior_policy_ref",
                "tokenizer_ref",
                "template_ref",
                "decoding_ref",
            },
        )
        self.assertEqual(
            {field.name for field in fields(ToolInput)},
            {"files", "queue_entry", "tool_spec", "dispatch_permitted"},
        )
        self.assertEqual(
            {field.name for field in fields(AuthorInput)},
            {"request_ref", "request", "script", "decisions", "disclosures"},
        )
        self.assertEqual(
            {field.name for field in fields(CheckInput)},
            {"request_ref", "request", "files", "evaluator_packet"},
        )

    def test_start_member_publishes_the_sealed_member_start(self):
        spec = _group_spec(self.fixture, self.entry, self.store)
        member_start = MemberStartV1(group_spec_ref=spec.identity(), ordinal=0)
        _assert_fault_matrix(
            self,
            self.environment,
            self.runtime,
            member_start,
            self.root,
            event_kind="rollout_started",
            entry_checkpoint=self.entry,
            lineage=spec.members[0].member_id,
        )
        started = self.environment.start_member(
            self.entry,
            member_start,
            self.root / "member-workspace",
        )
        self.assertEqual(started.state.position["lineage_id"], spec.members[0].member_id)
        self.assertEqual(self.store.read_head(spec.members[0].member_id) is not None, True)
        view = self.environment.verify(started)
        sampler = self.environment.port_input(view, next_step(view))
        self.assertIsInstance(sampler, SamplerInput)
        self.assertEqual(sampler.writer_seed, spec.members[0].writer_seed)
        self.assertEqual(sampler.model_ref, spec.policy["model_ref"])

    def test_enter_derives_and_materializes_an_unpublished_root(self):
        fixture = make_entry_fixture()
        root = self.root / "enter"
        root.mkdir()
        root.chmod(0o700)
        gate = LineageGate()
        store = TaskGraphStore(root / "store", verifier=gate)
        _persist_fixture(store, fixture, checkpoint=False)
        env = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        runtime = env.enter(fixture.node_id, fixture.params, root / "workspace")
        self.assertIsNone(store.read_head(fixture.params.lineage_id))
        self.assertEqual(runtime.state, fixture.state)
        self.assertEqual(env.verify(runtime).checkpoint_id, runtime.checkpoint_id)

    def test_candidate_view_cannot_reach_a_port(self):
        view = self.environment.verify(self.runtime)
        transition = DERIVE["WriterTurnV1"](
            view, make_turn(view, content="draft"), self.environment.reader
        )
        with self.assertRaises(ProjectionError):
            self.environment.port_input(transition.view, next_step(transition.view))

    def test_adapter_contract_rejection_writes_nothing_and_gate_keeps_projection_error(self):
        view = self.environment.verify(self.runtime)
        # Entry contracts currently cannot seed token limits; isolate the producer mapping.
        limited_view = with_budget(view, self.fixture.reader, limits={"generated_tokens": 1})
        missing_usage = make_turn(limited_view, usage={})
        with patch.object(self.environment, "_verify", return_value=limited_view):
            with self.assertRaises(AdapterContractError):
                self.environment.commit(self.runtime, missing_usage)

        bad = make_turn(
            view,
            usage={},
            adapter_trace={"generated_token_ids": [7]},
        )
        old_head = self.store.read_head(view.state.position["lineage_id"])
        with self.assertRaises(AdapterContractError):
            self.environment.commit(self.runtime, bad)
        self.assertEqual(self.store.read_head(view.state.position["lineage_id"]), old_head)

        bad_ref = self.store.put_artifact(bad.to_wire())
        event = EventV1(
            previous=view.head_event_id,
            seq=view.state.history["seq"] + 1,
            lineage_id=view.state.position["lineage_id"],
            rollout_id=view.state.position["lineage_id"],
            node_visit_id=view.state.position["visit_id"],
            kind="writer_action",
            actor="writer",
            audience=("controller", "trainer"),
            payload_ref=bad_ref,
            versions_ref=view.state.versions_ref,
            provenance_ref=view.state.provenance_ref,
        )
        with self.assertRaises(ProjectionError):
            self.gate.verify_commit(self.store, view.checkpoint_id, (event,), self.runtime.state)
        self.assertEqual(self.store.read_head(view.state.position["lineage_id"]), old_head)

    def test_missing_derived_artifact_fails_publish_without_advancing_head(self):
        view = self.environment.verify(self.runtime)
        turn = make_turn(view, content="persisted turn")
        original = self.environment._persist_artifact
        head = self.store.read_head(view.state.position["lineage_id"])

        def omit_budget(artifact):
            if artifact.ref != view.state.budgets_ref and artifact.kind == "artifact":
                # Only omit the changed budget, not the turn payload.
                if artifact.value_kind == "canonical_json":
                    body = load_canonical_json(artifact.value)
                    if body.get("schema") == 1 and "consumed" in body and "limits" in body:
                        return None
            return original(artifact)

        with patch.object(self.environment, "_persist_artifact", side_effect=omit_budget):
            with self.assertRaises(MissingReferenceError):
                self.environment.commit(self.runtime, turn)
        self.assertEqual(self.store.read_head(view.state.position["lineage_id"]), head)

    def test_environment_methods_share_one_outer_store_scope(self):
        original = self.store.operation
        depth = 0
        scopes = 0

        @contextmanager
        def counted():
            nonlocal depth, scopes
            if depth == 0:
                scopes += 1
            depth += 1
            try:
                with original():
                    yield
            finally:
                depth -= 1

        spec = _group_spec(self.fixture, self.entry, self.store)
        self.store.operation = counted
        entered = self.environment.enter(
            self.fixture.node_id, self.fixture.params, self.root / "scoped-entry"
        )
        self.assertEqual(entered.checkpoint_id, self.entry)
        self.assertEqual(scopes, 1)
        reopened = self.environment.open(self.entry, self.root / "scoped-open")
        self.assertEqual(scopes, 2)
        self.environment.start_member(
            self.entry,
            MemberStartV1(group_spec_ref=spec.identity(), ordinal=0),
            self.root / "scoped-member",
        )
        self.assertEqual(scopes, 3)
        view = self.environment.verify(reopened)
        self.assertEqual(scopes, 4)
        self.environment.port_input(view, next_step(view))
        self.assertEqual(scopes, 5)
        self.environment.commit(reopened, make_turn(view, content="one scope"))
        self.assertEqual(scopes, 6)

    def test_design_event_kind_inventory(self):
        self.assertEqual(len(EVENT_KINDS), 11)
        self.assertEqual(len(set(EVENT_KINDS)), 11)


if __name__ == "__main__":
    unittest.main()
