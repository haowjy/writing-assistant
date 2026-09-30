"""Shared fixture helpers for rollout-environment contract tests."""

from __future__ import annotations

import shutil
from dataclasses import replace
from unittest.mock import patch

from tests.task_graph_fixtures import make_outcome_fixture
from tests.test_task_graph_derive_author import _answer_rule
from writing_agent.task_graph import domain_hash
from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_contracts import (
    AuthorPacketV1,
    DecisionBindingsV1,
    InteractionContractV1,
    InteractionPolicyV1,
    ScriptedAuthorV1,
)
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_group import POLICY_FIELDS, derive_group_seed
from writing_agent.task_graph_record_contracts import GroupMemberSpecV1, GroupSpecV1
from writing_agent.task_graph_records import ContextContentV1, ContextRevisionV1
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_transition import DerivedArtifact

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
        store.persist_artifact(artifact)
    if checkpoint:
        return store.save_checkpoint(fixture.state)
    return None


def _foreign_context_revision(store, view):
    content = ContextContentV1(
        parent_ref=None,
        messages=view.context.messages,
        tools=view.context.tools,
        rendering={**view.context.rendering, "prefix_id": "other-context"},
    )
    revision = ContextRevisionV1(content.identity(), None, ())
    store.persist_artifact(DerivedArtifact(content.identity(), content, "context_node", "record"))
    store.persist_artifact(
        DerivedArtifact(revision.identity(), revision, "context_revision", "record")
    )
    return revision.identity()


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


def _group_spec(fixture, entry_checkpoint, store, *, model_ref=None, adapter_ref=None):
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
    if model_ref is not None:
        policy["model_ref"] = model_ref
    if adapter_ref is not None:
        policy["adapter_ref"] = adapter_ref
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
            cloned_runtime = type(runtime)(runtime.checkpoint_id, runtime.state, runtime.context)
            old_head = store.read_head(lineage)
            publish = store.publish

            def crash(
                *args,
                _stage=stage,
                _publish=publish,
                **kwargs,
            ):
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
                            )
                head = store.read_head(lineage)
                if stage == "after_head_publication":
                    test.assertNotEqual(head, old_head)
                else:
                    test.assertEqual(head, old_head)
                    test.assertEqual(
                        env.verify(cloned_runtime).checkpoint_id, runtime.checkpoint_id
                    )

                if stage == "after_head_publication":
                    # Recreate the store and gate to exercise the cold retry path.
                    retry_gate = LineageGate()
                    retry_store = TaskGraphStore(clone_root / "store", verifier=retry_gate)
                    retry_env = RolloutEnvironment(
                        retry_store,
                        environment.graph,
                        None,
                        retry_gate,
                        environment.admission_policy,
                    )
                    if entry_checkpoint is None:
                        commit_id = retry_env.commit(cloned_runtime, input_record).commit_id
                    else:
                        started = retry_env.start_member(entry_checkpoint, input_record)
                        commit_id = retry_store.read_head(lineage)
                        test.assertEqual(
                            started.checkpoint_id,
                            retry_store.load_commit(commit_id).checkpoint,
                        )
                    test.assertEqual(commit_id, head)
                elif entry_checkpoint is None:
                    commit_id = env.commit(cloned_runtime, input_record).commit_id
                else:
                    env.start_member(entry_checkpoint, input_record)
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
                    opened = retry_env.open(commit.checkpoint)
                    test.assertEqual(retry_env.verify(opened).checkpoint_id, commit.checkpoint)
        finally:
            shutil.rmtree(clone_root, ignore_errors=True)
    test.assertEqual(len(set(retry_ids)), 1)
