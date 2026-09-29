"""Sampling, adapter-evidence, and eligibility acceptance at producer and gate seams."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_rollout_fixtures import build_rollout_fixture
from tests.test_task_graph_rollout_env import _group_spec
from writing_agent.task_graph import canonical_bytes, domain_hash
from writing_agent.task_graph_composition import RuntimeSession
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_errors import AdapterContractError, ProjectionError
from writing_agent.task_graph_gate import derive_input
from writing_agent.task_graph_group_contract import derive_group_seed
from writing_agent.task_graph_local import (
    DeterministicEvaluator,
    LocalTextToolProvider,
    LocalWorkspaceEnvironment,
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import PortDescriptorV1, RuntimeDependenciesV1, SampleResult
from writing_agent.task_graph_record_contracts import GroupMemberSpecV1, GroupSpecV1
from writing_agent.task_graph_records import MemberStartV1


def _sample(content: str = "A complete draft.", *, usage=None, trace=None) -> SampleResult:
    return SampleResult(
        {"role": "assistant", "content": content, "tool_calls": []},
        usage=usage,
        trace=trace,
    )


def _event_count(store) -> tuple[int, int, int]:
    return tuple(
        sum(path.is_file() for path in (store.root / directory).glob("*.json"))
        for directory in ("events", "checkpoints", "commits")
    )


class SamplingAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_adapter_sampling_evidence_rejections_write_no_lineage_records(self) -> None:
        cases = (
            ("inline-logprobs", "none", _sample(trace={"per_token_logprobs": [1]})),
            (
                "negative-token-id",
                "none",
                _sample(usage={"completion_tokens": 1}, trace={"generated_token_ids": [-7]}),
            ),
            (
                "under-reported-usage",
                "token_limited",
                _sample(usage={"completion_tokens": 1}, trace={"generated_token_ids": [4, 5]}),
            ),
            ("malformed-usage", "token_limited", _sample(usage={"completion_tokens": "2"})),
            ("unbound-required-usage", "token_limited", _sample(usage={"total_tokens": 4})),
        )
        for name, mode, result in cases:
            with self.subTest(case=name):
                fixture = build_rollout_fixture(
                    self.root / name,
                    mode=mode,
                    sample_results=(result,),
                )
                before = _event_count(fixture.store)
                head = fixture.store.read_head(fixture.lineage_id)
                with self.assertRaises(AdapterContractError):
                    fixture.driver().run(fixture.runtime, max_steps=1)
                self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)
                self.assertEqual(_event_count(fixture.store), before)

    def test_malformed_and_unbound_sampling_forgery_is_projection_error_at_publish(self) -> None:
        # The forged payloads have correct content-addressed names. For malformed
        # payloads put_artifact intentionally refuses them, so this simulates a disk
        # rewrite that recomputes both the artifact envelope and its name.
        bodies = (
            (
                "inline-logprobs",
                "none",
                lambda wire: wire.__setitem__("adapter_trace", {"per_token_logprobs": [1]}),
            ),
            (
                "negative-token-id",
                "none",
                lambda wire: wire.__setitem__("adapter_trace", {"generated_token_ids": [-7]}),
            ),
            (
                "under-reported-usage",
                "none",
                lambda wire: (
                    wire.__setitem__("adapter_trace", {"generated_token_ids": [4, 5]}),
                    wire.__setitem__("usage", {"completion_tokens": 1}),
                ),
            ),
            (
                "unbound-required-usage",
                "token_limited",
                lambda wire: wire.__setitem__("usage", {}),
            ),
            (
                "malformed-usage",
                "none",
                lambda wire: wire.__setitem__("usage", {"completion_tokens": "2"}),
            ),
        )
        for name, mode, mutate in bodies:
            with self.subTest(forgery=name):
                fixture = build_rollout_fixture(self.root / f"persisted-{name}", mode=mode)
                view = fixture.env.verify(fixture.runtime)
                port = fixture.env.step_input(fixture.runtime)[2]
                sampled = fixture.gatherers.sampler.turn(port)
                # Build the event/state template from a valid sample on this exact
                # view, then replace only the hash-addressed input payload.
                honest = replace(
                    sampled,
                    usage={"completion_tokens": 2},
                    adapter_trace={"generated_token_ids": [4, 5]},
                )
                honest_transition = derive_input(view, honest, fixture.env.reader)
                for artifact in honest_transition.artifacts:
                    fixture.store.persist_artifact(artifact)
                fixture.store.persist(honest_transition.event)
                wire = honest.to_wire()
                mutate(wire)
                payload_ref = self._write_hash_correct_payload(fixture, wire)
                event = replace(honest_transition.event, payload_ref=payload_ref, id=None)
                state = replace(
                    honest_transition.state,
                    history={**honest_transition.state.history, "head": event.id},
                )
                with self.assertRaises(ProjectionError) as caught:
                    fixture.store.publish(
                        fixture.lineage_id,
                        None,
                        (event,),
                        state,
                        parent_checkpoint=fixture.runtime.checkpoint_id,
                    )
                expected_path = (
                    "event.payload_ref"
                    if name in {"inline-logprobs", "negative-token-id", "malformed-usage"}
                    else "input.usage.completion_tokens"
                )
                self.assertIn(expected_path, str(caught.exception))
                self.assertIsNone(fixture.store.read_head(fixture.lineage_id))

    @staticmethod
    def _write_hash_correct_payload(fixture, body) -> str:
        ref = domain_hash("payload", body)
        envelope = {"schema": 1, "domain": "payload", "encoding": "json", "body": body}
        path = fixture.store._artifact_path(ref, False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(envelope))
        return ref

    def test_manifest_change_after_bind_and_claim_relabelling_are_rejected(self) -> None:
        fixture = build_rollout_fixture(self.root / "manifest-session")
        tools = LocalTextToolProvider()
        backend = ScriptedSampleBackend(fixture.sample_results)
        dependencies = RuntimeDependenciesV1(
            backend,
            LocalWorkspaceEnvironment(tools),
            tools,
            DeterministicEvaluator(),
        )
        root_id = fixture.runtime.checkpoint_id
        unbound = RuntimeSession.create(fixture.store, dependencies)
        session = unbound.bind(fixture.store, unbound.manifest_ref)
        fixture.env = RolloutEnvironment(
            fixture.store,
            fixture.entry.graph,
            session,
            fixture.gate,
            fixture.entry.graph.policy,
        )
        fixture.runtime = fixture.env.open(root_id)
        backend.descriptor = PortDescriptorV1("sampling", "swapped-backend", "1")
        with self.assertRaises(AdapterContractError):
            fixture.env.verify(fixture.runtime)
        self.assertIsNone(fixture.store.read_head(fixture.lineage_id))

        # The group-start input carries a sealed RuntimeManifestV1. A sampling
        # claim that relabels it is rejected identically by producer and replay.
        fixture = build_rollout_fixture(self.root / "manifest-gate")
        root_id = fixture.runtime.checkpoint_id
        tools = LocalTextToolProvider()
        backend = ScriptedSampleBackend(fixture.sample_results)
        dependencies = RuntimeDependenciesV1(
            backend,
            LocalWorkspaceEnvironment(tools),
            tools,
            DeterministicEvaluator(),
        )
        unbound = RuntimeSession.create(fixture.store, dependencies)
        session = unbound.bind(fixture.store, unbound.manifest_ref)
        model_ref = fixture.store.put_artifact({"model_id": "pinned-model"})
        base = _group_spec(fixture.entry, root_id, fixture.store, model_ref=model_ref)
        policy = dict(base.policy)
        policy["adapter_ref"] = session.manifest_ref
        group_id = domain_hash(
            "payload",
            [
                "GroupIdV1",
                base.group_sequence,
                base.environment,
                policy,
                base.group_seed,
                base.runner_mode,
                len(base.members),
            ],
        )
        members = tuple(
            GroupMemberSpecV1(
                member_id=f"grp-{group_id[:24]}-{ordinal:02d}",
                ordinal=ordinal,
                writer_seed=derive_group_seed(base.group_seed, "writer", ordinal),
                environment_seed=derive_group_seed(base.group_seed, "environment"),
            )
            for ordinal in range(len(base.members))
        )
        spec = GroupSpecV1(
            group_id,
            base.group_sequence,
            base.group_seed,
            base.runner_mode,
            base.environment,
            policy,
            members,
        )
        group_spec_ref = fixture.store.put_artifact(spec.to_wire())
        receipt = fixture.store.root / "groups" / group_id / "spec.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_bytes(canonical_bytes(spec.to_wire()))
        member_env = RolloutEnvironment(
            fixture.store,
            fixture.entry.graph,
            session,
            fixture.gate,
            fixture.entry.graph.policy,
        )
        member_runtime = member_env.start_member(
            root_id, MemberStartV1(group_spec_ref=group_spec_ref, ordinal=0)
        )
        replay_env = RolloutEnvironment(
            fixture.store,
            fixture.entry.graph,
            None,
            fixture.gate,
            fixture.entry.graph.policy,
        )
        view = replay_env.verify(replay_env.open_head(member_runtime.state.position["lineage_id"]))
        port = replay_env.step_input(member_runtime)[2]
        sampled = fixture.gatherers.sampler.turn(port)
        changed_manifest = dict(fixture.store.get_artifact(session.manifest_ref))
        changed_manifest["ports"] = [dict(port) for port in changed_manifest["ports"]]
        changed_manifest["ports"][0]["implementation"] += "-changed"
        swapped_ref = fixture.store.put_artifact(changed_manifest)
        claims = {**sampled.adapter_trace, "adapter_ref": swapped_ref}
        forged = replace(sampled, adapter_trace=claims)
        before = _event_count(fixture.store)
        with self.assertRaises(AdapterContractError):
            replay_env.commit(member_runtime, forged)
        self.assertEqual(_event_count(fixture.store), before)

        honest = replace(
            sampled,
            adapter_trace={**sampled.adapter_trace, "adapter_ref": session.manifest_ref},
        )
        honest_transition = derive_input(view, honest, replay_env.reader)
        for artifact in honest_transition.artifacts:
            fixture.store.persist_artifact(artifact)
        fixture.store.persist(honest_transition.event)
        payload_ref = fixture.store.put_artifact(forged.to_wire())
        event = replace(honest_transition.event, payload_ref=payload_ref, id=None)
        state = replace(
            honest_transition.state,
            history={**honest_transition.state.history, "head": event.id},
        )
        head = fixture.store.read_head(member_runtime.state.position["lineage_id"])
        with self.assertRaises(ProjectionError) as caught:
            fixture.store.publish(
                member_runtime.state.position["lineage_id"],
                head,
                (event,),
                state,
            )
        self.assertIn("input.adapter_trace", str(caught.exception))
        self.assertEqual(fixture.store.read_head(member_runtime.state.position["lineage_id"]), head)


if __name__ == "__main__":
    unittest.main()
