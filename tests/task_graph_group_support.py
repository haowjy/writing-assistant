"""Shared deterministic group fixtures for focused coordinator test modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tests.task_graph_rollout_fixtures import (
    build_rollout_fixture,
    make_gatherers,
    run_slice,
)
from writing_agent.task_graph_composition import RuntimeSession
from writing_agent.task_graph_group import (
    POLICY_FIELDS,
    GroupCoordinatorV1,
    GroupMemberResultV1,
)
from writing_agent.task_graph_local import (
    DeterministicEvaluator,
    LocalTextToolProvider,
    LocalWorkspaceEnvironment,
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import (
    RuntimeDependenciesV1,
)
from writing_agent.task_graph_record_contracts import ContextPolicyV1
from writing_agent.task_graph_records import (
    DecodingDescriptorV1,
    RendererDescriptorV1,
    RuntimeManifestV2,
    RuntimePortDescriptorV1,
    TokenizerDescriptorV1,
)


class GroupCoordinatorFixtureMixin:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = build_rollout_fixture(self.root / "rollout")
        self.store = self.fixture.store
        self.env = self.fixture.env
        self.entry_id = self.fixture.runtime.checkpoint_id
        self.session = self.runtime_session(self.fixture)
        self.env.session = self.session
        self.coordinator = GroupCoordinatorV1(self.env, session=self.session)
        self.policy = self.policy_for(self.fixture, session=self.session)

    def policy_for(self, fixture, *, session=None):
        store = fixture.store
        rendering = fixture.runtime.context.rendering
        policy = {
            field: store.put_artifact({"pin": field})
            for field in POLICY_FIELDS
            if field != "rng_derivation_version"
        }
        if session is not None:
            policy["adapter_ref"] = session.manifest_ref
        policy["model_ref"] = store.put_artifact({"model_id": "group-model-v1"})
        context_policy = ContextPolicyV1(
            "compact", summarizer_version="visible-text-v1", max_summary_chars=20
        )
        policy["context_policy_ref"] = store.put_artifact(context_policy.to_dict())
        policy.update(
            tokenizer_ref=rendering["tokenizer_ref"],
            template_ref=rendering["template_ref"],
            rng_derivation_version="sha256-domain-v1",
        )
        return policy

    def group(self, sequence=0, mode="real"):
        return self.coordinator.seal(
            self.entry_id,
            policy=self.policy,
            group_seed=771,
            group_sequence=sequence,
            member_count=2,
            runner_mode=mode,
        )

    def native_manifest(self, capabilities, *, template_ref=None):
        rendering = self.fixture.runtime.context.rendering
        tokenizer = TokenizerDescriptorV1(
            model_id="tests/toy-tokenizer",
            revision="toy-r1",
            files_sha256={"tokenizer": "a" * 64},
        )
        tokenizer_ref = self.store.put_artifact(tokenizer.to_wire())
        decoding = DecodingDescriptorV1(
            temperature=1,
            top_p=1,
            top_k=0,
            processors=(),
            max_tokens_per_decision=64,
            seed_rule="writer_seed ⊕ action ordinal (sha256-domain-v1)",
            logprob_convention="log_softmax(model logits after model softcap), fp32",
            trainer_ratio="recomputed, num_iterations=1",
        )
        self.store.put_artifact(decoding.to_wire())
        renderer = RendererDescriptorV1(
            implementation="gemma4-native-append-v1",
            template_ref=template_ref or rendering["template_ref"],
            tokenizer_ref=tokenizer_ref,
            tool_schema_ref=rendering["tool_schema_ref"],
            stop_token_ids=(1, 106, 50),
            enable_thinking=False,
            suffix_rules_version="native-suffix-v1",
        )
        ports = tuple(
            RuntimePortDescriptorV1(
                schema=1,
                role=role,
                implementation=f"tests.{role.title()}",
                version="1",
                configuration={"capabilities": tuple(sorted(capabilities))}
                if role == "sampling"
                else {},
            )
            for role in ("sampling", "environment", "tools", "evaluator")
        )
        manifest = RuntimeManifestV2(
            schema=2,
            ports=ports,
            capabilities=("native_token_ledger", "sampled_logprobs", "usage_reporting"),
            renderer=renderer,
            tokenizer=tokenizer,
            decoding=decoding,
        )
        return manifest, self.store.put_artifact(manifest.to_wire())

    def seal_native(self, *, adapter_ref, policy=None, sequence=0):
        if policy is None:
            group_policy = self.policy_for(self.fixture, session=None)
        else:
            group_policy = dict(policy)
        group_policy["adapter_ref"] = adapter_ref
        return self.coordinator.seal(
            self.entry_id,
            policy=group_policy,
            group_seed=17,
            group_sequence=sequence,
            member_count=2,
            training_mode="native",
        )

    @staticmethod
    def runtime_session(fixture, *, backend=None):
        backend = backend or ScriptedSampleBackend(fixture.sample_results)
        tools = LocalTextToolProvider()
        dependencies = RuntimeDependenciesV1(
            backend,
            LocalWorkspaceEnvironment(tools),
            tools,
            DeterministicEvaluator(),
        )
        unbound = RuntimeSession.create(fixture.store, dependencies)
        return unbound.bind(fixture.store, unbound.manifest_ref)

    def start_members(self, spec):
        return tuple(
            self.coordinator.start(spec, ordinal, policy=self.policy) for ordinal in range(2)
        )

    def run_member(self, spec, ordinal):
        runtime = self.coordinator.start(spec, ordinal, policy=self.policy)
        self.fixture.gatherers = make_gatherers(self.fixture)
        runtime = run_slice(self.fixture, runtime=runtime)
        view = self.env.verify(runtime)
        outcome_ref = runtime.state.outcome_ref
        if view.outcome.reward_ref is not None:
            reward = self.store.get_artifact(view.outcome.reward_ref)
            outcome_ref = reward["terminal_outcome_ref"]
        return runtime, GroupMemberResultV1(
            group_id=spec.group_id,
            member_id=spec.members[ordinal].member_id,
            start_checkpoint_id=self.coordinator.start_receipt(spec, ordinal)[
                "start_checkpoint_id"
            ],
            final_checkpoint_id=runtime.checkpoint_id,
            terminal_outcome_ref=outcome_ref,
            availability_ref=view.outcome.reward_ref,
            execution_status="valid",
        )
