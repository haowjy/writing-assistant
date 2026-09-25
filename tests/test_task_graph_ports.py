"""Alternate offline adapters run through the unmodified graph control flow."""

import unittest

from writing_agent.task_graph_checks import DeterministicChecksV1
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_environment import EnvironmentTransactionService
from writing_agent.task_graph_group import GroupCoordinatorV1
from writing_agent.task_graph_group_contract import POLICY_FIELDS
from writing_agent.task_graph_local import (
    DeterministicEvaluator,
    LocalSamplingHarness,
    LocalTextToolProvider,
)
from writing_agent.task_graph_ports import PortDescriptorV1, RuntimeDependenciesV1, ToolExecution
from writing_agent.task_graph_writer import TransactionalWriterV1


class FakeSampling(LocalSamplingHarness):
    descriptor = PortDescriptorV1("sampling", "fake-offline-sampler", "1")

    def __init__(self):
        self.prepared = 0
        self.sampled = 0

    def prepared_request(self, context, payload_ref, *, verified):
        self.prepared += 1
        return super().prepared_request(context, payload_ref, verified=verified)

    def evidence(
        self,
        action_id,
        context,
        request_ref,
        prepared_request_ref,
        raw_output_ref,
        usage,
        adapter_trace,
    ):
        self.sampled += 1
        return super().evidence(
            action_id,
            context,
            request_ref,
            prepared_request_ref,
            raw_output_ref,
            usage,
            {**(adapter_trace or {}), "offline_adapter": "fake"},
        )


class FakeEnvironment(EnvironmentTransactionService):
    @property
    def descriptor(self):
        return PortDescriptorV1("environment", "fake-observed-cas", "1")

    def __init__(self, store, rollout_id, entry_checkpoint_id):
        super().__init__(store, rollout_id, entry_checkpoint_id)
        self.batches = 0

    def batch(self, runtime, *, restore_prefix):
        self.batches += 1
        return super().batch(runtime, restore_prefix=restore_prefix)


class FakeToolProvider(LocalTextToolProvider):
    descriptor = PortDescriptorV1("tools", "fake-text-observation", "1")

    def __init__(self):
        self.calls = 0

    def execute(self, files, name, arguments, **kwargs):
        self.calls += 1
        if name == "read_file":
            return ToolExecution(
                {"ok": True, "valid": True, "result": "alternate offline observation"},
                dict(files),
            )
        return super().execute(files, name, arguments, **kwargs)


class FakeEvaluator(DeterministicEvaluator):
    descriptor = PortDescriptorV1("evaluator", "fake-independent-file-check", "1")

    def __init__(self):
        self.calls = 0

    def evaluate(self, check, files):
        self.calls += 1
        return super().evaluate(check, files)


class RuntimePortsIntegrationTest(unittest.TestCase):
    def test_alternate_ports_and_content_addressed_group_manifest(self):
        from tests.test_task_graph_scripted import ScriptedFixture

        fixture = ScriptedFixture()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        sampling = FakeSampling()
        environment = FakeEnvironment(fixture.store, "rollout-1", fixture.start)
        tools = FakeToolProvider()
        evaluator = FakeEvaluator()
        deps = RuntimeDependenciesV1(sampling, environment, tools, evaluator)
        manifest = deps.manifest()
        manifest_ref = fixture.store.put_artifact(manifest.to_wire())
        self.assertEqual(manifest_ref, manifest.identity())

        writer = TransactionalWriterV1(
            fixture.store,
            fixture.writer.graph,
            "rollout-1",
            fixture.start,
            dependencies=deps,
        )
        prepared = writer.prepare_verified_messages(
            fixture.runtime,
            {"messages": [message.to_dict() for message in fixture.runtime.context.messages]},
        )
        action = writer.submit_action(
            fixture.runtime,
            fixture.action(fixture.call("read_file", {"path": "draft.txt"})),
            prepared_request_ref=prepared,
        )
        trace = fixture.store.get_artifact(
            fixture.store.get_artifact(action.record_ref)["trace_ref"]
        )
        self.assertEqual(trace["adapter_trace"]["offline_adapter"], "fake")
        self.assertFalse(trace["native_on_policy_eligible"])
        observation = writer.step_tool(action.runtime)
        self.assertEqual(
            fixture.store.get_artifact(observation.record_ref)["observation"]["result"],
            "alternate offline observation",
        )
        final = writer.submit_action(observation.runtime, fixture.action(content="Final revision."))
        checks = DeterministicChecksV1(writer)
        batch = checks.request_checks(final.runtime)
        result = checks.check_next(batch.runtime)
        self.assertEqual(
            fixture.store.get_artifact(result.runtime.state.external_inputs_ref)["entries"][-1][
                "kind"
            ],
            "check_recorded",
        )
        self.assertGreaterEqual(sampling.prepared, 2)
        self.assertEqual(sampling.sampled, 2)
        self.assertEqual(tools.calls, 1)
        self.assertEqual(evaluator.calls, 1)
        self.assertGreaterEqual(environment.batches, 4)

        policy = {
            field: fixture.store.put_artifact({"pin": field})
            for field in POLICY_FIELDS
            if field != "rng_derivation_version"
        }
        policy.update(
            adapter_ref=manifest_ref,
            tokenizer_ref=fixture.runtime.context.rendering["tokenizer_ref"],
            template_ref=fixture.runtime.context.rendering["template_ref"],
            context_policy_ref=fixture.store.put_artifact(ContextPolicyV1("carry").to_dict()),
            rng_derivation_version="sha256-domain-v1",
        )
        spec = GroupCoordinatorV1(fixture.store, fixture.root / "port-group").seal(
            fixture.start,
            policy=policy,
            group_seed=7,
            group_sequence=0,
            member_count=2,
        )
        self.assertEqual(spec.policy["adapter_ref"], manifest_ref)


if __name__ == "__main__":
    unittest.main()
