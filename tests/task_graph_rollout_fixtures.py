"""Reusable S4 fixtures: one scripted Phase 5 slice and one writer-only node.

``build_rollout_fixture`` owns a temporary store's verified entry and scripted ports;
``run_slice`` returns the final handle plus its ordered checkpoint identities. Setting
``raising_ports=True`` makes every gatherer-backed port fail on use for replay probes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tests.task_graph_fixtures import EntryFixture
from tests.test_task_graph_derive_outcome import make_outcome_fixture
from writing_agent.task_graph import load_canonical_json
from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_contracts import (
    AuthorPacketV1,
    DecisionBindingsV1,
    InteractionContractV1,
    InteractionPolicyV1,
    ScriptedAuthorV1,
)
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_gatherers import (
    CheckRunner,
    Gatherers,
    SamplingRunner,
    ScriptedAuthorSource,
    ToolRunner,
)
from writing_agent.task_graph_local import DeterministicEvaluator, LocalTextToolProvider
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_rollout import RolloutDriver
from writing_agent.task_graph_rollout_env import RolloutEnvironment, RuntimeHandle
from writing_agent.task_graph_store import TaskGraphStore

AUTHOR_PACKET_CANARY = "AUTHOR_PACKET_CANARY_S4_1937"
EVALUATOR_PACKET_CANARY = "EVALUATOR_PACKET_CANARY_S4_2841"
LEDGER_CANARY = "REQUIREMENT_LEDGER_CANARY_S4_9752"
CANARIES = {
    "author_packet": AUTHOR_PACKET_CANARY,
    "evaluator_packet": EVALUATOR_PACKET_CANARY,
    "ledger": LEDGER_CANARY,
}


def _call(name: str, arguments: dict[str, Any], raw_id: str) -> dict[str, Any]:
    return {
        "id": raw_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _scripted_samples() -> tuple[SampleResult, ...]:
    return (
        SampleResult(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call(
                        "write_file",
                        {"path": "draft.txt", "content": "A draft with a blue door."},
                        "draft-write",
                    )
                ],
            }
        ),
        SampleResult(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call(
                        "ask_author",
                        {
                            "question": "Which color should the door be?",
                            "decision_ids": ["door"],
                            "proposals": [],
                            "option_refs": [],
                        },
                        "ask-door",
                    )
                ],
            }
        ),
        SampleResult(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call(
                        "write_file",
                        {"path": "draft.txt", "content": "The amber door opens into light."},
                        "revision-write",
                    )
                ],
            }
        ),
        SampleResult(
            {"role": "assistant", "content": "The revision is ready for review.", "tool_calls": []}
        ),
    )


class PortCallCounter:
    """Count all four gatherers' port calls, optionally raising before each response."""

    def __init__(self, *, raising: bool = False) -> None:
        self.raising = raising
        self.counts = {name: 0 for name in ("sampling", "tools", "author", "evaluator")}

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def called(self, name: str) -> None:
        self.counts[name] += 1
        if self.raising:
            raise AssertionError(f"raising replay port invoked: {name}")


class _SamplePort:
    def __init__(self, counter: PortCallCounter, results: tuple[SampleResult, ...]) -> None:
        from writing_agent.task_graph_local import ScriptedSampleBackend

        self.counter = counter
        self.backend = ScriptedSampleBackend(results)
        self.prepared_inputs = []

    def sample(self, prepared):
        self.counter.called("sampling")
        self.prepared_inputs.append(prepared)
        return self.backend.sample(prepared)


class _ToolPort(LocalTextToolProvider):
    def __init__(self, counter: PortCallCounter) -> None:
        self.counter = counter

    def execute(self, spec, snapshot, action):
        self.counter.called("tools")
        return super().execute(spec, snapshot, action)


class _EvaluatorPort(DeterministicEvaluator):
    def __init__(self, counter: PortCallCounter) -> None:
        self.counter = counter

    def evaluate(self, request):
        self.counter.called("evaluator")
        return super().evaluate(request)


class _AuthorPort:
    def __init__(self, counter: PortCallCounter) -> None:
        self.counter = counter
        self.source = ScriptedAuthorSource()

    def reply(self, port):
        self.counter.called("author")
        return self.source.reply(port)


@dataclass
class RolloutFixture:
    entry: EntryFixture
    store: TaskGraphStore
    gate: LineageGate
    env: RolloutEnvironment
    runtime: RuntimeHandle
    gatherers: Gatherers
    counter: PortCallCounter
    canaries: dict[str, str]
    sample_results: tuple[SampleResult, ...]
    root: Path

    def driver(self, *, alternatives=None) -> RolloutDriver:
        if alternatives is None:
            return RolloutDriver(self.env, self.gatherers)
        return RolloutDriver(self.env, self.gatherers, alternatives)

    def recovery_gatherers(
        self,
        runtime: RuntimeHandle,
        *,
        store: TaskGraphStore | None = None,
        raising: bool = False,
        counter: PortCallCounter | None = None,
    ) -> Gatherers:
        return _gatherers(
            self.store if store is None else store,
            self.entry,
            self.sample_results[len(runtime.state.history["action_ids"]) :],
            counter or PortCallCounter(raising=raising),
        )


def _entry_fixture(mode: str) -> EntryFixture:
    fixture = make_outcome_fixture()
    node = fixture.graph.node(fixture.node_id)
    check = node.checks["nonempty"]
    check = replace(
        check,
        id=EVALUATOR_PACKET_CANARY,
        spec={**check.spec, "id": EVALUATOR_PACKET_CANARY},
    )
    reward = replace(
        node.reward_contract,
        components={EVALUATOR_PACKET_CANARY: node.reward_contract.components["nonempty"]},
    )
    packet = replace(
        node.evaluator_packet,
        reward_contract_ref=reward.identity(),
        check_ids=(EVALUATOR_PACKET_CANARY,),
    )
    fixture.reader.private.update(
        {
            check.identity(): check.to_dict(),
            reward.identity(): reward.to_dict(),
            packet.identity(): packet.to_dict(),
        }
    )
    completion = replace(
        node.contract.completion_contract,
        required_check_ids=(EVALUATOR_PACKET_CANARY,),
        evaluation_packet_ref=packet.identity(),
    )
    contract = replace(
        node.contract,
        completion=completion,
        mandatory_checks=(check.identity(),),
    )
    if mode == "slice":
        author_packet = AuthorPacketV1(
            preferences={"choice_key_91": "amber", "private_key_91": AUTHOR_PACKET_CANARY},
            requirements={"private": LEDGER_CANARY},
        )
        script = ScriptedAuthorV1(
            answers={
                "door": {
                    "mode": "fixed_answer",
                    "utterance": "Make the door amber.",
                    "value": "amber",
                    "selector": None,
                    "prerequisite_check_ids": [],
                }
            }
        )
        policy = InteractionPolicyV1(public_decisions=({"id": "door", "label": "door color"},))
        bindings = DecisionBindingsV1(bindings={"door": "choice_key_91"})
        for item in (author_packet, script, bindings):
            fixture.reader.private[item.identity()] = item.to_dict()
        fixture.reader.public[policy.identity()] = policy.to_dict()
        interaction = InteractionContractV1(
            mode="scripted_author",
            script_ref=script.identity(),
            author_packet_ref=author_packet.identity(),
            interaction_policy_ref=policy.identity(),
            decision_bindings_ref=bindings.identity(),
        )
        entry_contract = replace(
            contract.entry_contract,
            tool_allowlist=(*contract.entry_contract.tool_allowlist, "ask_author"),
        )
        budgets = replace(contract.budget_contract, max_author_calls=2)
        contract = replace(
            contract,
            entry=entry_contract,
            interaction=interaction,
            budgets=budgets,
        )
    fixture.reader.public[contract.identity()] = contract.to_dict()
    spec = replace(node.spec, entry_contract=contract.identity())
    instance = replace(fixture.graph.instance, nodes=(spec,))
    graph = admit_graph(
        instance,
        MappingArtifactResolver(fixture.reader.public, fixture.reader.private),
        policy=fixture.graph.policy,
    )
    entry = derive_entry(graph, fixture.node_id, fixture.params, fixture.reader)
    return replace(fixture, graph=graph, state=entry.state, artifacts=entry.artifacts)


def _persist_entry(store: TaskGraphStore, fixture: EntryFixture) -> str:
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
    return store.save_checkpoint(fixture.state)


def _gatherers(
    store: TaskGraphStore,
    entry: EntryFixture,
    samples: tuple[SampleResult, ...],
    counter: PortCallCounter,
) -> Gatherers:
    evaluator = _EvaluatorPort(counter)
    checks = {
        check.identity(): check
        for node in entry.graph.nodes.values()
        for check in node.checks.values()
    }
    return Gatherers(
        SamplingRunner(store, _SamplePort(counter, samples)),
        ToolRunner(_ToolPort(counter)),
        _AuthorPort(counter),
        CheckRunner(store, evaluator, checks),
    )


def build_rollout_fixture(
    root: Path,
    *,
    mode: str = "slice",
    raising_ports: bool = False,
    sample_results: tuple[SampleResult, ...] | None = None,
) -> RolloutFixture:
    """Build the store, fresh gate/environment, runtime and scripted/raising ports."""
    if mode not in {"slice", "none"}:
        raise ValueError("mode must be 'slice' or 'none'")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    entry = _entry_fixture(mode)
    if sample_results is None:
        sample_results = (
            _scripted_samples()
            if mode == "slice"
            else (
                SampleResult(
                    {"role": "assistant", "content": "A complete draft.", "tool_calls": []}
                ),
            )
        )
    gate = LineageGate()
    store = TaskGraphStore(root / "store", verifier=gate)
    checkpoint = _persist_entry(store, entry)
    env = RolloutEnvironment(store, entry.graph, None, gate, entry.graph.policy)
    runtime = env.open(checkpoint, root / "workspace")
    counter = PortCallCounter(raising=raising_ports)
    gatherers = _gatherers(store, entry, sample_results, counter)
    return RolloutFixture(
        entry,
        store,
        gate,
        env,
        runtime,
        gatherers,
        counter,
        dict(CANARIES),
        sample_results,
        root,
    )


def run_slice(
    fixture: RolloutFixture, *, max_steps: int = 40
) -> tuple[RuntimeHandle, tuple[str, ...]]:
    """Run a fixture end to end and return its final handle and ordered checkpoints."""
    checkpoint_ids = [fixture.runtime.checkpoint_id]
    commit = fixture.env.commit

    def record_commit(runtime, input_record):
        result = commit(runtime, input_record)
        checkpoint_ids.append(result.runtime.checkpoint_id)
        return result

    fixture.env.commit = record_commit
    try:
        final = fixture.driver().run(fixture.runtime, max_steps=max_steps)
    finally:
        fixture.env.commit = commit
    return final, tuple(checkpoint_ids)


__all__ = [
    "AUTHOR_PACKET_CANARY",
    "CANARIES",
    "EVALUATOR_PACKET_CANARY",
    "LEDGER_CANARY",
    "PortCallCounter",
    "RolloutFixture",
    "build_rollout_fixture",
    "run_slice",
]
