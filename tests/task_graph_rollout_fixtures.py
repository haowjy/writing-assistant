"""Shared fixture API for the rollout, forgery, privacy, and group lanes.

``build_rollout_fixture(root, *, mode="slice", raising_ports=False, sample_results=None,
session=None)`` creates a verified entry, store, gate, environment, and default gatherers.
Modes are ``slice`` (writer/tool/author/check), ``none`` (writer-only reward),
``feedback`` (mandatory feedback plus requirement supersession), ``token_limited`` (a real
generated-token budget), and ``halt`` (no admitted evaluation). The fixture exposes
``entry``, ``store``, ``gate``, ``env``, ``runtime``, ``lineage_id``, ``counter``,
``sampler_inputs``, ordered ``checkpoint_ids``, and a composable ``commit_observer`` hook.

``run_slice(fixture, *, alternatives=None, runtime=None, until=None)`` returns the handle
before the first directive matched by ``until`` or the final handle. ``make_gatherers``
accepts optional ``sampler``, ``tools``, ``author``, and ``evaluator`` port replacements.
``ports_disabled()`` raises if offline replay reaches any producer port.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.task_graph_fixtures import EntryFixture, make_entry_fixture, make_outcome_fixture
from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_contracts import (
    AuthorPacketV1,
    DecisionBindingsV1,
    InteractionContractV1,
    InteractionPolicyV1,
    RequirementUpdateV1,
    ScriptedAuthorV1,
)
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_environment import RolloutEnvironment, RuntimeHandle
from writing_agent.task_graph_errors import DriverBudgetError
from writing_agent.task_graph_evaluation import FAMILIES
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_gatherers import (
    CheckRunner,
    Gatherers,
    SamplingRunner,
    ScriptedAuthorSource,
    ToolRunner,
)
from writing_agent.task_graph_local import (
    DeterministicEvaluator,
    LocalTextToolProvider,
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_rollout import RolloutDriver
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
    def __init__(self, counter: PortCallCounter, backend) -> None:
        self.counter = counter
        self.backend = backend
        self.prepared_inputs = []

    def sample(self, prepared):
        self.counter.called("sampling")
        self.prepared_inputs.append(prepared)
        return self.backend.sample(prepared)


class _ToolPort:
    def __init__(self, counter: PortCallCounter, provider) -> None:
        self.counter = counter
        self.provider = provider

    def execute(self, spec, snapshot, action):
        self.counter.called("tools")
        return self.provider.execute(spec, snapshot, action)


class _EvaluatorPort:
    def __init__(self, counter: PortCallCounter, evaluator) -> None:
        self.counter = counter
        self.evaluator = evaluator
        self.family = evaluator.family

    def evaluate(self, request):
        self.counter.called("evaluator")
        return self.evaluator.evaluate(request)


class _AuthorPort:
    def __init__(self, counter: PortCallCounter, source) -> None:
        self.counter = counter
        self.source = source

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
    gatherers: Gatherers | None
    counter: PortCallCounter
    canaries: dict[str, str]
    sample_results: tuple[SampleResult, ...]
    root: Path
    lineage_id: str
    checkpoint_ids: list[str]
    sampler_inputs: list[Any]
    commit_observer: Any = None

    def driver(self, *, alternatives=None) -> RolloutDriver:
        if alternatives is None:
            assert self.gatherers is not None
            return RolloutDriver(self.env, self.gatherers)
        assert self.gatherers is not None
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

    def observe_commit(self, result) -> None:
        self.checkpoint_ids.append(result.runtime.checkpoint_id)


def _entry_fixture(mode: str, evaluator_family: str) -> EntryFixture:
    if mode == "halt":
        return make_entry_fixture()
    fixture = make_outcome_fixture()
    node = fixture.graph.node(fixture.node_id)
    base_check = node.checks["nonempty"]
    family = FAMILIES[evaluator_family]
    check = (
        base_check
        if evaluator_family == "deterministic-file-v1"
        else replace(
            base_check,
            evaluator_version=family.check_version,
            spec={
                "id": base_check.id,
                "metric": "Q1",
                "kind": family.program_kind,
                "method": family.program_method,
                "required": True,
            },
        )
    )
    canary_check = replace(
        base_check,
        spec={**base_check.spec, "private_fixture_canary": EVALUATOR_PACKET_CANARY},
    )
    packet = replace(node.evaluator_packet, check_ids=(check.id,))
    contract = (
        node.contract
        if evaluator_family == "deterministic-file-v1"
        else replace(node.contract, mandatory_checks=(check.identity(),))
    )
    fixture.reader.private.update(
        {
            check.identity(): check.to_dict(),
            canary_check.identity(): canary_check.to_dict(),
            packet.identity(): packet.to_dict(),
        }
    )
    entry_contract = contract.entry_contract
    requirement_ref = None
    if mode in {"slice", "feedback"}:
        requirement_ref = fixture.reader.add(
            {"requirements": {"baseline": LEDGER_CANARY}}, private=True
        )
        entry_contract = replace(entry_contract, requirement_version=requirement_ref)
    interaction = contract.interaction_contract
    budgets = contract.budget_contract
    if mode in {"slice", "feedback"}:
        author_packet = AuthorPacketV1(
            preferences={"choice_key_91": "amber", "private_key_91": AUTHOR_PACKET_CANARY},
            requirements={},
        )
        entry_contract = replace(
            entry_contract,
            tool_allowlist=(*entry_contract.tool_allowlist, "ask_author"),
        )
        feedback = ()
        if mode == "feedback":
            update = RequirementUpdateV1(
                id="revised", supersedes="baseline", replacement="Revise for " + LEDGER_CANARY
            )
            fixture.reader.private[update.identity()] = update.to_dict()
            feedback = (
                {
                    "id": "feedback-1",
                    "utterance": "Revise the draft to satisfy the updated requirement.",
                    "prerequisite_check_ids": [],
                    "requirement_update_ref": update.identity(),
                },
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
            },
            feedback=feedback,
        )
        policy = InteractionPolicyV1(
            public_decisions=({"id": "door", "label": "door color"},),
            mandatory_feedback=tuple(item["id"] for item in feedback),
        )
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
            mandatory_feedback=tuple(item["id"] for item in feedback),
        )
        entry = replace(
            contract,
            entry=entry_contract,
            interaction=interaction,
            budgets=replace(budgets, max_author_calls=2),
        )
    elif mode == "token_limited":
        entry = replace(
            contract,
            budgets=replace(budgets, max_generated_tokens=100),
        )
    else:
        entry = replace(contract, entry=entry_contract)
    fixture.reader.public[entry.identity()] = entry.to_dict()
    spec = replace(node.spec, entry_contract=entry.identity())
    graph = admit_graph(
        replace(fixture.graph.instance, nodes=(spec,)),
        MappingArtifactResolver(fixture.reader.public, fixture.reader.private),
        policy=fixture.graph.policy,
    )
    derived = derive_entry(graph, fixture.node_id, fixture.params, fixture.reader)
    return replace(fixture, graph=graph, state=derived.state, artifacts=derived.artifacts)


def _persist_entry(store: TaskGraphStore, fixture: EntryFixture) -> str:
    for body in fixture.reader.public.values():
        store.put_artifact(body)
    for body in fixture.reader.private.values():
        store.put_artifact(body, private=True)
    store.persist(fixture.graph.instance)
    for artifact in fixture.artifacts:
        store.persist_artifact(artifact)
    return store.save_checkpoint(fixture.state)


def _gatherers(
    store: TaskGraphStore,
    entry: EntryFixture,
    samples: tuple[SampleResult, ...],
    counter: PortCallCounter,
    *,
    sampler=None,
    tools=None,
    author=None,
    evaluator=None,
    sampler_inputs=None,
) -> Gatherers:
    evaluator = _EvaluatorPort(counter, evaluator or DeterministicEvaluator())
    checks = {
        check.identity(): check
        for node in entry.graph.nodes.values()
        for check in node.checks.values()
    }
    return Gatherers(
        SamplingRunner(
            store,
            _SamplePort(counter, sampler or ScriptedSampleBackend(samples)),
            sampler_inputs.append if sampler_inputs is not None else None,
        ),
        ToolRunner(_ToolPort(counter, tools or LocalTextToolProvider())),
        _AuthorPort(counter, author or ScriptedAuthorSource()),
        CheckRunner(store, evaluator, checks),
    )


def make_gatherers(
    fixture: RolloutFixture, *, sampler=None, tools=None, author=None, evaluator=None
) -> Gatherers:
    """Build public fixture gatherers with optional port substitutions."""
    return _gatherers(
        fixture.store,
        fixture.entry,
        fixture.sample_results,
        fixture.counter,
        sampler=sampler,
        tools=tools,
        author=author,
        evaluator=evaluator,
        sampler_inputs=fixture.sampler_inputs,
    )


def build_rollout_fixture(
    root: Path,
    *,
    mode: str = "slice",
    raising_ports: bool = False,
    sample_results: tuple[SampleResult, ...] | None = None,
    evaluator_family: str = "deterministic-file-v1",
    session=None,
) -> RolloutFixture:
    """Build the store, fresh gate/environment, runtime and scripted/raising ports."""
    if mode not in {"slice", "none", "feedback", "token_limited", "halt"}:
        raise ValueError("unsupported rollout fixture mode")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    if evaluator_family not in FAMILIES:
        raise ValueError("unsupported evaluator family")
    entry = _entry_fixture(mode, evaluator_family)
    if sample_results is None:
        sample_results = (
            _scripted_samples()
            if mode in {"slice", "feedback"}
            else (
                SampleResult(
                    {"role": "assistant", "content": "A complete draft.", "tool_calls": []}
                ),
            )
        )
    gate = LineageGate()
    store = TaskGraphStore(root / "store", verifier=gate)
    checkpoint = _persist_entry(store, entry)
    env = RolloutEnvironment(store, entry.graph, session, gate, entry.graph.policy)
    runtime = env.open(checkpoint)
    counter = PortCallCounter(raising=raising_ports)
    sampler_inputs = []
    if mode == "feedback" and len(sample_results) == 4:
        sample_results += (
            SampleResult(
                {
                    "role": "assistant",
                    "content": "The final revised draft is ready.",
                    "tool_calls": [],
                }
            ),
        )
    fixture = RolloutFixture(
        entry,
        store,
        gate,
        env,
        runtime,
        None,
        counter,
        dict(CANARIES),
        sample_results,
        root,
        entry.params.lineage_id,
        [checkpoint],
        sampler_inputs,
    )
    fixture.commit_observer = fixture.observe_commit
    env.commit_observer = lambda result: fixture.commit_observer(result)
    fixture.gatherers = make_gatherers(fixture)
    return fixture


def run_slice(
    fixture: RolloutFixture, *, alternatives=None, runtime=None, until=None
) -> RuntimeHandle:
    """Run to completion or return the handle at the first matching directive."""
    current = fixture.runtime if runtime is None else runtime
    driver = fixture.driver(alternatives=alternatives)
    for _ in range(40):
        directive = next_step(fixture.env.verify(current))
        if until is not None and until(directive):
            return current
        try:
            result = driver.run(current, max_steps=1)
        except DriverBudgetError as exc:
            current = exc.runtime
        else:
            return result.runtime
    raise DriverBudgetError(40, current)


@contextmanager
def ports_disabled():
    """Fail if replay, entry opening, or any checkpoint fold invokes a producer port."""
    calls = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"offline replay called {name}")

        return fail

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(ScriptedSampleBackend, "sample", side_effect=forbidden("sample"))
        )
        stack.enter_context(
            patch.object(LocalTextToolProvider, "execute", side_effect=forbidden("tool"))
        )
        stack.enter_context(
            patch.object(
                DeterministicEvaluator,
                "evaluate",
                side_effect=forbidden("evaluator"),
            )
        )
        stack.enter_context(
            patch(
                "writing_agent.task_graph_evaluation.produce_evaluation_evidence",
                side_effect=forbidden("evaluation evidence"),
            )
        )
        stack.enter_context(
            patch.object(ScriptedAuthorSource, "reply", side_effect=forbidden("author"))
        )
        yield
        if calls:
            raise AssertionError(f"offline replay reached producer ports: {calls}")


__all__ = [
    "AUTHOR_PACKET_CANARY",
    "CANARIES",
    "EVALUATOR_PACKET_CANARY",
    "LEDGER_CANARY",
    "PortCallCounter",
    "RolloutFixture",
    "build_rollout_fixture",
    "make_gatherers",
    "ports_disabled",
    "run_slice",
]
