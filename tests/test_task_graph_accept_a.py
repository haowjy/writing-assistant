"""Pipeline forgeries for category a of the transition-seam acceptance map."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from tests.task_graph_rollout_fixtures import build_rollout_fixture, make_gatherers, run_slice
from writing_agent.task_graph import tree_hash
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_errors import AdapterContractError, ProjectionError
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_records import (
    ContextOperationInputV1,
    EnvironmentStepV1,
    ToolObservationV1,
    WriterTurnV1,
)


def _call(name: str, arguments: dict[str, Any], raw_id: str = "accept-a") -> dict[str, Any]:
    return {
        "id": raw_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _writer_turn(fixture, runtime=None) -> WriterTurnV1:
    runtime = fixture.runtime if runtime is None else runtime
    view = fixture.env.verify(runtime)
    port = fixture.env.port_input(view, next_step(view))
    return make_gatherers(fixture).sampler.turn(port)


def _copy_immutables(source: Path, destination: Path) -> None:
    """Copy a donor's immutable candidate objects without copying its authority ref."""
    for name in (
        "artifacts",
        "bytes",
        "checkpoints",
        "commits",
        "context_content",
        "context_revisions",
        "events",
        "instances",
        "private",
    ):
        source_dir = source / name
        if not source_dir.exists():
            continue
        target_dir = destination / name
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, target_dir / item.name)


def _writer_candidate(root: Path):
    donor = build_rollout_fixture(root / "donor")
    target = build_rollout_fixture(root / "target")
    parent_id = donor.runtime.checkpoint_id
    turn = _writer_turn(donor)
    result = donor.env.commit(donor.runtime, turn)
    event = donor.store.load_event(result.event_id)
    _copy_immutables(donor.store.root, target.store.root)
    return target, parent_id, event, result.runtime.state


def _publish_rejected(test, target, parent_id, event, state, path: str) -> None:
    with test.assertRaises(ProjectionError) as rejected:
        target.store.publish(
            target.lineage_id,
            None,
            (event,),
            state,
            parent_checkpoint=parent_id,
        )
    test.assertIn(path, str(rejected.exception))
    test.assertIsNone(target.store.read_head(target.lineage_id))


class WriterAndToolInputForgeryTests(unittest.TestCase):
    def test_writer_turn_inputs_reject_active_identity_and_sampling_binding_forgeries(self):
        cases = (
            ("action_id", ProjectionError),
            ("context_revision_ref", ProjectionError),
            ("request_ref", AdapterContractError),
            ("prepared_request_ref", ProjectionError),
        )
        for field, error_type in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = build_rollout_fixture(Path(directory) / "store")
                turn = _writer_turn(fixture)
                forged = replace(turn, **{field: "0" * 64})
                with self.assertRaises(error_type):
                    fixture.env.commit(fixture.runtime, forged)
                self.assertIsNone(fixture.store.read_head(fixture.lineage_id))

    def test_writer_directive_and_context_operation_inputs_are_rejected(self):
        cases = (
            EnvironmentStepV1(directive={"kind": "request_checks"}),
            ContextOperationInputV1(policy_ref="0" * 64),
        )
        for input_record in cases:
            with self.subTest(record=input_record.RECORD_TYPE), tempfile.TemporaryDirectory() as d:
                fixture = build_rollout_fixture(Path(d) / "store")
                with self.assertRaises(ProjectionError):
                    fixture.env.commit(fixture.runtime, input_record)
                self.assertIsNone(fixture.store.read_head(fixture.lineage_id))

    def test_tool_observation_inputs_reject_queue_and_effect_forgeries(self):
        def change_dispatch(record: ToolObservationV1, **changes) -> ToolObservationV1:
            dispatch = dict(record.dispatch or {})
            for name, value in changes.items():
                dispatch[name] = value
            return replace(record, dispatch=dispatch)

        cases = (
            ("call_id", ProjectionError, None),
            ("missing_dispatch", ProjectionError, None),
            ("wrong_spec", ProjectionError, None),
            ("fake_delta", AdapterContractError, None),
            ("false_success_with_delta", AdapterContractError, None),
            ("read_delete", AdapterContractError, None),
        )
        for case, error_type, _path in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                options = {}
                if case == "read_delete":
                    options["sample_results"] = (
                        SampleResult(
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [_call("read_file", {"path": "draft.txt"})],
                            }
                        ),
                    )
                fixture = build_rollout_fixture(Path(directory) / "store", **options)
                runtime = run_slice(fixture, until=lambda step: step.kind == "execute_tool")
                view = fixture.env.verify(runtime)
                port = fixture.env.port_input(view, next_step(view))
                valid = fixture.gatherers.tools.observe(port)
                published_before = fixture.store.read_head(fixture.lineage_id)
                dispatch = dict(valid.dispatch)
                effect = dict(dispatch["effect"])
                observation = dict(dispatch["observation"])
                if case == "call_id":
                    forged = replace(valid, call_id="unrelated-call")
                elif case == "missing_dispatch":
                    forged = replace(valid, dispatch=None)
                elif case == "wrong_spec":
                    spec = {**dispatch["spec"], "max_file_bytes": 1}
                    forged = change_dispatch(valid, spec=spec)
                elif case == "fake_delta":
                    effect["unrelated.txt"] = {"before": None, "after": "forged"}
                    forged = change_dispatch(valid, effect=effect)
                elif case == "false_success_with_delta":
                    observation["ok"] = False
                    dispatch["observation"] = observation
                    forged = replace(
                        valid,
                        dispatch={
                            **dispatch,
                            "effect": {"forged.txt": {"before": None, "after": "x"}},
                        },
                    )
                else:
                    dispatch["effect"] = {
                        "draft.txt": {"before": port.files["draft.txt"], "after": None}
                    }
                    forged = replace(valid, dispatch=dispatch)
                with self.assertRaises(error_type) as rejected:
                    fixture.env.commit(runtime, forged)
                if _path is not None:
                    self.assertIn(_path, str(rejected.exception))
                self.assertEqual(fixture.store.read_head(fixture.lineage_id), published_before)

    @unittest.expectedFailure
    def test_forged_read_observation_is_rejected(self):
        """Finding S5.1-A-OBS-1: a forged result currently reaches the published input."""
        with tempfile.TemporaryDirectory() as directory:
            read_sample = SampleResult(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_call("read_file", {"path": "draft.txt"})],
                }
            )
            fixture = build_rollout_fixture(
                Path(directory) / "store", sample_results=(read_sample,)
            )
            runtime = run_slice(fixture, until=lambda step: step.kind == "execute_tool")
            view = fixture.env.verify(runtime)
            valid = fixture.gatherers.tools.observe(fixture.env.port_input(view, next_step(view)))
            dispatch = dict(valid.dispatch)
            dispatch["observation"] = {**dispatch["observation"], "result": "forged read"}
            forged = replace(valid, dispatch=dispatch)

            published_before = fixture.store.read_head(fixture.lineage_id)
            caught = None
            try:
                fixture.env.commit(runtime, forged)
            except Exception as exc:  # noqa: BLE001 - retain the observed class for the finding.
                caught = exc
            self.assertIs(type(caught), ProjectionError)
            self.assertEqual(fixture.store.read_head(fixture.lineage_id), published_before)


class PublishedEventAndStateForgeryTests(unittest.TestCase):
    def test_derived_state_forgery_matrix_rejects_at_first_state_path(self):
        def position_phase(state):
            position = {**state.position, "phase": "checking"}
            return replace(state, position=position)

        def continuation_cursor(state):
            continuation = {**state.continuation, "next_call": state.continuation["next_call"] + 1}
            return replace(state, continuation=continuation)

        def action_history(state):
            history = {**state.history, "action_ids": ("forged:action:99",)}
            return replace(state, history=history)

        def files(state):
            changed = {**state.files, "unattributed.txt": "forged"}
            return replace(state, files=changed, tree_hash=tree_hash(changed))

        def changed_position(state, field, value):
            position = {**state.position, field: value}
            return replace(state, position=position)

        def changed_continuation(state, field, value):
            continuation = state.to_dict()["continuation"]
            if field == "tool_queue":
                continuation[field] = []
            else:
                continuation[field] = value
            return replace(state, continuation=continuation)

        def changed_history(state, field, value):
            history = {**state.history, field: value}
            return replace(state, history=history)

        def changed_artifact_ref(target, state, field, *, private=False):
            ref = target.store.put_artifact({"forged_state_field": field}, private=private)
            return replace(state, **{field: ref})

        def changed_entry_contract(target, state):
            return changed_position(state, "entry_contract", state.provenance_ref)

        def changed_start_checkpoint(target, state):
            return changed_position(state, "start_checkpoint", target.runtime.checkpoint_id)

        def changed_instance(target, state):
            instance = target.entry.graph.instance
            modified = replace(instance, budgets={**instance.budgets, "forged-budget": 1})
            return replace(state, instance_ref=target.store.persist(modified))

        def changed_versions(target, state):
            versions = dict(target.store.get_artifact(state.versions_ref))
            tool_spec = dict(versions["tool_spec"])
            tool_spec["max_file_bytes"] += 1
            versions["tool_spec"] = tool_spec
            return replace(state, versions_ref=target.store.put_artifact(versions))

        cases = [
            ("phase", "state.position.phase", lambda target, state: position_phase(state)),
            (
                "node_id",
                "state.position.node_id",
                lambda target, state: changed_position(state, "node_id", "forged-node"),
            ),
            (
                "visit_id",
                "state.position.visit_id",
                lambda target, state: changed_position(state, "visit_id", "forged-visit"),
            ),
            ("entry_contract", "state.position.entry_contract", changed_entry_contract),
            ("start_checkpoint", "state.position.start_checkpoint", changed_start_checkpoint),
            (
                "instance_ref",
                "state.instance_ref",
                changed_instance,
            ),
            (
                "loop_counts",
                "state.position.loop_counts['forged-loop']",
                lambda target, state: changed_position(state, "loop_counts", {"forged-loop": 1}),
            ),
            (
                "cursor",
                "state.continuation.next_call",
                lambda target, state: continuation_cursor(state),
            ),
            (
                "tool_queue",
                "state.continuation.tool_queue[0]",
                lambda target, state: changed_continuation(state, "tool_queue", []),
            ),
            (
                "author_request",
                "state.continuation.author_request",
                lambda target, state: changed_continuation(
                    state, "author_request", state.author_packet_ref
                ),
            ),
            (
                "check_requests",
                "state.continuation.check_requests[0]",
                lambda target, state: changed_continuation(
                    state, "check_requests", (state.author_packet_ref,)
                ),
            ),
            (
                "external_requests",
                "state.continuation.external_requests[0]",
                lambda target, state: changed_continuation(
                    state, "external_requests", ("forged-request",)
                ),
            ),
            (
                "applied_responses",
                "state.continuation.applied_responses[0]",
                lambda target, state: changed_continuation(
                    state, "applied_responses", ("forged-response",)
                ),
            ),
            (
                "feedback_cursor",
                "state.continuation.feedback_cursor",
                lambda target, state: changed_continuation(state, "feedback_cursor", 1),
            ),
            (
                "action_history",
                "state.history.action_ids[0]",
                lambda target, state: action_history(state),
            ),
            (
                "tool_result_history",
                "state.history.tool_result_ids[0]",
                lambda target, state: changed_history(
                    state, "tool_result_ids", ("forged:tool_result:99",)
                ),
            ),
            (
                "requirements_ref",
                "state.requirements_ref",
                lambda target, state: changed_artifact_ref(
                    target, state, "requirements_ref", private=True
                ),
            ),
            (
                "context_ref",
                "state.context_ref",
                lambda target, state: replace(state, context_ref=target.runtime.state.context_ref),
            ),
            (
                "decisions_ref",
                "state.decisions_ref",
                lambda target, state: changed_artifact_ref(target, state, "decisions_ref"),
            ),
            (
                "disclosures_ref",
                "state.disclosures_ref",
                lambda target, state: changed_artifact_ref(target, state, "disclosures_ref"),
            ),
            (
                "author_packet_ref",
                "state.author_packet_ref",
                lambda target, state: changed_artifact_ref(
                    target, state, "author_packet_ref", private=True
                ),
            ),
            (
                "budgets_ref",
                "state.budgets_ref",
                lambda target, state: changed_budget(target, state),
            ),
            (
                "rng_ref",
                "state.rng_ref",
                lambda target, state: changed_artifact_ref(target, state, "rng_ref"),
            ),
            (
                "external_inputs_ref",
                "state.external_inputs_ref",
                lambda target, state: changed_artifact_ref(target, state, "external_inputs_ref"),
            ),
            ("versions_ref", "state.versions_ref", changed_versions),
            (
                "outcome_ref",
                "state.outcome_ref",
                lambda target, state: changed_artifact_ref(target, state, "outcome_ref"),
            ),
            (
                "provenance_ref",
                "state.provenance_ref",
                lambda target, state: changed_artifact_ref(target, state, "provenance_ref"),
            ),
            ("files", "state.files['unattributed.txt']", lambda target, state: files(state)),
        ]

        def changed_budget(target, state):
            budget = dict(target.store.get_artifact(state.budgets_ref))
            consumed = dict(budget["consumed"])
            consumed["writer_turns"] = consumed.get("writer_turns", 0) + 1
            budget["consumed"] = consumed
            return replace(state, budgets_ref=target.store.put_artifact(budget))

        for name, path, mutate in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                target, parent_id, event, state = _writer_candidate(Path(directory))
                _publish_rejected(self, target, parent_id, event, mutate(target, state), path)

    def test_derived_event_forgery_matrix_rejects_at_first_event_path(self):
        mutations = (
            ("kind", "event.kind", {"kind": "tool_result"}),
            ("actor", "event.actor", {"actor": "author"}),
            ("audience", "event.audience[1]", {"audience": ("controller",)}),
        )
        for name, path, changes in mutations:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                target, parent_id, event, state = _writer_candidate(Path(directory))
                forged = replace(event, id=None, **changes)
                forged_state = replace(
                    state,
                    history={**state.history, "head": forged.id},
                )
                _publish_rejected(self, target, parent_id, forged, forged_state, path)

    def test_author_evaluator_and_terminal_inputs_reject_semantic_mismatches(self):
        cases = ("author_reply", "evaluator_result", "terminal_step")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = build_rollout_fixture(Path(directory) / "store")
                if case == "author_reply":
                    runtime = run_slice(
                        fixture, until=lambda step: step.kind == "await_author_reply"
                    )
                    view = fixture.env.verify(runtime)
                    port = fixture.env.port_input(view, next_step(view))
                    reply = fixture.gatherers.author.reply(port)
                    inputs = (
                        replace(reply, utterance="The author script never said this."),
                        replace(reply, request_ref=view.state.author_packet_ref),
                    )
                elif case == "evaluator_result":
                    runtime = run_slice(
                        fixture, until=lambda step: step.kind == "await_check_result"
                    )
                    view = fixture.env.verify(runtime)
                    port = fixture.env.port_input(view, next_step(view))
                    result = fixture.gatherers.evaluator.result(port)
                    inputs = (
                        replace(result, status="fail" if result.status == "pass" else "pass"),
                        replace(result, request_ref=view.state.author_packet_ref),
                        replace(result, evidence_ref=view.state.provenance_ref),
                    )
                else:
                    runtime = run_slice(
                        fixture, until=lambda step: step.kind == "commit_transition"
                    )
                    directive = next_step(fixture.env.verify(runtime))
                    correct = EnvironmentStepV1.of(directive)
                    inputs = (
                        EnvironmentStepV1(
                            directive={**correct.directive, "edge_id": "forged-edge"}
                        ),
                    )
                for index, forged in enumerate(inputs):
                    with self.subTest(forgery=case, index=index):
                        previous_head = fixture.store.read_head(fixture.lineage_id)
                        with self.assertRaises(ProjectionError):
                            fixture.env.commit(runtime, forged)
                        self.assertEqual(fixture.store.read_head(fixture.lineage_id), previous_head)


if __name__ == "__main__":
    unittest.main()
