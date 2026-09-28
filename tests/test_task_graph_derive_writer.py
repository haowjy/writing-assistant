"""Rule tables for the pure writer-turn and tool-result derives."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from typing import Any

from tests.task_graph_fixtures import make_entry_fixture
from writing_agent.task_graph import CheckpointV1, canonical_bytes, domain_hash
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_derive_writer import derive_tool_result, derive_writer_turn
from writing_agent.task_graph_errors import AdapterContractError, ProjectionError
from writing_agent.task_graph_record_contracts import (
    GroupMemberSpecV1,
    GroupSpecV1,
    _group_hash,
    _group_seed,
)
from writing_agent.task_graph_records import (
    ToolObservationV1,
    WriterRequestV1,
    WriterTurnV1,
)
from writing_agent.task_graph_sampling import bind_group_sampling_claims
from writing_agent.task_graph_transition import (
    CallSource,
    CheckpointChain,
    LineageView,
    ToolSpec,
)


def make_view(fixture=None) -> tuple[Any, LineageView]:
    fixture = make_entry_fixture() if fixture is None else fixture
    from writing_agent.task_graph_derive_entry import derive_entry

    return fixture, derive_entry(
        fixture.graph, fixture.node_id, fixture.params, fixture.reader
    ).view


def make_turn(
    view: LineageView,
    *,
    content: Any = "final text",
    calls: Any = (),
    tool_calls_was_list: bool = True,
    usage: dict[str, Any] | None = None,
    adapter_trace: dict[str, Any] | None = None,
) -> WriterTurnV1:
    raw_message = {"content": content}
    if tool_calls_was_list:
        raw_message["tool_calls"] = list(calls)
    elif calls is not None:
        raw_message["tool_calls"] = calls
    return WriterTurnV1(
        action_id=f"{view.state.position['lineage_id']}:action:{len(view.state.history['action_ids'])}",
        context_revision_ref=view.context.revision_ref,
        request_ref=None,
        prepared_request_ref=None,
        raw_output_ref=None,
        usage={} if usage is None else usage,
        adapter_trace=adapter_trace,
        message=intake_message(raw_message),
    )


def call(name: str, arguments: dict[str, Any], raw_id: str = "raw-1") -> dict[str, Any]:
    return {
        "id": raw_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def with_budget(view: LineageView, reader, *, limits=None, consumed=None) -> LineageView:
    budget = json.loads(canonical_bytes(view.budget))
    budget["limits"].update(limits or {})
    budget["consumed"].update(consumed or {})
    ref = reader.add(budget)
    state = replace(view.state, budgets_ref=ref)
    checkpoint_id = CheckpointV1(state=state, event_head=state.history["head"]).identity()
    context = view.context
    return replace(
        view,
        root_checkpoint_id=checkpoint_id,
        checkpoint_id=checkpoint_id,
        state=state,
        budget=budget,
        ancestry=CheckpointChain(checkpoint_id, context),
    )


def save_artifacts(reader, transition) -> None:
    for artifact in transition.artifacts:
        value = artifact.value
        if isinstance(value, bytes):
            value = json.loads(value)
        elif hasattr(value, "to_wire"):
            value = value.to_wire()
        if artifact.kind == "artifact":
            reader.public[artifact.ref] = value
        elif artifact.kind == "private":
            reader.private[artifact.ref] = value
        elif artifact.kind == "context_node":
            reader.context_nodes[artifact.ref] = value
        elif artifact.kind == "context_revision":
            reader.context_revisions[artifact.ref] = value


def writer_output(transition) -> dict[str, Any]:
    return {
        "event": transition.event.to_dict(),
        "state": transition.state.to_dict(),
        "artifacts": [
            {
                "ref": artifact.ref,
                "kind": artifact.kind,
                "value": (
                    artifact.value.hex()
                    if isinstance(artifact.value, bytes)
                    else artifact.value.to_wire()
                    if hasattr(artifact.value, "to_wire")
                    else artifact.value
                ),
            }
            for artifact in transition.artifacts
        ],
    }


def assert_fixed_point(test: unittest.TestCase, derive, view, record, reader) -> Any:
    if isinstance(record, WriterTurnV1):
        decoded = WriterTurnV1.from_json(record.to_json())
    else:
        decoded = ToolObservationV1.from_json(record.to_json())
    before = derive(view, record, reader)
    after = derive(view, decoded, reader)
    test.assertEqual(canonical_bytes(writer_output(before)), canonical_bytes(writer_output(after)))
    return before


def producer_path(derive, view, record, reader):
    """Model the producer-boundary exception translation used by the rollout layer."""
    try:
        return derive(view, record, reader)
    except ProjectionError as exc:
        raise AdapterContractError("adapter input violates the transition contract") from exc


def make_group(view: LineageView, reader) -> tuple[GroupSpecV1, LineageView]:
    policy = {
        key: reader.add({"pin": key})
        for key in (
            "model_ref",
            "behavior_policy_ref",
            "tokenizer_ref",
            "template_ref",
            "adapter_ref",
            "decoding_ref",
            "simulator_ref",
            "context_policy_ref",
            "controller_ref",
        )
    }
    policy["tokenizer_ref"] = view.context.rendering["tokenizer_ref"]
    policy["template_ref"] = view.context.rendering["template_ref"]
    policy["rng_derivation_version"] = "sha256-domain-v1"
    reader.public[policy["model_ref"]] = {"model_id": "sealed-model"}
    fields = (
        "entry_state_hash entry_tree_hash instance_hash graph_hash node_contract_hash "
        "controller_contract_hash check_contracts_hash source_refs_hash request_refs_hash "
        "visible_prefix_hash context_messages_hash rendering_hash tool_schemas_hash budget_hash "
        "versions_hash budget_ref versions_ref decisions_ref disclosures_ref external_inputs_ref "
        "rng_ref outcome_ref provenance_ref entry_checkpoint_id context_revision_ref "
        "requirements_ref continuation_hash"
    ).split()
    environment = {key: "a" * 64 for key in fields}
    environment.update(
        node_id=view.state.position["node_id"],
        node_visit_id=view.state.position["visit_id"],
        horizon="fixture",
        author_packet_ref=None,
        reward_contract_hash=None,
        simulator_contract_hash=None,
    )
    seed = 771
    sequence = 47
    members = tuple(
        GroupMemberSpecV1(
            member_id=f"member-{index}",
            ordinal=index,
            writer_seed=_group_seed(seed, "writer", index),
            environment_seed=_group_seed(seed, "environment"),
        )
        for index in range(2)
    )
    group_id = _group_hash(["GroupIdV1", sequence, environment, policy, seed, "real", 2])
    members = tuple(
        replace(member, member_id=f"grp-{group_id[:24]}-{member.ordinal:02d}") for member in members
    )
    spec = GroupSpecV1(
        group_id=group_id,
        group_sequence=sequence,
        group_seed=seed,
        runner_mode="real",
        environment=environment,
        policy=policy,
        members=members,
    )
    state_body = view.state.to_dict()
    state_body["position"]["lineage_id"] = members[0].member_id
    state = type(view.state).from_dict(state_body)
    checkpoint_id = CheckpointV1(state=state, event_head=state.history["head"]).identity()
    member_view = replace(
        view,
        root_checkpoint_id=checkpoint_id,
        checkpoint_id=checkpoint_id,
        state=state,
        group=spec,
        ancestry=CheckpointChain(checkpoint_id, view.context),
    )
    return spec, member_view


class WriterDeriveTests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.view = make_view()
        self.reader = self.fixture.reader

    def test_negative_token_ids_are_rejected_by_writer_turn_codec(self):
        wire = make_turn(self.view).to_dict()
        wire["adapter_trace"] = {"generated_token_ids": [-1]}
        with self.assertRaises((TypeError, ValueError)):
            WriterTurnV1.from_dict(wire)

    def test_writer_turn_rule_table_and_fixed_points(self):
        valid_call = call("write_file", {"path": "draft.txt", "content": "edited\n"})
        invalid_call = {"id": "raw-bad", "type": "not-a-function", "payload": "sampled"}
        non_array = {"not": "an array"}
        rows = (
            (
                "accepted write queue",
                make_turn(self.view, content="", calls=(valid_call,)),
                "writer_action",
            ),
            (
                "accepted malformed call",
                make_turn(self.view, content="", calls=(invalid_call,)),
                "writer_action",
            ),
            (
                "empty final accepted",
                make_turn(self.view, content="  \n", calls=()),
                "writer_action",
            ),
            (
                "non-array calls accepted as invalid call",
                make_turn(self.view, content="", calls=non_array, tool_calls_was_list=False),
                "writer_action",
            ),
            (
                "overrun precedes malformed calls and empty final",
                make_turn(
                    with_budget(self.view, self.reader, limits={"generated_tokens": 1}),
                    content="  ",
                    calls=non_array,
                    tool_calls_was_list=False,
                    usage={"completion_tokens": 2},
                ),
                "budget_charged",
            ),
        )
        for name, turn, expected_kind in rows:
            with self.subTest(name=name):
                view = (
                    with_budget(self.view, self.reader, limits={"generated_tokens": 1})
                    if name == "overrun precedes malformed calls and empty final"
                    else self.view
                )
                result = assert_fixed_point(self, derive_writer_turn, view, turn, self.reader)
                self.assertEqual(result.event.kind, expected_kind)
                if name == "empty final accepted":
                    self.assertEqual(result.state.position["phase"], "checking")
                    self.assertEqual(result.view.context.messages[-1].content[0]["text"], "  \n")
                if name == "accepted malformed call":
                    parts = result.view.context.messages[-1].content
                    self.assertEqual(parts[0]["type"], "invalid_tool_call")
                    self.assertEqual(
                        parts[0]["raw"],
                        intake_message({"tool_calls": [invalid_call]}).calls[0]["value"],
                    )
                    self.assertNotIn("invalid_call", canonical_bytes(parts).decode())
                if name == "non-array calls accepted as invalid call":
                    part = result.view.context.messages[-1].content[0]
                    self.assertEqual(part["type"], "invalid_tool_call")
                    self.assertEqual(part["raw"], result.input.message.calls)
                if expected_kind == "budget_charged":
                    self.assertEqual(result.state.position["phase"], "terminal")
                    self.assertEqual(result.view.outcome.stop_reason, "generated_tokens_budget")

    def test_usage_evidence_follows_final_classification_and_token_count_is_strict(self):
        limited = with_budget(self.view, self.reader, limits={"generated_tokens": 10})
        with self.assertRaises(ProjectionError):
            derive_writer_turn(limited, make_turn(limited, content=""), self.reader)
        with self.assertRaises(AdapterContractError):
            producer_path(derive_writer_turn, limited, make_turn(limited, content=""), self.reader)
        mismatched = make_turn(
            self.view,
            usage={"completion_tokens": 1},
            adapter_trace={"generated_token_ids": [11, 12]},
        )
        with self.assertRaises(ProjectionError):
            derive_writer_turn(self.view, mismatched, self.reader)
        with self.assertRaises(AdapterContractError):
            producer_path(derive_writer_turn, self.view, mismatched, self.reader)

    def test_invalid_call_keeps_sampled_value_in_trainable_part(self):
        raw = {"id": "raw-fail", "type": "wrong", "value": {"private": "sampled"}}
        result = derive_writer_turn(
            self.view, make_turn(self.view, content="", calls=(raw,)), self.reader
        )
        message = result.view.context.messages[-1]
        self.assertTrue(message.loss_eligible)
        part = message.content[0]
        self.assertEqual(
            part, {"type": "invalid_tool_call", "id": "rollout-fixture:call:0:0", "raw": raw}
        )
        self.assertNotIn("invalid_call", canonical_bytes(message).decode())
        self.assertEqual(result.state.continuation["tool_queue"][0]["name"], "invalid_call")
        self.assertEqual(
            result.state.continuation["tool_queue"][0]["rejection"],
            "Invalid tool call envelope",
        )

    def test_missing_sampling_context_and_eligibility_claims_fail_closed(self):
        with self.assertRaises(ProjectionError):
            derive_writer_turn(
                self.view,
                replace(make_turn(self.view), context_revision_ref="f" * 64),
                self.reader,
            )
        with self.assertRaises(ValueError):
            make_turn(self.view, adapter_trace={"native_on_policy_eligible": False})

    def test_group_bindings_use_sealed_seed_model_and_policy_not_trace_expectations(self):
        spec, view = make_group(self.view, self.reader)
        base = {
            "seed": spec.members[0].writer_seed,
            "model": "sealed-model",
            "model_ref": spec.policy["model_ref"],
            "behavior_policy_ref": spec.policy["behavior_policy_ref"],
        }
        good = make_turn(view, adapter_trace=base)
        good_result = assert_fixed_point(self, derive_writer_turn, view, good, self.reader)
        self.assertEqual(good_result.event.kind, "writer_action")
        drift_rows = (
            ("seed", {**base, "seed": spec.members[0].writer_seed + 1}),
            ("model", {**base, "model": "other-model"}),
            (
                "policy",
                {**base, "behavior_policy_ref": domain_hash("payload", {"other": "policy"})},
            ),
            (
                "context",
                {
                    **base,
                    "context_revision_ref": "b" * 64,
                    "context_content_hash": view.context.content_ref,
                    "rendering": dict(view.context.rendering),
                },
            ),
        )
        for name, claims in drift_rows:
            with self.subTest(name=name):
                if name == "context":
                    with self.assertRaises(ProjectionError):
                        derive_writer_turn(view, make_turn(view, adapter_trace=claims), self.reader)
                else:
                    with self.assertRaises(ProjectionError):
                        derive_writer_turn(view, make_turn(view, adapter_trace=claims), self.reader)

    def test_group_binding_checks_present_policy_seed_and_model_claims(self):
        policy = {"behavior_policy_ref": "a" * 64}
        claims = {"behavior_policy_ref": policy["behavior_policy_ref"], "seed": 4}
        bind_group_sampling_claims(policy, 4, claims, model_id="model-v1")
        for changed in (
            {**claims, "behavior_policy_ref": "b" * 64},
            {**claims, "seed": 5},
            {**claims, "model": "different"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ProjectionError):
                bind_group_sampling_claims(policy, 4, changed, model_id="model-v1")


class ToolResultDeriveTests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.root_view = make_view()
        self.reader = self.fixture.reader

    def _action(self, raw_call, *, view=None, content=""):
        view = self.root_view if view is None else view
        turn = make_turn(view, content=content, calls=(raw_call,))
        action = assert_fixed_point(self, derive_writer_turn, view, turn, self.reader)
        save_artifacts(self.reader, action)
        return action

    def _observation(self, action, *, observation, effect=None, dispatch=True, spec=None):
        queued = action.state.continuation["tool_queue"][0]
        value = None
        if dispatch:
            value = {
                "spec": spec
                or {
                    "max_file_bytes": action.view.tool_spec.max_file_bytes,
                    "max_workspace_bytes": action.view.tool_spec.max_workspace_bytes,
                },
                "observation": observation,
                "effect": effect or {},
            }
        return ToolObservationV1(call_id=queued["call_id"], dispatch=value)

    def test_reused_prior_raw_call_id_is_rejected(self):
        raw_id = "reused-raw-id"
        first = self._action(call("unavailable_tool", {}, raw_id))
        rejected = derive_tool_result(
            first.view,
            ToolObservationV1(
                call_id=first.state.continuation["tool_queue"][0]["call_id"],
                dispatch=None,
            ),
            self.reader,
        )
        self.assertIn(raw_id, rejected.view.raw_call_ids)
        self.assertEqual(next_step(rejected.view).kind, "sample_writer")

        repeated = make_turn(
            rejected.view,
            content="",
            calls=(call("write_file", {"path": "draft.txt", "content": "next\n"}, raw_id),),
        )
        transition = derive_writer_turn(rejected.view, repeated, self.reader)
        queued = transition.state.continuation["tool_queue"][0]
        self.assertEqual(queued["name"], "invalid_call")
        self.assertEqual(queued["rejection"], "Duplicate tool call id")

    def test_tool_result_success_and_read_overrun_fixed_points(self):
        action = self._action(call("write_file", {"path": "draft.txt", "content": "revised\n"}))
        obs = self._observation(
            action,
            observation={"ok": True, "valid": True, "result": "written"},
            effect={"draft.txt": {"before": "alpha\n", "after": "revised\n"}},
        )
        result = assert_fixed_point(self, derive_tool_result, action.view, obs, self.reader)
        self.assertEqual(result.event.kind, "tool_result")
        self.assertEqual(result.state.files["draft.txt"], "revised\n")
        self.assertEqual(result.state.continuation["next_call"], 1)

        identical_action = self._action(
            call("write_file", {"path": "draft.txt", "content": "alpha\n"}, "same-write")
        )
        identical = self._observation(
            identical_action,
            observation={"ok": True, "valid": True, "result": "written"},
        )
        unchanged = assert_fixed_point(
            self, derive_tool_result, identical_action.view, identical, self.reader
        )
        self.assertEqual(dict(unchanged.state.files), dict(identical_action.state.files))

        read_view = with_budget(self.root_view, self.reader, limits={"read_tokens": 0})
        action = self._action(call("read_file", {"path": "draft.txt"}, "raw-read"), view=read_view)
        read_obs = self._observation(
            action, observation={"ok": True, "valid": True, "result": "many words"}
        )
        result = assert_fixed_point(self, derive_tool_result, action.view, read_obs, self.reader)
        part = result.view.context.messages[-1].content[0]
        self.assertEqual(
            part["content"], {"ok": False, "valid": True, "error": "Read-token budget exceeded"}
        )
        self.assertEqual(dict(result.state.files), dict(action.state.files))
        self.assertEqual(result.view.budget["consumed"]["read_tokens"], 0)

    def test_identical_content_write_is_a_noop(self):
        action = self._action(call("write_file", {"path": "draft.txt", "content": "alpha\n"}))
        result = assert_fixed_point(
            self,
            derive_tool_result,
            action.view,
            self._observation(
                action,
                observation={"ok": True, "valid": True, "result": "written"},
            ),
            self.reader,
        )

        self.assertEqual(dict(result.state.files), dict(action.state.files))
        self.assertEqual(result.state.tree_hash, action.state.tree_hash)
        self.assertEqual(result.view.budget["consumed"]["tool_calls"], 1)

    def test_tool_error_precedence_budget_then_call_rejection(self):
        malformed = {"id": "bad-envelope", "type": "function"}
        view = with_budget(
            self.root_view,
            self.reader,
            consumed={"tool_calls": 10},
        )
        action = self._action(malformed, view=view)
        obs = self._observation(
            action,
            observation={"ok": False, "valid": True, "error": "Tool-call budget exceeded"},
            dispatch=False,
        )
        result = assert_fixed_point(self, derive_tool_result, action.view, obs, self.reader)
        self.assertEqual(
            result.view.context.messages[-1].content[0]["content"]["error"],
            "Tool-call budget exceeded",
        )
        self.assertEqual(result.view.budget["consumed"]["tool_calls"], 10)

        action = self._action(malformed)
        rejection = action.state.continuation["tool_queue"][0]["name"]
        self.assertEqual(rejection, "invalid_call")
        obs = self._observation(
            action,
            observation={"ok": False, "valid": False, "error": "Invalid tool call envelope"},
            dispatch=False,
        )
        result = assert_fixed_point(self, derive_tool_result, action.view, obs, self.reader)
        self.assertEqual(
            result.view.context.messages[-1].content[0]["content"]["error"],
            "Invalid tool call envelope",
        )

    def test_non_array_tool_calls_flow_through_the_error_result_queue(self):
        turn = make_turn(
            self.root_view,
            content="",
            calls={"not": "an array"},
            tool_calls_was_list=False,
        )
        action = assert_fixed_point(self, derive_writer_turn, self.root_view, turn, self.reader)
        save_artifacts(self.reader, action)
        obs = self._observation(
            action,
            observation={"ok": False, "valid": False, "error": "tool_calls must be an array"},
            dispatch=False,
        )
        result = assert_fixed_point(self, derive_tool_result, action.view, obs, self.reader)
        self.assertEqual(
            result.view.context.messages[-1].content[0]["content"]["error"],
            "tool_calls must be an array",
        )

    def test_eligible_ask_author_is_not_drained(self):
        state_body = self.root_view.state.to_dict()
        state_body["position"]["phase"] = "ready_writer"
        state_body["continuation"]["tool_queue"] = [
            {"call_id": "ask-1", "name": "ask_author", "arguments": {}}
        ]
        state = type(self.root_view.state).from_dict(state_body)
        budget = json.loads(canonical_bytes(self.root_view.budget))
        budget["limits"]["author_calls"] = 1
        view = replace(
            self.root_view,
            state=state,
            budget=budget,
            mode=replace(self.root_view.mode, interaction="scripted_author", ask_semantics=True),
            call_sources={"ask-1": CallSource("rollout-fixture:action:0", 0)},
        )
        obs = ToolObservationV1(
            call_id="ask-1",
            dispatch=None,
        )
        with self.assertRaises(ProjectionError):
            derive_tool_result(view, obs, self.reader)

    def test_high4_effect_and_tool_spec_violations_map_at_producer_boundary(self):
        rows = (
            (
                "read deletes a file",
                call("read_file", {"path": "draft.txt"}, "read-delete"),
                {"draft.txt": {"before": "alpha\n", "after": None}},
                {"ok": True, "valid": True, "result": "alpha"},
                ToolSpec(128_000, 4_096),
            ),
            (
                "write commits other bytes",
                call("write_file", {"path": "draft.txt", "content": "wanted"}, "wrong-write"),
                {"notes/source.md": {"before": "snow\nmoon\n", "after": "forged"}},
                {"ok": True, "valid": True, "result": "written"},
                ToolSpec(128_000, 4_096),
            ),
            (
                "oversize snapshot",
                call("write_file", {"path": "draft.txt", "content": "x" * 101}, "large-write"),
                {"draft.txt": {"before": "alpha\n", "after": "x" * 101}},
                {"ok": True, "valid": True, "result": "written"},
                ToolSpec(100, 300),
            ),
        )
        for name, raw, effect, observation, tool_spec in rows:
            with self.subTest(name=name):
                view = replace(self.root_view, tool_spec=tool_spec)
                action = self._action(raw, view=view)
                obs = self._observation(action, observation=observation, effect=effect)
                with self.assertRaises(ProjectionError):
                    derive_tool_result(action.view, obs, self.reader)
                with self.assertRaises(AdapterContractError):
                    producer_path(derive_tool_result, action.view, obs, self.reader)

        action = self._action(call("write_file", {"path": "draft.txt", "content": "ok"}, "spec"))
        drifted = self._observation(
            action,
            observation={"ok": True, "valid": True, "result": "ok"},
            spec={"max_file_bytes": 127_999, "max_workspace_bytes": 4_096},
        )
        with self.assertRaises(ProjectionError):
            derive_tool_result(action.view, drifted, self.reader)
        with self.assertRaises(AdapterContractError):
            producer_path(derive_tool_result, action.view, drifted, self.reader)

    def test_effect_codec_rejects_noop_paths(self):
        with self.assertRaises(ValueError):
            ToolObservationV1(
                call_id="noop",
                dispatch={
                    "spec": {"max_file_bytes": 10, "max_workspace_bytes": 20},
                    "observation": {"ok": True, "valid": True},
                    "effect": {"draft.txt": {"before": "same", "after": "same"}},
                },
            )


class SamplingCodecTests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.view = make_view()
        self.reader = self.fixture.reader

    def test_ref_only_logprobs_are_producer_and_recorded_contract_errors(self):
        base = make_turn(self.view)
        wire = base.to_wire()
        wire["adapter_trace"] = {"per_token_logprobs_ref": "a" * 64}
        with self.assertRaises(AdapterContractError):
            try:
                WriterTurnV1.from_dict(wire)
            except (TypeError, ValueError) as exc:
                raise AdapterContractError("invalid adapter output") from exc
        with self.assertRaises(ProjectionError):
            try:
                WriterTurnV1.from_dict(wire)
            except (TypeError, ValueError) as exc:
                raise ProjectionError("invalid recorded writer turn") from exc

    def test_typed_logprob_shape_must_match_generated_token_ids(self):
        ref = "b" * 64
        self.reader.byte_values[ref] = b"\0" * 8
        turn = make_turn(
            self.view,
            usage={"completion_tokens": 2},
            adapter_trace={
                "generated_token_ids": [1, 2],
                "per_token_logprobs_ref": ref,
                "per_token_logprobs_codec": "f32-le",
                "per_token_logprobs_shape": [2],
            },
        )
        self.assertEqual(
            derive_writer_turn(self.view, turn, self.reader).event.kind, "writer_action"
        )
        bad_wire = turn.to_wire()
        bad_wire["adapter_trace"]["per_token_logprobs_shape"] = [1]
        # WriterTurnV1's codec accepts the nonnegative shape; the sampling decoder
        # compares it with the token IDs and the addressed byte count.
        bad = WriterTurnV1.from_dict(bad_wire)
        with self.assertRaises(ProjectionError):
            derive_writer_turn(self.view, bad, self.reader)

    def test_prepared_writer_request_binds_payload_and_current_messages(self):
        payload = {"messages": [message.to_dict() for message in self.view.context.messages]}
        payload_ref = self.reader.add(payload)
        prepared = WriterRequestV1(
            context_revision_ref=self.view.context.revision_ref,
            payload_ref=payload_ref,
            verified_messages=True,
        )
        prepared_ref = self.reader.add(prepared.to_wire())
        turn = replace(
            make_turn(self.view), request_ref=payload_ref, prepared_request_ref=prepared_ref
        )
        self.assertEqual(
            derive_writer_turn(self.view, turn, self.reader).event.kind, "writer_action"
        )

        stale_payload_ref = self.reader.add({"messages": []})
        stale = WriterRequestV1(
            context_revision_ref=self.view.context.revision_ref,
            payload_ref=stale_payload_ref,
            verified_messages=True,
        )
        stale_ref = self.reader.add(stale.to_wire())
        with self.assertRaises(ProjectionError):
            derive_writer_turn(
                self.view,
                replace(
                    make_turn(self.view),
                    request_ref=stale_payload_ref,
                    prepared_request_ref=stale_ref,
                ),
                self.reader,
            )


if __name__ == "__main__":
    unittest.main()
