import json
import unittest
from pathlib import Path

from writing_agent.task_graph import (
    EVENT_KINDS,
    CheckpointV1,
    CommitV1,
    ContextContentV1,
    ContextRevisionV1,
    EnvironmentStateV1,
    EventV1,
    GraphInstanceV1,
    LineageRefV1,
    MessageV1,
    NodeSpecV1,
    canonical_json,
    domain_hash,
    domain_hash_bytes,
    file_hash,
    load_canonical_json,
    tree_hash,
    validate_file_tree,
)

H = "0" * 64
FIXTURE = json.loads((Path(__file__).parent / "fixtures/task_graph_hashes.json").read_text())
CHAIN_FIXTURE = json.loads((Path(__file__).parent / "fixtures/task_graph_chain.json").read_text())


class TaskGraphRecordsTest(unittest.TestCase):
    def state(self, files=None):
        files = {"a.txt": "hi"} if files is None else files
        return EnvironmentStateV1(
            instance_ref=H,
            position={
                "node_id": "n",
                "visit_id": "v1",
                "phase": "ready_writer",
                "entry_contract": H,
                "start_checkpoint": None,
                "loop_counts": {},
                "lineage_id": "lin",
            },
            files=files,
            tree_hash=tree_hash(files),
            history={
                "head": None,
                "seq": 0,
                "branch_base": None,
                "imported_refs": (),
                "action_ids": (),
                "tool_result_ids": (),
            },
            context_ref=H,
            requirements_ref=H,
            decisions_ref=H,
            disclosures_ref=H,
            versions_ref=H,
            budgets_ref=H,
            rng_ref=H,
            external_inputs_ref=H,
            outcome_ref=H,
            provenance_ref=H,
            continuation={
                "tool_queue": (),
                "next_call": 0,
                "author_request": None,
                "check_requests": (),
                "external_requests": (),
                "applied_responses": (),
                "feedback_cursor": 0,
            },
            in_flight_effects=(),
        )

    def test_canonical_fixture_and_round_trip(self):
        self.assertEqual(canonical_json({"b": "é", "a": [1, True, None]}), FIXTURE["canonical"])
        message = MessageV1(content=("hello",), origin="a")
        context = ContextRevisionV1(messages=(message,))
        context_content = ContextContentV1.from_revision(context)
        instance = GraphInstanceV1(
            template_ref=H,
            entry_node="n",
            nodes=(NodeSpecV1(id="n", entry_contract=H),),
        )
        state = self.state()
        checkpoint = CheckpointV1(state=state)
        event = EventV1(
            lineage_id="l",
            kind="writer_action",
            audience=("writer",),
            payload_ref=H,
            versions_ref=H,
            provenance_ref=H,
        )
        records = (
            (message, "message"),
            (context, "context"),
            (context_content, "context_content"),
            (instance, "instance"),
            (state, "state"),
            (checkpoint, "checkpoint"),
            (event, "event"),
        )
        for record, expected in records:
            self.assertEqual(record.identity(), FIXTURE[expected])
            self.assertEqual(type(record).from_json(record.to_json()), record)
        self.assertEqual(file_hash("x"), FIXTURE["file"])
        self.assertEqual(tree_hash({"a.txt": "hi"}), FIXTURE["tree"])

    def test_duplicate_keys_and_noncanonical_input_rejected(self):
        with self.assertRaises(ValueError):
            load_canonical_json('{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            load_canonical_json('{"a": 1}')
        with self.assertRaises(ValueError):
            load_canonical_json('{"a":1.0}')

    def test_paths_utf8_and_mutation_sensitivity(self):
        validate_file_tree({"empty": "", "unicode/é.txt": "é\n"})
        for path in ("../x", "/x", "a//b", "a/./b", "a\\b", ""):
            with self.assertRaises(ValueError):
                validate_file_tree({path: "x"})
        validate_file_tree({"A": "x", "a": "x"})
        self.assertNotEqual(file_hash("x"), file_hash("x\n"))
        self.assertNotEqual(tree_hash({"a": "x"}), tree_hash({"a": "y"}))

    def test_state_and_checkpoint_invariants(self):
        state = self.state()
        with self.assertRaises(ValueError):
            EnvironmentStateV1(**{**state.to_dict(), "tree_hash": H})
        with self.assertRaises(ValueError):
            CheckpointV1(state=state, event_head="1" * 64)
        with self.assertRaises(ValueError):
            EnvironmentStateV1(**{**state.to_dict(), "in_flight_effects": ["call"]})

    def test_reference_records_and_mutation(self):
        commit = CommitV1(events=(H,), checkpoint=H)
        lineage = LineageRefV1(lineage_id="line", head_commit=commit.identity())
        self.assertEqual(CommitV1.from_json(commit.to_json()), commit)
        self.assertEqual(LineageRefV1.from_json(lineage.to_json()), lineage)
        self.assertNotEqual(
            commit.identity(), CommitV1(events=(H,), checkpoint="1" * 64).identity()
        )

    def test_strict_wire_loading_and_schema_types(self):
        message = MessageV1(content=("x",), origin="a")
        body = message.to_dict()
        for key in ("schema", "role", "content", "call_id", "origin", "trust", "loss_eligible"):
            missing = dict(body)
            missing.pop(key)
            with self.assertRaises(ValueError):
                MessageV1.from_dict(missing)
        bad = dict(body)
        bad["schema"] = True
        with self.assertRaises(ValueError):
            MessageV1.from_dict(bad)

    def test_state_shape_and_strict_counters(self):
        state = self.state()
        for key in ("visit_id", "entry_contract", "start_checkpoint", "loop_counts", "lineage_id"):
            position = dict(state.position)
            position.pop(key)
            with self.assertRaises((TypeError, ValueError)):
                EnvironmentStateV1(**{**state.to_dict(), "position": position})
        continuation = dict(state.continuation)
        continuation["next_call"] = True
        with self.assertRaises(ValueError):
            EnvironmentStateV1(**{**state.to_dict(), "continuation": continuation})
        history = dict(state.history)
        history["seq"] = True
        with self.assertRaises(ValueError):
            EnvironmentStateV1(**{**state.to_dict(), "history": history})

    def test_immutability_and_message_parts(self):
        parts = [{"type": "text", "text": "x"}]
        message = MessageV1(content=parts, origin="author")
        before = message.identity()
        parts[0]["text"] = "changed"
        self.assertEqual(before, message.identity())
        with self.assertRaises(TypeError):
            MessageV1(content=(object(),), origin="author")
        from dataclasses import dataclass

        @dataclass
        class Mutable:
            value: list[int]

        with self.assertRaises(TypeError):
            MessageV1(content=(Mutable([1]),), origin="author")

    def test_event_allowlist_and_sequence_directions(self):
        kwargs = dict(
            lineage_id="line", audience=("writer",), payload_ref=H, versions_ref=H, provenance_ref=H
        )
        with self.assertRaises(ValueError):
            EventV1(kind="unknown", **kwargs)
        with self.assertRaises(ValueError):
            EventV1(kind="writer_action", previous=H, seq=1, **kwargs)
        with self.assertRaises(ValueError):
            EventV1(kind="writer_action", seq=2, **kwargs)
        expected = frozenset(
            {
                "rollout_started",
                "writer_action",
                "tool_result",
                "author_turn",
                "external_requested",
                "request_entered",
                "seed_attached",
                "requirements_changed",
                "decision_disclosed",
                "check_recorded",
                "transition_committed",
                "termination_recorded",
                "context_changed",
                "fetch_recorded",
                "external_response",
                "budget_charged",
            }
        )
        self.assertEqual(EVENT_KINDS, expected)
        for kind in EVENT_KINDS:
            EventV1(kind=kind, **kwargs)
        for alias in ("writer_observation", "tool_call", "author_reply", "context_revision"):
            with self.assertRaises(ValueError):
                EventV1(kind=alias, **kwargs)

    def test_context_content_separates_provenance(self):
        message = MessageV1(content=("x",), origin="author")
        a = ContextRevisionV1(messages=(message,), provenance_refs=())
        b = ContextRevisionV1(messages=(message,), provenance_refs=(H,))
        self.assertEqual(a.content_hash, b.content_hash)
        self.assertNotEqual(a.identity(), b.identity())
        changed = MessageV1(content=("y",), origin="author")
        self.assertNotEqual(a.content_hash, ContextRevisionV1(messages=(changed,)).content_hash)
        with_tool = ContextRevisionV1(
            messages=(message,), tools=({"name": "read_file", "schema": {"type": "object"}},)
        )
        self.assertNotEqual(a.content_hash, with_tool.content_hash)

    def test_utf8_boundary_and_binary_domain(self):
        with self.assertRaises(ValueError):
            MessageV1(content=("\ud800",), origin="author")
        with self.assertRaises(ValueError):
            validate_file_tree({"\ud800": "x"})
        with self.assertRaises(TypeError):
            domain_hash("file", {})
        self.assertEqual(domain_hash("file", "{}"), file_hash("{}"))
        self.assertNotEqual(file_hash("{}"), domain_hash_bytes("file", b"{}"))

    def test_graph_nodes_normalize_and_lineage_expected_head_is_request_only(self):
        typed = GraphInstanceV1(
            template_ref=H,
            entry_node="n",
            nodes=(NodeSpecV1(id="n", entry_contract=H),),
        )
        shorthand = GraphInstanceV1(
            template_ref=H,
            entry_node="n",
            nodes=({"id": "n", "entry_contract": H},),
        )
        self.assertEqual(typed, shorthand)
        self.assertIsInstance(shorthand.nodes[0], NodeSpecV1)
        a = LineageRefV1(lineage_id="line", expected_head=H)
        b = LineageRefV1(lineage_id="line", expected_head="1" * 64)
        self.assertEqual(a.identity(), b.identity())

    def test_graph_reference_arrays_are_real_arrays(self):
        base = GraphInstanceV1(
            template_ref=H,
            entry_node="n",
            nodes=(NodeSpecV1(id="n", entry_contract=H),),
        )
        for field in ("source_refs", "request_refs"):
            for malformed in ({"ref": H}, "not-an-array", 7, True, None):
                with self.subTest(field=field, malformed=malformed):
                    with self.assertRaises(TypeError):
                        GraphInstanceV1(**{**base.to_dict(), field: malformed})
                    with self.assertRaises(TypeError):
                        GraphInstanceV1.from_dict({**base.to_dict(), field: malformed})
            for valid in ([], [H]):
                with self.subTest(field=field, valid=valid):
                    record = GraphInstanceV1(**{**base.to_dict(), field: valid})
                    self.assertEqual(getattr(record, field), tuple(valid))
                    wire = GraphInstanceV1.from_dict({**base.to_dict(), field: valid})
                    self.assertEqual(getattr(wire, field), tuple(valid))

    def test_wire_decoder_rejects_constructor_shorthands_and_preserves_bytes(self):
        message = MessageV1(content=("x",), origin="author")
        body = message.to_dict()
        with self.assertRaises(ValueError):
            MessageV1.from_dict({**body, "content": ["x"]})
        with self.assertRaises(TypeError):
            MessageV1.from_dict({**body, "content": tuple(body["content"])})
        self.assertEqual(MessageV1.from_json(message.to_json().encode()), message)

        instance = GraphInstanceV1(
            template_ref=H,
            entry_node="n",
            nodes=(NodeSpecV1(id="n", entry_contract=H),),
        )
        with self.assertRaises(ValueError):
            GraphInstanceV1.from_dict({**instance.to_dict(), "nodes": [{"id": "n"}]})
        with self.assertRaises(TypeError):
            GraphInstanceV1.from_dict(
                {**instance.to_dict(), "nodes": tuple(instance.to_dict()["nodes"])}
            )

        event = EventV1(
            lineage_id="line",
            kind="writer_action",
            audience=("writer",),
            payload_ref=H,
            versions_ref=H,
            provenance_ref=H,
        )
        with self.assertRaises(ValueError):
            EventV1.from_dict({**event.to_dict(), "id": None})
        context = ContextRevisionV1(messages=(message,))
        with self.assertRaises(ValueError):
            ContextRevisionV1.from_dict({**context.to_dict(), "content_hash": None})

    def test_history_uses_logical_ids_and_queue_call_ids_are_unique(self):
        state = self.state()
        history = {
            **state.history,
            "action_ids": ("r1:action:0",),
            "tool_result_ids": ("r1:tool_result:0",),
        }
        call = {"call_id": "call-1", "name": "write_file", "arguments": {}}
        valid = EnvironmentStateV1(
            **{
                **state.to_dict(),
                "history": history,
                "continuation": {**state.continuation, "tool_queue": (call,)},
            }
        )
        self.assertEqual(valid.history["action_ids"], ("r1:action:0",))
        with self.assertRaises(ValueError):
            EnvironmentStateV1(**{**state.to_dict(), "history": {**history, "action_ids": (H,)}})
        with self.assertRaises(ValueError):
            EnvironmentStateV1(
                **{**state.to_dict(), "history": {**history, "tool_result_ids": (H,)}}
            )
        with self.assertRaises(ValueError):
            EnvironmentStateV1(
                **{
                    **state.to_dict(),
                    "continuation": {**state.continuation, "tool_queue": (call, call)},
                }
            )

    def test_acyclic_record_round_trip(self):
        message = MessageV1(content=("hello",), origin="author")
        context = ContextRevisionV1(messages=(message,))
        event = EventV1(
            lineage_id="line",
            kind="context_changed",
            audience=("writer",),
            payload_ref=context.identity(),
            versions_ref=H,
            provenance_ref=H,
        )
        state = self.state()
        state = EnvironmentStateV1(
            **{
                **state.to_dict(),
                "context_ref": context.identity(),
                "history": {**state.history, "head": event.identity(), "seq": 1},
            }
        )
        checkpoint = CheckpointV1(state=state, event_head=event.identity())
        commit = CommitV1(events=(event.identity(),), checkpoint=checkpoint.identity())
        for record in (context, event, checkpoint, commit):
            self.assertEqual(type(record).from_json(record.to_json()), record)

    def test_independent_content_event_revision_checkpoint_commit_fixture(self):
        payloads = CHAIN_FIXTURE["payloads"]
        for payload in payloads.values():
            body = load_canonical_json(payload["body"])
            self.assertEqual(canonical_json(body), payload["body"])
            self.assertEqual(domain_hash("payload", body), payload["hash"])
        scenario = CHAIN_FIXTURE["scenario"]
        action_payload = load_canonical_json(payloads["action"]["body"])
        trace_payload = load_canonical_json(payloads["trace"]["body"])
        result_payload = load_canonical_json(payloads["result"]["body"])
        self.assertEqual(action_payload["action_id"], scenario["action_id"])
        self.assertEqual(action_payload["call_ids"], [scenario["call_id"]])
        self.assertEqual(trace_payload["action_id"], scenario["action_id"])
        self.assertEqual(result_payload["action_id"], scenario["action_id"])
        self.assertEqual(result_payload["call_id"], scenario["call_id"])
        self.assertEqual(result_payload["result_id"], scenario["result_id"])

        records = (
            CheckpointV1.from_json(CHAIN_FIXTURE["p0"]["body"]),
            ContextContentV1.from_json(CHAIN_FIXTURE["content_before_observation"]["body"]),
            EventV1.from_json(CHAIN_FIXTURE["event_action"]["body"]),
            ContextRevisionV1.from_json(CHAIN_FIXTURE["revision_before_observation"]["body"]),
            EnvironmentStateV1.from_json(CHAIN_FIXTURE["state_p1"]["body"]),
            CheckpointV1.from_json(CHAIN_FIXTURE["p1"]["body"]),
            CommitV1.from_json(CHAIN_FIXTURE["k1"]["body"]),
            ContextContentV1.from_json(CHAIN_FIXTURE["content_after_observation"]["body"]),
            EventV1.from_json(CHAIN_FIXTURE["event_result"]["body"]),
            ContextRevisionV1.from_json(CHAIN_FIXTURE["revision_after_observation"]["body"]),
            EnvironmentStateV1.from_json(CHAIN_FIXTURE["state_p2"]["body"]),
            CheckpointV1.from_json(CHAIN_FIXTURE["p2"]["body"]),
            CommitV1.from_json(CHAIN_FIXTURE["k2"]["body"]),
        )
        keys = (
            "p0",
            "content_before_observation",
            "event_action",
            "revision_before_observation",
            "state_p1",
            "p1",
            "k1",
            "content_after_observation",
            "event_result",
            "revision_after_observation",
            "state_p2",
            "p2",
            "k2",
        )
        for record, key in zip(records, keys, strict=True):
            expected = CHAIN_FIXTURE[key]
            self.assertEqual(record.to_json(), expected["body"])
            self.assertEqual(record.identity(), expected["hash"])
        (
            p0,
            content1,
            event1,
            revision1,
            state1,
            p1,
            k1,
            content2,
            event2,
            revision2,
            state2,
            p2,
            k2,
        ) = records
        self.assertEqual(event1.payload_ref, payloads["action"]["hash"])
        self.assertEqual(event2.payload_ref, payloads["result"]["hash"])
        self.assertEqual(event2.previous, event1.identity())
        self.assertNotIn(event2.identity(), content2.to_json())
        self.assertEqual(revision1.content_hash, content1.identity())
        self.assertIn(event1.identity(), revision1.provenance_refs)
        self.assertEqual(revision2.content_hash, content2.identity())
        self.assertIn(event2.identity(), revision2.provenance_refs)
        self.assertEqual(state1.context_ref, revision1.identity())
        self.assertEqual(state2.context_ref, revision2.identity())
        self.assertEqual(p1.parents, (p0.identity(),))
        self.assertEqual(p2.parents, (p1.identity(),))
        self.assertEqual(k1.parent_commit, None)
        self.assertEqual(k2.parent_commit, k1.identity())
        self.assertEqual(k1.checkpoint, p1.identity())
        self.assertEqual(k2.checkpoint, p2.identity())
        self.assertTrue(state1.continuation["tool_queue"])
        self.assertFalse(state2.continuation["tool_queue"])
        self.assertEqual(state1.history["action_ids"], (scenario["action_id"],))
        self.assertEqual(state2.history["tool_result_ids"], (scenario["result_id"],))


if __name__ == "__main__":
    unittest.main()
