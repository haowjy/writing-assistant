import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from writing_agent import task_graph_writer
from writing_agent.agent import SYSTEM_PROMPT
from writing_agent.legacy_graph import compile_legacy_scenario
from writing_agent.task_graph import (
    ContextRevisionV1,
    EnvironmentStateV1,
    EventV1,
    MessageV1,
    tree_hash,
)
from writing_agent.task_graph_projection import ProjectionError, project_writer_context
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_writer import TransactionalWriterV1, WriterRuntimeError
from writing_agent.workspace import TOOL_SCHEMAS, Workspace


def scenario():
    return {
        "id": "writer-runtime",
        "family": "F2",
        "role": "development",
        "condition": "workspace",
        "source_groups": ["synthetic"],
        "provenance": "synthetic",
        "visible": {
            "brief": "Revise the text files.",
            "initial_files": {"draft.txt": "alpha\n", "notes/source.md": "snow\nmoon\n"},
            "followups": [],
            "tools": ["list_dir", "read_file", "search", "write_file", "patch_file"],
            "budgets": {
                "max_steps": 5,
                "max_tool_calls": 10,
                "max_read_tokens": 100,
                "max_total_bytes": 4096,
            },
            "prose": [],
        },
        "labels": {
            "rubric_version": 1,
            "checks": [],
            "rubrics": {},
            "knowledge": [],
            "source_cutoff": "supplied files only",
        },
    }


class WriterFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.store = TaskGraphStore(self.root / "store")
        self.bundle = compile_legacy_scenario(scenario())
        for identity, body in self.bundle.public_artifacts.items():
            self.assertEqual(self.store.put_artifact(body), identity)
        for identity, body in self.bundle.private_artifacts.items():
            self.assertEqual(self.store.put_artifact(body, private=True), identity)
        self.store.persist(self.bundle.instance)
        node = self.bundle.instance.nodes[0]
        contract = self.bundle.admission().node(node.id).contract
        files = dict(self.bundle.initial_files)
        rendering = {
            "projection_version": "v1",
            "prefix_id": "root",
            "template_ref": self.store.put_artifact({"pin": "template"}),
            "tokenizer_ref": self.store.put_artifact({"pin": "tokenizer"}),
            "tool_schema_ref": self.store.put_artifact({"pin": "tools"}),
        }
        tools = tuple(
            schema for schema in TOOL_SCHEMAS if schema["function"]["name"] in self.bundle.tools
        )
        context = ContextRevisionV1(
            messages=(
                MessageV1(
                    role="system", content=(SYSTEM_PROMPT,), origin="system:1", trust="instructions"
                ),
                MessageV1(role="user", content=("Revise the text files.",), origin="request:1"),
            ),
            tools=tools,
            rendering=rendering,
        )
        self.store.persist(context)
        limits = {
            "writer_turns": contract.budget_contract.max_steps,
            "tool_calls": contract.budget_contract.max_tool_calls,
            "read_tokens": contract.budget_contract.max_read_tokens,
            "storage_bytes": contract.budget_contract.max_total_bytes,
        }
        budget = {
            "schema": 1,
            "limits": limits,
            "consumed": {"storage_bytes": sum(len(text.encode()) for text in files.values())},
            "read_tokenizer": "whitespace-v1",
        }
        refs = {
            name: self.store.put_artifact({"seed": name})
            for name in (
                "requirements",
                "decisions",
                "disclosures",
                "versions",
                "rng",
                "external",
                "outcome",
                "provenance",
            )
        }
        self.state = EnvironmentStateV1(
            instance_ref=self.bundle.instance.identity(),
            position={
                "node_id": node.id,
                "visit_id": "visit-1",
                "phase": "ready_writer",
                "entry_contract": node.entry_contract,
                "start_checkpoint": None,
                "loop_counts": {},
                "lineage_id": "rollout-1",
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
            context_ref=context.identity(),
            requirements_ref=refs["requirements"],
            decisions_ref=refs["decisions"],
            disclosures_ref=refs["disclosures"],
            author_packet_ref=None,
            versions_ref=refs["versions"],
            budgets_ref=self.store.put_artifact(budget),
            rng_ref=refs["rng"],
            external_inputs_ref=refs["external"],
            outcome_ref=refs["outcome"],
            provenance_ref=refs["provenance"],
            continuation={
                "tool_queue": (),
                "next_call": 0,
                "author_request": None,
                "check_requests": (),
                "external_requests": (),
                "applied_responses": (),
                "feedback_cursor": 0,
            },
        )
        self.start = self.store.save_checkpoint(self.state)
        self.runtime = self.store.restore(self.start, self.root / "workspace-0")
        self.writer = TransactionalWriterV1(
            self.store, self.bundle.admission(), "rollout-1", self.start
        )

    def call(self, name, arguments, call_id="backend-1"):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def action(self, *calls, content=""):
        return {"role": "assistant", "content": content, "tool_calls": list(calls)}

    def record(self, step):
        return self.store.get_artifact(step.record_ref)

    def entry_with_budget(self, *, consumed=None, limits=None):
        budget = self.store.get_artifact(self.runtime.state.budgets_ref)
        budget["consumed"].update(consumed or {})
        budget["limits"].update(limits or {})
        state = self.runtime.state.to_dict()
        state["budgets_ref"] = self.store.put_artifact(budget)
        checkpoint = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        runtime = self.store.restore(checkpoint, self.root / f"entry-{checkpoint[:8]}")
        writer = TransactionalWriterV1(self.store, self.bundle.admission(), "rollout-1", checkpoint)
        return writer, runtime, checkpoint


class TransactionalWriterTest(WriterFixture):
    def test_non_array_call_batch_rejected_without_publication(self):
        with self.assertRaisesRegex(WriterRuntimeError, "tool_calls must be an array"):
            self.writer.submit_action(
                self.runtime, {"role": "assistant", "content": "", "tool_calls": None}
            )
        self.assertIsNone(self.store.read_head("rollout-1"))

    def test_malformed_envelopes_are_paired_charged_and_nonmutating(self):
        calls = [
            None,
            {},
            {
                **self.call("write_file", {"path": "bad.txt", "content": "bad"}, "wrong"),
                "type": "other",
            },
            self.call(
                "write_file", '{"path":"a.txt","path":"bad.txt","content":"bad"}', "duplicate"
            ),
            self.call("bad tool", {}, "space"),
            self.call("bad\x00tool", {}, "nul"),
            self.call([], {}, "non-string-name"),
            self.call("write_file", '{"path":"bad.txt","content":"\\ud800"}', "surrogate"),
            self.call("write_file", {"path": "bad.txt", "content": "\ud800"}, "literal"),
            self.call("write_file", b'{"path":"bad.txt","content":"\xff"}', "invalid-bytes"),
        ]
        action = self.writer.submit_action(self.runtime, self.action(*calls))
        self.assertEqual(len(self.record(action)["calls"]), len(calls))
        runtime = action.runtime
        for _ in calls:
            result = self.writer.step_tool(runtime)
            runtime = result.runtime
            self.assertFalse(self.record(result)["observation"]["valid"])
        self.assertEqual(runtime.state.files, self.runtime.state.files)
        self.assertEqual(runtime.state.continuation["next_call"], len(calls))
        self.assertEqual(
            self.store.get_artifact(runtime.state.budgets_ref)["consumed"]["attempted_tool_calls"],
            len(calls),
        )
        self.assertEqual(
            self.store.replay("rollout-1", self.start, self._commits()), runtime.checkpoint_id
        )

    def _commits(self):
        head = self.store.read_head("rollout-1")
        commits = []
        while head:
            commits.append(head)
            head = self.store.load_commit(head).parent_commit
        return list(reversed(commits))

    def test_infrastructure_failure_interrupts_without_call_charge(self):
        action = self.writer.submit_action(
            self.runtime,
            self.action(self.call("write_file", {"path": "new.txt", "content": "yes"})),
        )
        head = self.store.read_head("rollout-1")
        with patch.object(Workspace, "write_file", side_effect=OSError(errno.ENOSPC, "disk full")):
            with self.assertRaises(OSError):
                self.writer.step_tool(action.runtime)
        self.assertEqual(self.store.read_head("rollout-1"), head)
        self.assertEqual(action.runtime.state.continuation["next_call"], 0)
        self.assertEqual(
            self.store.get_artifact(action.runtime.state.budgets_ref)["consumed"].get(
                "attempted_tool_calls", 0
            ),
            0,
        )
        retry = self.writer.step_tool(action.runtime)
        self.assertEqual(retry.runtime.state.files["new.txt"], "yes")

    def test_graph_dispatch_preserves_other_infrastructure_failures(self):
        for code in (errno.EIO, errno.EROFS, errno.EACCES):
            with self.subTest(errno=code):
                with patch.object(
                    Workspace, "write_file", side_effect=OSError(code, "I/O failure")
                ):
                    with self.assertRaises(OSError):
                        task_graph_writer._graph_dispatch(
                            Workspace(self.root / "dispatch-stage"),
                            "write_file",
                            {"path": "draft.txt", "content": "new"},
                        )

    def test_stale_handle_dispatches_zero_times(self):
        action = self.writer.submit_action(
            self.runtime,
            self.action(self.call("write_file", {"path": "new.txt", "content": "yes"})),
        )
        self.writer.step_tool(action.runtime)
        with patch.object(
            task_graph_writer, "_graph_dispatch", side_effect=AssertionError("dispatched")
        ):
            with self.assertRaisesRegex(WriterRuntimeError, "stale runtime handle"):
                self.writer.step_tool(action.runtime)

    def test_forged_result_origin_hash_and_charge_fail_restore_projection_replay(self):
        action = self.writer.submit_action(
            self.runtime, self.action(self.call("read_file", {"path": "draft.txt"}))
        )
        publish = self.writer._publish

        def forged(*args, **kwargs):
            record = kwargs["record"]
            record["action_id"] = "unrelated:action:99"
            record["before_execution_hash"] = "0" * 64
            record["after_execution_hash"] = "0" * 64
            record["budget_charge"]["read_tokens"] = 999
            message = kwargs["message"].to_dict()
            message["origin"] = "unrelated:action:99"
            kwargs["message"] = MessageV1.from_dict(message)
            return publish(*args, **kwargs)

        with patch.object(self.writer, "_publish", forged):
            with self.assertRaises(ProjectionError):
                self.writer.step_tool(action.runtime)
        head = self.store.read_head("rollout-1")
        checkpoint = self.store.load_commit(head).checkpoint
        with self.assertRaises(ProjectionError):
            self.store.restore(checkpoint, self.root / "forged-restore")
        with self.assertRaises(ProjectionError):
            project_writer_context(self.store, self.start, checkpoint)
        with self.assertRaises(ProjectionError):
            self.store.replay("rollout-1", self.start, self._commits())

    def test_independent_forged_result_fields_fail_semantic_validation(self):
        for field in (
            "origin",
            "before_execution_hash",
            "after_execution_hash",
            "budget_charge",
            "file_delta",
        ):
            with self.subTest(field=field):
                fixture = WriterFixture()
                fixture.setUp()
                try:
                    action = fixture.writer.submit_action(
                        fixture.runtime,
                        fixture.action(fixture.call("read_file", {"path": "draft.txt"})),
                    )
                    original = fixture.writer._publish

                    def forged(*args, field=field, original=original, **kwargs):
                        record = kwargs["record"]
                        if field == "origin":
                            record["action_id"] = "unrelated:action:99"
                            message = kwargs["message"].to_dict()
                            message["origin"] = "unrelated:action:99"
                            kwargs["message"] = MessageV1.from_dict(message)
                        elif field in {"before_execution_hash", "after_execution_hash"}:
                            record[field] = "0" * 64
                        elif field == "budget_charge":
                            record[field]["read_tokens"] = 999
                        else:
                            record[field] = {"draft.txt": {"before": "alpha\n", "after": "forged"}}
                        return original(*args, **kwargs)

                    with patch.object(fixture.writer, "_publish", forged):
                        with self.assertRaises(ProjectionError):
                            fixture.writer.step_tool(action.runtime)
                    head = fixture.store.read_head("rollout-1")
                    checkpoint = fixture.store.load_commit(head).checkpoint
                    with self.assertRaises(ProjectionError):
                        project_writer_context(fixture.store, fixture.start, checkpoint)
                    with self.assertRaises(ProjectionError):
                        fixture.store.replay("rollout-1", fixture.start, [action.commit_id, head])
                finally:
                    fixture.doCleanups()

    def test_context_event_cannot_hide_unattributed_file_change(self):
        action = self.writer.submit_action(
            self.runtime, self.action(self.call("read_file", {"path": "draft.txt"}))
        )
        state = action.runtime.state
        effect = {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": state.identity(),
            "file_delta": {"injected.txt": {"before": None, "after": "private"}},
            "set": {"context_ref": state.context_ref},
            "history_set": {},
        }
        event = EventV1(
            previous=state.history["head"],
            seq=state.history["seq"] + 1,
            lineage_id="rollout-1",
            kind="context_changed",
            actor="environment",
            audience=("controller", "trainer"),
            payload_ref=self.store.put_artifact(effect),
            versions_ref=state.versions_ref,
            provenance_ref=state.provenance_ref,
        )
        next_state = self.store._apply_recorded_effect_body(state, event, effect)
        commit = self.store.publish("rollout-1", action.commit_id, (event,), next_state)
        checkpoint = self.store.load_commit(commit).checkpoint
        with self.assertRaisesRegex(ProjectionError, "unrelated execution effect"):
            project_writer_context(self.store, self.start, checkpoint)
        with self.assertRaises(ProjectionError):
            self.store.replay("rollout-1", self.start, [action.commit_id, commit])

    def test_interruption_restore_replay_projection_and_final_reply(self):
        prepared_ref = self.writer.prepare_request(
            self.runtime, {"rendered": "exact bytes\n", "template": "v1"}
        )
        self.assertIsNone(self.store.read_head("rollout-1"))
        first = self.writer.submit_action(
            self.runtime,
            self.action(
                self.call("write_file", {"path": "draft.txt", "content": "βeta\n"}, "a"),
                self.call("read_file", {"path": "draft.txt"}, "b"),
                self.call("search", {"query": "moon", "path": "notes"}, "c"),
            ),
            prepared_request_ref=prepared_ref,
            raw_output=b"exact writer bytes\n",
            usage={"prompt_tokens": 9, "completion_tokens": 5},
        )
        self.assertEqual(first.runtime.state.continuation["next_call"], 0)
        self.assertEqual(len(first.runtime.state.continuation["tool_queue"]), 3)
        self.assertEqual(
            [message.role for message in first.runtime.context.messages],
            ["system", "user", "assistant"],
        )
        trace = self.store.get_artifact(self.record(first)["trace_ref"])
        self.assertEqual(trace["prepared_request_ref"], prepared_ref)
        self.assertEqual(trace["token_evidence"], "missing")
        self.assertFalse(trace["native_on_policy_eligible"])
        self.assertEqual(trace["raw_output_evidence"], "supplied")
        self.assertEqual(
            self.store.get_artifact(trace["raw_output_ref"], expected_domain="payload:bytes"),
            b"exact writer bytes\n",
        )
        self.assertEqual(
            self.store.get_artifact(trace["exact_request_ref"]),
            {"rendered": "exact bytes\n", "template": "v1"},
        )
        second = self.writer.step_tool(first.runtime)
        self.assertEqual(second.runtime.state.files["draft.txt"], "βeta\n")
        restored = self.store.restore(second.runtime.checkpoint_id, self.root / "restored")
        third = self.writer.step_tool(restored)
        fourth = self.writer.step_tool(third.runtime)
        self.assertEqual(fourth.runtime.state.continuation["next_call"], 3)
        self.assertEqual(self.record(third)["observation"]["result"], "βeta\n")
        self.assertEqual(self.record(fourth)["observation"]["result"][0]["text"], "moon")
        self.assertEqual(self.record(second)["budget_charge"]["file_byte_delta"], 0)
        self.assertGreater(self.record(third)["budget_charge"]["read_tokens"], 0)
        projected = project_writer_context(self.store, self.start, fourth.runtime.checkpoint_id)
        self.assertEqual(
            [message.role for message in projected.messages],
            ["system", "user", "assistant", "tool", "tool", "tool"],
        )
        final = self.writer.submit_action(fourth.runtime, self.action(content="Done."))
        self.assertEqual(final.runtime.state.position["phase"], "checking")
        self.assertEqual(
            [message.loss_eligible for message in final.runtime.context.messages],
            [False, False, True, False, False, False, True],
        )
        self.assertEqual(
            project_writer_context(self.store, self.start, final.runtime.checkpoint_id),
            final.runtime.context,
        )
        commits = [
            first.commit_id,
            second.commit_id,
            third.commit_id,
            fourth.commit_id,
            final.commit_id,
        ]
        self.assertEqual(
            self.store.replay("rollout-1", self.start, commits), final.runtime.checkpoint_id
        )
        self.assertEqual(
            [
                self.store.load_event(event).kind
                for commit in commits
                for event in self.store.load_commit(commit).events
            ],
            [
                "writer_action",
                "context_changed",
                "tool_result",
                "context_changed",
                "tool_result",
                "context_changed",
                "tool_result",
                "context_changed",
                "writer_action",
                "context_changed",
            ],
        )
        with self.assertRaises(WriterRuntimeError):
            self.writer.step_tool(final.runtime)

    def test_invalid_duplicate_unavailable_mixed_and_operational_failure(self):
        action = self.writer.submit_action(
            self.runtime,
            self.action(
                self.call("write_file", {"path": "good.txt", "content": "yes"}, "same"),
                self.call("write_file", {"path": "bad.txt", "content": "no"}, "same"),
                self.call("bash", {}, "shell"),
                self.call(
                    "patch_file", {"path": "draft.txt", "old": "absent", "new": "X"}, "patch"
                ),
                self.call("read_file", {"path": "../secret"}, "path"),
                self.call("write_file", {"path": "wrong.txt", "content": 3}, "args"),
                self.call("list_dir", {}, ""),
                self.call("write_file", {"path": "evil\\name", "content": "no"}, "slash"),
            ),
        )
        runtime = action.runtime
        observations = []
        for _ in range(8):
            step = self.writer.step_tool(runtime)
            runtime = step.runtime
            observations.append(self.record(step)["observation"])
        self.assertTrue(observations[0]["ok"])
        self.assertEqual(runtime.state.files["good.txt"], "yes")
        self.assertNotIn("bad.txt", runtime.state.files)
        self.assertEqual(
            [item["valid"] for item in observations],
            [True, False, False, True, False, False, False, False],
        )
        self.assertFalse(observations[3]["ok"])
        self.assertFalse(observations[4]["ok"])
        self.assertEqual(
            self.store.get_artifact(runtime.state.budgets_ref)["consumed"]["attempted_tool_calls"],
            8,
        )
        self.assertEqual(
            len({call["call_id"] for call in runtime.state.continuation["tool_queue"]}), 8
        )
        project_writer_context(self.store, self.start, runtime.checkpoint_id)

    def test_control_file_mix_rejects_all_before_execution(self):
        action = self.writer.submit_action(
            self.runtime,
            self.action(
                self.call("write_file", {"path": "new.txt", "content": "bad"}, "file"),
                self.call("ask_author", {"question": "Which?"}, "author"),
            ),
        )
        runtime = self.writer.drain_tools(action.runtime)
        self.assertNotIn("new.txt", runtime.state.files)
        self.assertEqual(runtime.state.continuation["next_call"], 2)
        self.assertEqual(
            self.store.get_artifact(runtime.state.budgets_ref)["consumed"]["attempted_tool_calls"],
            2,
        )

    def test_read_limit_hides_content_and_file_bytes_use_utf8(self):
        budget = self.store.get_artifact(self.runtime.state.budgets_ref)
        budget["consumed"]["read_tokens"] = budget["limits"]["read_tokens"]
        # Change only fixture state before saving a new entry checkpoint.
        from writing_agent.task_graph import EnvironmentStateV1

        state = self.runtime.state.to_dict()
        state["budgets_ref"] = self.store.put_artifact(budget)
        start = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        runtime = self.store.restore(start, self.root / "small-read")
        writer = TransactionalWriterV1(self.store, self.bundle.admission(), "rollout-1", start)
        action = writer.submit_action(
            runtime, self.action(self.call("read_file", {"path": "draft.txt"}))
        )
        result = writer.step_tool(action.runtime)
        self.assertNotIn("alpha", str(self.record(result)["observation"]))

    def test_projection_does_not_render_private_event(self):
        private = self.store.put_artifact({"secret": "hidden reward"}, private=True)
        effect = {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": self.runtime.state.identity(),
            "file_delta": {},
            "set": {},
            "history_set": {},
        }
        event = EventV1(
            previous=None,
            seq=1,
            lineage_id="rollout-1",
            kind="check_recorded",
            actor="evaluator",
            audience=("controller", "evaluator", "writer"),
            payload_ref=self.store.put_artifact(effect),
            versions_ref=self.runtime.state.versions_ref,
            provenance_ref=self.runtime.state.provenance_ref,
        )
        next_state = self.store._apply_recorded_effect_body(self.runtime.state, event, effect)
        commit = self.store.publish(
            "rollout-1",
            None,
            (event,),
            next_state,
            parent_checkpoint=self.start,
            artifact_refs=(private,),
        )
        checkpoint = self.store.load_commit(commit).checkpoint
        projection = project_writer_context(self.store, self.start, checkpoint)
        self.assertEqual(projection.messages, self.runtime.context.messages)
        self.assertNotIn("hidden reward", str(projection.to_dict()))

    def test_projection_rejects_context_injection_from_operational_event(self):
        injected = ContextRevisionV1(
            messages=(
                *self.runtime.context.messages,
                MessageV1(role="user", content=("private rubric: pass",), origin="attack:1"),
            ),
            tools=self.runtime.context.tools,
            rendering=self.runtime.context.rendering,
        )
        self.store.persist(injected)
        effect = {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": self.runtime.state.identity(),
            "file_delta": {},
            "set": {"context_ref": injected.identity()},
            "history_set": {},
        }
        event = EventV1(
            previous=None,
            seq=1,
            lineage_id="rollout-1",
            kind="check_recorded",
            actor="evaluator",
            audience=("controller", "evaluator", "writer"),
            payload_ref=self.store.put_artifact(effect),
            versions_ref=self.runtime.state.versions_ref,
            provenance_ref=self.runtime.state.provenance_ref,
        )
        next_state = self.store._apply_recorded_effect_body(self.runtime.state, event, effect)
        commit = self.store.publish(
            "rollout-1", None, (event,), next_state, parent_checkpoint=self.start
        )
        checkpoint = self.store.load_commit(commit).checkpoint
        with self.assertRaisesRegex(ProjectionError, "context revision was changed"):
            project_writer_context(self.store, self.start, checkpoint)
        poisoned = self.store.restore(checkpoint, self.root / "poisoned")
        with self.assertRaises(ProjectionError):
            self.writer.submit_action(poisoned, self.action(content="I saw it"))

    def test_entry_rejects_unadmitted_tool_schema_and_tokenizer_switch(self):
        context = ContextRevisionV1(
            messages=self.runtime.context.messages,
            tools=(*self.runtime.context.tools, {"type": "function", "function": {"name": "bash"}}),
            rendering=self.runtime.context.rendering,
        )
        self.store.persist(context)
        state = self.runtime.state.to_dict()
        state["context_ref"] = context.identity()
        checkpoint = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        poisoned = self.store.restore(checkpoint, self.root / "poisoned-tools")
        writer = TransactionalWriterV1(self.store, self.bundle.admission(), "rollout-1", checkpoint)
        with self.assertRaisesRegex(WriterRuntimeError, "outside the admitted"):
            writer.submit_action(poisoned, self.action(content="No shell"))
        switched = TransactionalWriterV1(
            self.store,
            self.bundle.admission(),
            "rollout-1",
            self.start,
            read_tokenizer="different-v1",
        )
        with self.assertRaisesRegex(WriterRuntimeError, "read tokenizer differs"):
            switched.submit_action(self.runtime, self.action(content="No switch"))
        forged_request = ContextRevisionV1(
            messages=(
                self.runtime.context.messages[0],
                MessageV1(role="user", content=("private rubric",), origin="request:1"),
            ),
            tools=self.runtime.context.tools,
            rendering=self.runtime.context.rendering,
        )
        self.store.persist(forged_request)
        state = self.runtime.state.to_dict()
        state["context_ref"] = forged_request.identity()
        checkpoint = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        with self.assertRaisesRegex(WriterRuntimeError, "request differs"):
            TransactionalWriterV1(self.store, self.bundle.admission(), "rollout-1", checkpoint)

    def test_projection_rejects_false_context_provenance(self):
        action = self.writer.submit_action(self.runtime, self.action(content="A reply."))
        forged = ContextRevisionV1(
            messages=action.runtime.context.messages,
            tools=action.runtime.context.tools,
            event_head=action.event_id,
            provenance_refs=(action.event_id, action.event_id),
            rendering=action.runtime.context.rendering,
        )
        self.store.persist(forged)
        effect = {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": action.runtime.state.identity(),
            "file_delta": {},
            "set": {"context_ref": forged.identity()},
            "history_set": {},
        }
        event = EventV1(
            previous=action.runtime.state.history["head"],
            seq=action.runtime.state.history["seq"] + 1,
            lineage_id="rollout-1",
            kind="context_changed",
            actor="environment",
            audience=("controller", "trainer"),
            payload_ref=self.store.put_artifact(effect),
            versions_ref=action.runtime.state.versions_ref,
            provenance_ref=action.runtime.state.provenance_ref,
        )
        next_state = self.store._apply_recorded_effect_body(action.runtime.state, event, effect)
        commit = self.store.publish("rollout-1", action.commit_id, (event,), next_state)
        checkpoint = self.store.load_commit(commit).checkpoint
        with self.assertRaisesRegex(ProjectionError, "false source-event provenance"):
            project_writer_context(self.store, self.start, checkpoint)

    def test_valid_list_patch_and_unicode_byte_accounting(self):
        action = self.writer.submit_action(
            self.runtime,
            self.action(
                self.call("list_dir", {"path": "."}, "list"),
                self.call(
                    "patch_file", {"path": "draft.txt", "old": "alpha", "new": "snow"}, "patch"
                ),
                self.call("write_file", {"path": "unicode.txt", "content": "雪"}, "unicode"),
                self.call("read_file", {"path": "unicode.txt"}, "read"),
            ),
        )
        runtime = action.runtime
        results = []
        for _ in range(4):
            step = self.writer.step_tool(runtime)
            runtime = step.runtime
            results.append(self.record(step))
        self.assertEqual(results[0]["observation"]["result"], ["draft.txt", "notes/"])
        self.assertEqual(runtime.state.files["draft.txt"], "snow\n")
        self.assertEqual(runtime.state.files["unicode.txt"], "雪")
        self.assertEqual(results[2]["budget_charge"]["file_byte_delta"], 3)
        self.assertEqual(results[3]["observation"]["result"], "雪")
        self.assertEqual(
            self.store.get_artifact(runtime.state.budgets_ref)["consumed"]["storage_bytes"],
            sum(len(text.encode("utf-8")) for text in runtime.state.files.values()),
        )
        self.assertTrue(
            all(item["before_execution_hash"] != item["after_execution_hash"] for item in results)
        )
        project_writer_context(self.store, self.start, runtime.checkpoint_id)

    def test_tool_and_turn_limit_boundaries(self):
        writer, runtime, start = self.entry_with_budget(consumed={"tool_calls": 9})
        action = writer.submit_action(
            runtime,
            self.action(
                self.call("write_file", {"path": "first.txt", "content": "kept"}, "first"),
                self.call("write_file", {"path": "second.txt", "content": "blocked"}, "second"),
            ),
        )
        one = writer.step_tool(action.runtime)
        two = writer.step_tool(one.runtime)
        self.assertEqual(two.runtime.state.files["first.txt"], "kept")
        self.assertNotIn("second.txt", two.runtime.state.files)
        self.assertEqual(self.record(two)["observation"]["error"], "Tool-call budget exceeded")
        self.assertEqual(
            self.store.get_artifact(two.runtime.state.budgets_ref)["consumed"][
                "attempted_tool_calls"
            ],
            2,
        )
        project_writer_context(self.store, start, two.runtime.checkpoint_id)

        # The final allowed reply enters checking; no check or termination is invented.
        budget = self.store.get_artifact(runtime.state.budgets_ref)
        budget["consumed"]["writer_turns"] = 4
        state = runtime.state.to_dict()
        state["position"]["lineage_id"] = "rollout-2"
        state["budgets_ref"] = self.store.put_artifact(budget)
        checkpoint = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        ready = self.store.restore(checkpoint, self.root / "last-ready")
        last_writer = TransactionalWriterV1(
            self.store, self.bundle.admission(), "rollout-2", checkpoint
        )
        final = last_writer.submit_action(ready, self.action(content="Final answer."))
        self.assertEqual(final.runtime.state.position["phase"], "checking")
        with self.assertRaises(WriterRuntimeError):
            last_writer.submit_action(final.runtime, self.action(content="Extra"))

        state["position"]["lineage_id"] = "rollout-3"
        terminal_start = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        terminal_ready = self.store.restore(terminal_start, self.root / "terminal-ready")
        terminal_writer = TransactionalWriterV1(
            self.store, self.bundle.admission(), "rollout-3", terminal_start
        )
        tool_action = terminal_writer.submit_action(
            terminal_ready,
            self.action(self.call("write_file", {"path": "last.txt", "content": "saved"}, "last")),
        )
        tool_result = terminal_writer.step_tool(tool_action.runtime)
        stopped = terminal_writer.stop_exhausted(tool_result.runtime)
        self.assertEqual(stopped.runtime.state.position["phase"], "terminal")
        self.assertEqual(
            self.store.get_artifact(stopped.runtime.state.outcome_ref)["stop_reason"],
            "writer_budget",
        )
        self.assertEqual(
            project_writer_context(self.store, terminal_start, stopped.runtime.checkpoint_id),
            stopped.runtime.context,
        )
        self.assertEqual(
            self.store.replay(
                "rollout-3",
                terminal_start,
                [tool_action.commit_id, tool_result.commit_id, stopped.commit_id],
            ),
            stopped.runtime.checkpoint_id,
        )

    def test_reused_backend_ids_and_nested_usage_are_not_double_charged(self):
        stale_request = self.writer.prepare_request(self.runtime, b"first-context")
        first = self.writer.submit_action(
            self.runtime,
            self.action(self.call("read_file", {"path": "draft.txt"}, "reused")),
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
            trace={"generated_token_ids": [1, 2, 3]},
        )
        self.assertEqual(
            self.store.get_artifact(first.runtime.state.budgets_ref)["consumed"]["total_tokens"], 10
        )
        trace = self.store.get_artifact(self.record(first)["trace_ref"])
        self.assertEqual(trace["token_evidence"], "supplied")
        self.assertEqual(trace["logprob_evidence"], "missing")
        self.assertFalse(trace["native_on_policy_eligible"])
        drained = self.writer.step_tool(first.runtime).runtime
        with self.assertRaisesRegex(WriterRuntimeError, "sampling context"):
            self.writer.submit_action(
                drained,
                self.action(content="Stale request"),
                prepared_request_ref=stale_request,
            )
        second = self.writer.submit_action(
            drained,
            self.action(
                self.call("write_file", {"path": "blocked.txt", "content": "no"}, "reused")
            ),
        )
        result = self.writer.step_tool(second.runtime)
        self.assertEqual(self.record(result)["observation"]["error"], "Duplicate tool call id")
        self.assertNotIn("blocked.txt", result.runtime.state.files)

    def test_optional_token_limits_require_evidence_and_do_not_infer_masks(self):
        with self.assertRaisesRegex(WriterRuntimeError, "context_tokens cannot be enforced"):
            self.entry_with_budget(limits={"context_tokens": 10})
        writer, runtime, start = self.entry_with_budget(
            limits={"generated_tokens": 2, "total_tokens": 7}
        )
        with self.assertRaisesRegex(WriterRuntimeError, "usage evidence"):
            writer.submit_action(runtime, self.action(content="Reply"))
        logprob_ref = self.store.put_bytes_artifact(b"opaque-logprob-tensor")
        prepared = writer.prepare_request(runtime, b"exact request")
        action = writer.submit_action(
            runtime,
            self.action(content="Reply"),
            prepared_request_ref=prepared,
            raw_output=b"Reply",
            usage={"prompt_tokens": 5, "completion_tokens": 2},
            trace={"per_token_logprobs_ref": logprob_ref},
        )
        trace = self.store.get_artifact(self.record(action)["trace_ref"])
        self.assertEqual(trace["logprob_evidence"], "supplied")
        self.assertFalse(trace["native_on_policy_eligible"])
        self.assertEqual(
            self.store.get_artifact(logprob_ref, expected_domain="payload:bytes"),
            b"opaque-logprob-tensor",
        )
        self.assertEqual(
            self.store.replay("rollout-1", start, [action.commit_id]), action.runtime.checkpoint_id
        )

    def test_sampled_token_overshoot_is_durable_and_terminal(self):
        writer, runtime, start = self.entry_with_budget(limits={"generated_tokens": 1})
        prepared = writer.prepare_request(runtime, b"paid request")
        stop = writer.submit_action(
            runtime,
            self.action(self.call("write_file", {"path": "no.txt", "content": "no"})),
            prepared_request_ref=prepared,
            raw_output=b"sampled raw",
            usage={"prompt_tokens": 5, "completion_tokens": 2},
        )
        self.assertEqual(stop.runtime.state.position["phase"], "terminal")
        self.assertNotIn("no.txt", stop.runtime.state.files)
        self.assertEqual(
            self.store.get_artifact(stop.runtime.state.outcome_ref)["stop_reason"],
            "generated_tokens_budget",
        )
        self.assertEqual(
            self.store.get_artifact(stop.runtime.state.budgets_ref)["consumed"]["generated_tokens"],
            2,
        )
        self.assertEqual(self.record(stop)["raw_output_ref"] is not None, True)
        self.assertEqual(
            self.store.replay("rollout-1", start, [stop.commit_id]), stop.runtime.checkpoint_id
        )
        with self.assertRaises(WriterRuntimeError):
            writer.prepare_request(stop.runtime, b"retry")
        with self.assertRaises(WriterRuntimeError):
            writer.step_tool(stop.runtime)

    def test_zero_token_capacity_rejected_before_preparation(self):
        writer, runtime, start = self.entry_with_budget(limits={"generated_tokens": 0})
        with self.assertRaisesRegex(WriterRuntimeError, "before sampling"):
            writer.prepare_request(runtime, b"must not prepare")
        stopped = writer.stop_exhausted(runtime)
        self.assertEqual(
            self.store.get_artifact(stopped.runtime.state.outcome_ref)["stop_reason"],
            "generated_tokens_budget",
        )
        self.assertEqual(
            self.store.replay("rollout-1", start, [stopped.commit_id]),
            stopped.runtime.checkpoint_id,
        )

    def test_total_token_overshoot_retains_usage_and_rejects_retry(self):
        writer, runtime, start = self.entry_with_budget(limits={"total_tokens": 3})
        prepared = writer.prepare_request(runtime, b"request")
        stopped = writer.submit_action(
            runtime,
            self.action(content="sampled reply"),
            prepared_request_ref=prepared,
            raw_output=b"sampled reply",
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )
        self.assertEqual(
            self.store.get_artifact(stopped.runtime.state.outcome_ref)["stop_reason"],
            "total_tokens_budget",
        )
        self.assertEqual(
            self.store.get_artifact(stopped.runtime.state.budgets_ref)["consumed"]["total_tokens"],
            4,
        )
        self.assertEqual(self.record(stopped)["usage"]["total_tokens"], 4)
        self.assertEqual(
            self.store.replay("rollout-1", start, [stopped.commit_id]),
            stopped.runtime.checkpoint_id,
        )
        with self.assertRaisesRegex(WriterRuntimeError, "stale runtime handle"):
            writer.submit_action(runtime, self.action(content="retry"))

    def test_sampled_overrun_is_durable_even_with_bad_batch_envelope(self):
        writer, runtime, start = self.entry_with_budget(limits={"generated_tokens": 1})
        stopped = writer.submit_action(
            runtime,
            {"role": "assistant", "content": "", "tool_calls": None},
            raw_output=b"bad parsed envelope",
            usage={"completion_tokens": 2},
        )
        self.assertEqual(stopped.runtime.state.position["phase"], "terminal")
        self.assertEqual(self.record(stopped)["usage"]["completion_tokens"], 2)
        self.assertEqual(
            self.store.replay("rollout-1", start, [stopped.commit_id]),
            stopped.runtime.checkpoint_id,
        )

    def test_storage_limit_rejects_write_without_effect(self):
        action = self.writer.submit_action(
            self.runtime,
            self.action(self.call("write_file", {"path": "too-big.txt", "content": "x" * 4096})),
        )
        result = self.writer.step_tool(action.runtime)
        observation = self.record(result)["observation"]
        self.assertFalse(observation["ok"])
        self.assertTrue(observation["valid"])
        self.assertIn("storage budget", observation["error"])
        self.assertEqual(result.runtime.state.files, self.runtime.state.files)
        self.assertEqual(self.record(result)["budget_charge"]["file_byte_delta"], 0)


if __name__ == "__main__":
    unittest.main()
