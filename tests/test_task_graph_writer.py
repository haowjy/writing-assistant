import errno
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
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
from writing_agent.task_graph_projection import (
    ProjectionError,
    project_writer_context,
    validate_action_trace,
    validate_writer_effect,
)
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
    def test_generic_phase2_budget_event_is_not_a_writer_stop(self):
        effect = self.writer._effect(self.runtime.state, changes={})
        effect_ref = self.store.put_artifact(effect)
        event = self.writer._event(
            self.runtime.state,
            "budget_charged",
            effect_ref,
            audience=("controller",),
            actor="environment",
        )
        next_state = self.store._apply_recorded_effect_body(self.runtime.state, event, effect)
        commit = self.store.publish(
            "rollout-1",
            None,
            (event,),
            next_state,
            parent_checkpoint=self.start,
            artifact_refs=(effect_ref,),
        )
        checkpoint = self.store.load_commit(commit).checkpoint
        self.store.restore(checkpoint, self.root / "generic-restored")
        project_writer_context(self.store, self.start, checkpoint)
        self.assertEqual(self.store.replay("rollout-1", self.start, [commit]), checkpoint)

    def test_generic_budget_event_can_follow_a_writer_log_without_claiming_it(self):
        action = self.writer.submit_action(
            self.runtime, self.action(self.call("read_file", {"path": "draft.txt"}))
        )
        effect = self.writer._effect(action.runtime.state, changes={})
        effect_ref = self.store.put_artifact(effect)
        event = self.writer._event(
            action.runtime.state,
            "budget_charged",
            effect_ref,
            audience=("controller",),
            actor="environment",
        )
        next_state = self.store._apply_recorded_effect_body(action.runtime.state, event, effect)
        commit = self.store.publish(
            "rollout-1", action.commit_id, (event,), next_state, artifact_refs=(effect_ref,)
        )
        checkpoint = self.store.load_commit(commit).checkpoint
        self.store.restore(checkpoint, self.root / "generic-after-writer")
        project_writer_context(self.store, self.start, checkpoint)
        self.assertEqual(
            self.store.replay("rollout-1", self.start, [action.commit_id, commit]), checkpoint
        )

    def _assert_mutation_rejected(self, prepare, mutate, submit):
        """The same one-field forgery must fail before CAS and after forced persistence."""
        for force_persistence in (False, True):
            fixture = WriterFixture()
            fixture.setUp()
            try:
                writer, runtime, start = prepare(fixture)
                bypass = (
                    patch.object(task_graph_writer, "project_writer_context")
                    if force_persistence
                    else nullcontext()
                )
                with mutate(fixture, writer), bypass:
                    with self.assertRaises(ProjectionError):
                        submit(fixture, writer, runtime)
                head = fixture.store.read_head("rollout-1")
                if force_persistence:
                    self.assertIsNotNone(head)
                    checkpoint = fixture.store.load_commit(head).checkpoint
                    with self.assertRaises(ProjectionError):
                        fixture.store.restore(checkpoint, fixture.root / "forged-restore")
                    with self.assertRaises(ProjectionError):
                        project_writer_context(fixture.store, start, checkpoint)
                    with self.assertRaises(ProjectionError):
                        fixture.store.replay("rollout-1", start, [head])
                else:
                    self.assertIsNone(head)
                    fixture.store.restore(start, fixture.root / "unchanged-restore")
            finally:
                fixture.doCleanups()

    def test_owned_action_values_reject_before_cas_and_on_recovery(self):
        def prepare(fixture):
            return fixture.writer, fixture.runtime, fixture.start

        def submit(fixture, writer, runtime):
            writer.submit_action(runtime, fixture.action(content="done"))

        for field in ("action_ids", "model_calls"):
            with self.subTest(field=field):

                def mutate(fixture, writer, field=field):
                    original = writer._publish

                    def forged(*args, **kwargs):
                        if field == "action_ids":
                            kwargs["history"]["action_ids"] = ["forged:action:99"]
                        else:
                            budget = fixture.store.get_artifact(kwargs["changes"]["budgets_ref"])
                            budget["consumed"]["model_calls"] = 99
                            kwargs["changes"]["budgets_ref"] = fixture.store.put_artifact(budget)
                        return original(*args, **kwargs)

                    return patch.object(writer, "_publish", forged)

                self._assert_mutation_rejected(prepare, mutate, submit)

    def test_action_and_result_cannot_drop_runtime_log(self):
        for kind in ("writer_action", "tool_result"):
            for force_persistence in (False, True):
                with self.subTest(kind=kind, force_persistence=force_persistence):
                    fixture = WriterFixture()
                    fixture.setUp()
                    try:
                        writer = fixture.writer
                        runtime = fixture.runtime
                        previous = None
                        commits = []
                        if kind == "tool_result":
                            action = writer.submit_action(
                                runtime,
                                fixture.action(fixture.call("read_file", {"path": "draft.txt"})),
                            )
                            runtime = action.runtime
                            previous = action.commit_id
                            commits.append(previous)
                        original = writer._effect

                        def forged(state, *, changes, history=(), delta=(), original=original):
                            changes = dict(changes)
                            if "external_inputs_ref" in changes:
                                changes["external_inputs_ref"] = state.external_inputs_ref
                            return original(state, changes=changes, history=history, delta=delta)

                        bypass = (
                            patch.object(task_graph_writer, "project_writer_context")
                            if force_persistence
                            else nullcontext()
                        )
                        with patch.object(writer, "_effect", forged), bypass:
                            with self.assertRaises(ProjectionError):
                                if kind == "tool_result":
                                    writer.step_tool(runtime)
                                else:
                                    writer.submit_action(runtime, fixture.action(content="done"))
                        head = fixture.store.read_head("rollout-1")
                        if force_persistence:
                            self.assertNotEqual(head, previous)
                            checkpoint = fixture.store.load_commit(head).checkpoint
                            with self.assertRaises(ProjectionError):
                                fixture.store.restore(checkpoint, fixture.root / "forged-restore")
                            with self.assertRaises(ProjectionError):
                                project_writer_context(fixture.store, fixture.start, checkpoint)
                            with self.assertRaises(ProjectionError):
                                fixture.store.replay("rollout-1", fixture.start, [*commits, head])
                        else:
                            self.assertEqual(head, previous)
                            fixture.store.restore(
                                runtime.checkpoint_id, fixture.root / "unchanged-restore"
                            )
                    finally:
                        fixture.doCleanups()

    def test_first_stop_log_reason_and_ordinal_mutation_matrix(self):
        for stop_kind, field in (
            ("sampled", "log"),
            ("sampled", "log_entries"),
            ("sampled", "log_record_ref"),
            ("sampled", "reason"),
            ("sampled", "record_action_id"),
            ("sampled", "budget"),
            ("sampled", "actor"),
            ("exhausted", "log"),
            ("exhausted", "log_entries"),
            ("exhausted", "log_record_ref"),
            ("exhausted", "reason"),
            ("exhausted", "actor"),
        ):
            with self.subTest(stop_kind=stop_kind, field=field):

                def prepare(fixture, stop_kind=stop_kind):
                    limit = {"generated_tokens": 1 if stop_kind == "sampled" else 0}
                    return fixture.entry_with_budget(limits=limit)

                def submit(fixture, writer, runtime, stop_kind=stop_kind):
                    if stop_kind == "sampled":
                        writer.submit_action(
                            runtime,
                            fixture.action(content="overrun"),
                            usage={"completion_tokens": 2},
                        )
                    else:
                        writer.stop_exhausted(runtime)

                def mutate(fixture, writer, field=field, stop_kind=stop_kind):
                    if field == "actor":
                        original_event = writer._event

                        def forged_event(*args, **kwargs):
                            return replace(
                                original_event(*args, **kwargs), id=None, actor="environment"
                            )

                        return patch.object(writer, "_event", forged_event)
                    if field in {"log", "log_entries", "log_record_ref", "budget"}:
                        original = writer._effect

                        def forged(state, *, changes, history=(), delta=()):
                            changes = dict(changes)
                            if field == "log":
                                changes["external_inputs_ref"] = state.external_inputs_ref
                            elif field in {"log_entries", "log_record_ref"}:
                                log = fixture.store.get_artifact(changes["external_inputs_ref"])
                                if field == "log_entries":
                                    log["entries"] = []
                                else:
                                    log["entries"][-1]["record_ref"] = fixture.store.put_artifact(
                                        {"forged": "record"}
                                    )
                                changes["external_inputs_ref"] = fixture.store.put_artifact(log)
                            else:
                                budget = fixture.store.get_artifact(changes["budgets_ref"])
                                budget["consumed"]["model_calls"] = 99
                                changes["budgets_ref"] = fixture.store.put_artifact(budget)
                            return original(state, changes=changes, history=history, delta=delta)

                        return patch.object(writer, "_effect", forged)
                    original = fixture.store.put_artifact
                    record_kind = (
                        "WriterSampledBudgetStopV1"
                        if stop_kind == "sampled"
                        else "WriterExhaustedStopV1"
                    )

                    def forged(body, *args, **kwargs):
                        if isinstance(body, dict) and body.get("record_type") == record_kind:
                            body = dict(body)
                            if field == "reason":
                                body["reason"] = "total_tokens_budget"
                            else:
                                body["action_id"] = "forged:action:99"
                        return original(body, *args, **kwargs)

                    return patch.object(fixture.store, "put_artifact", forged)

                self._assert_mutation_rejected(prepare, mutate, submit)

    def test_action_and_sampled_trace_cross_binding_matrix(self):
        fields = (
            "record.action_id",
            "record.trace_ref",
            "record.request_ref",
            "record.prepared_request_ref",
            "record.raw_output_ref",
            "record.logprob_ref",
            "record.usage",
            "record.model",
            "record.seed",
            "trace.action_id",
            "trace.context_content_hash",
            "trace.context_revision_ref",
            "trace.rendering",
            "trace.exact_request_ref",
            "trace.prepared_request_ref",
            "trace.raw_output_ref",
            "trace.logprob_ref",
            "trace.adapter_trace",
            "trace.usage",
            "trace.model",
            "trace.seed",
            "adapter.model",
            "adapter.seed",
            "adapter.usage",
            "adapter.per_token_logprobs_ref",
            "adapter.missing_logprob_ref",
        )
        for stop_kind in ("action", "sampled"):
            for field in fields:
                with self.subTest(stop_kind=stop_kind, field=field):

                    def prepare(fixture, stop_kind=stop_kind):
                        if stop_kind == "sampled":
                            return fixture.entry_with_budget(limits={"generated_tokens": 1})
                        return fixture.writer, fixture.runtime, fixture.start

                    def submit(fixture, writer, runtime):
                        logprob = fixture.store.put_bytes_artifact(b"original-logprobs")
                        prepared = writer.prepare_request(runtime, b"exact request")
                        writer.submit_action(
                            runtime,
                            fixture.action(content="sampled text"),
                            prepared_request_ref=prepared,
                            raw_output=b"raw output",
                            trace={
                                "model": "model-a",
                                "seed": 7,
                                "usage": {"completion_tokens": 2},
                                "per_token_logprobs_ref": logprob,
                            },
                            usage={"completion_tokens": 2},
                        )

                    def mutate(fixture, writer, field=field, stop_kind=stop_kind):
                        alternate_ref = fixture.store.put_artifact({"alternate": True})
                        alternate_logprob = fixture.store.put_bytes_artifact(b"other-logprobs")
                        original = fixture.store.put_artifact

                        def forged(body, *args, **kwargs):
                            record_kind = (
                                "WriterActionV1"
                                if stop_kind == "action"
                                else "WriterSampledBudgetStopV1"
                            )
                            if isinstance(body, dict) and (
                                body.get("record_type") == "WriterActionTraceV1"
                                and not field.startswith("record.")
                                or body.get("record_type") == record_kind
                                and field.startswith("record.")
                            ):
                                body = dict(body)
                                if field.startswith("record."):
                                    name = field.removeprefix("record.")
                                    body[name] = {
                                        "action_id": "forged:action:99",
                                        "trace_ref": alternate_ref,
                                        "request_ref": alternate_ref,
                                        "prepared_request_ref": alternate_ref,
                                        "raw_output_ref": alternate_ref,
                                        "logprob_ref": alternate_logprob,
                                        "usage": {"completion_tokens": 99},
                                        "model": "model-b",
                                        "seed": 8,
                                    }[name]
                                elif (
                                    field.startswith("trace.")
                                    and body.get("record_type") == "WriterActionTraceV1"
                                ):
                                    name = field.removeprefix("trace.")
                                    body[name] = {
                                        "action_id": "forged:action:99",
                                        "context_content_hash": "0" * 64,
                                        "context_revision_ref": alternate_ref,
                                        "rendering": {"projection_version": "forged"},
                                        "exact_request_ref": alternate_ref,
                                        "prepared_request_ref": alternate_ref,
                                        "raw_output_ref": alternate_ref,
                                        "logprob_ref": alternate_logprob,
                                        "adapter_trace": None,
                                        "usage": {"completion_tokens": 99},
                                        "model": "model-b",
                                        "seed": 8,
                                    }[name]
                                elif (
                                    field.startswith("adapter.")
                                    and body.get("record_type") == "WriterActionTraceV1"
                                ):
                                    adapter = dict(body["adapter_trace"])
                                    name = field.removeprefix("adapter.")
                                    if name == "missing_logprob_ref":
                                        adapter.pop("per_token_logprobs_ref")
                                    else:
                                        adapter[name] = {
                                            "model": "model-b",
                                            "seed": 8,
                                            "usage": {"completion_tokens": 99},
                                            "per_token_logprobs_ref": alternate_logprob,
                                        }[name]
                                    body["adapter_trace"] = adapter
                            return original(body, *args, **kwargs)

                        return patch.object(fixture.store, "put_artifact", forged)

                    self._assert_mutation_rejected(prepare, mutate, submit)

    def test_action_authority_and_trace_forgery_fail_before_publication(self):
        original = self.writer._publish
        for forged in ("outcome_ref", "trace_action_id"):
            with self.subTest(forged=forged):

                def publish(*args, forged=forged, **kwargs):
                    if forged == "outcome_ref":
                        kwargs["changes"]["outcome_ref"] = self.store.put_artifact(
                            {
                                "task_status": "complete",
                                "execution_status": "valid",
                                "reward_status": "accepted",
                            }
                        )
                    else:
                        record = kwargs["record"]
                        trace = self.store.get_artifact(record["trace_ref"])
                        trace["action_id"] = "forged:action:99"
                        record["trace_ref"] = self.store.put_artifact(trace)
                    return original(*args, **kwargs)

                with patch.object(self.writer, "_publish", publish):
                    with self.assertRaises(ProjectionError):
                        self.writer.submit_action(self.runtime, self.action(content="done"))
                self.assertIsNone(self.store.read_head("rollout-1"))

    def test_action_authority_and_trace_forgery_fail_restore_projection_replay(self):
        for forged in ("outcome_ref", "trace_action_id"):
            with self.subTest(forged=forged):
                fixture = WriterFixture()
                fixture.setUp()
                try:
                    original = fixture.writer._publish

                    def publish(*args, forged=forged, fixture=fixture, original=original, **kwargs):
                        if forged == "outcome_ref":
                            kwargs["changes"]["outcome_ref"] = fixture.store.put_artifact(
                                {
                                    "task_status": "complete",
                                    "execution_status": "valid",
                                    "reward_status": "accepted",
                                }
                            )
                        else:
                            record = kwargs["record"]
                            trace = fixture.store.get_artifact(record["trace_ref"])
                            trace["action_id"] = "forged:action:99"
                            record["trace_ref"] = fixture.store.put_artifact(trace)
                        return original(*args, **kwargs)

                    with (
                        patch.object(fixture.writer, "_publish", publish),
                        patch.object(task_graph_writer, "project_writer_context"),
                    ):
                        with self.assertRaises(ProjectionError):
                            fixture.writer.submit_action(
                                fixture.runtime, fixture.action(content="done")
                            )
                    head = fixture.store.read_head("rollout-1")
                    checkpoint = fixture.store.load_commit(head).checkpoint
                    with self.assertRaises(ProjectionError):
                        fixture.store.restore(checkpoint, fixture.root / "forged-restore")
                    with self.assertRaises(ProjectionError):
                        project_writer_context(fixture.store, fixture.start, checkpoint)
                    with self.assertRaises(ProjectionError):
                        fixture.store.replay("rollout-1", fixture.start, [head])
                finally:
                    fixture.doCleanups()

    def test_complete_writer_event_field_ownership_and_phases(self):
        action = self.writer.submit_action(
            self.runtime, self.action(self.call("read_file", {"path": "draft.txt"}))
        )
        event = self.store.load_event(action.event_id)
        effect = self.store.get_artifact(event.payload_ref)
        action_state = self.store._apply_recorded_effect_body(self.runtime.state, event, effect)
        for field in set(self.runtime.state.to_dict()) - {
            "continuation",
            "position",
            "budgets_ref",
            "external_inputs_ref",
        }:
            with self.subTest(field=field):
                forged = {
                    **effect,
                    "set": {**effect["set"], field: self.runtime.state.to_dict()[field]},
                }
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(self.runtime.state, action_state, event, forged)
        for field in set(self.runtime.state.history) - {"head", "seq", "action_ids"}:
            with self.subTest(history=field):
                forged = {
                    **effect,
                    "history_set": {
                        **effect["history_set"],
                        field: self.runtime.state.history[field],
                    },
                }
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(self.runtime.state, action_state, event, forged)
        wrong = self.runtime.state.to_dict()
        wrong["position"]["phase"] = "checking"
        with self.assertRaisesRegex(ProjectionError, "pre-phase"):
            validate_writer_effect(EnvironmentStateV1.from_dict(wrong), action_state, event, effect)
        wrong_post = action_state.to_dict()
        wrong_post["position"]["phase"] = "checking"
        with self.assertRaises(ProjectionError):
            validate_writer_effect(
                self.runtime.state, EnvironmentStateV1.from_dict(wrong_post), event, effect
            )
        bad_status = self.runtime.state.to_dict()
        bad_status["outcome_ref"] = self.store.put_artifact(
            {"task_status": "complete", "execution_status": "valid", "reward_status": "accepted"}
        )
        bad_before = EnvironmentStateV1.from_dict(bad_status)
        bad_effect = {**effect, "before_state_ref": bad_before.identity()}
        bad_after = self.store._apply_recorded_effect_body(bad_before, event, bad_effect)
        with self.assertRaisesRegex(ProjectionError, "pre-status"):
            validate_writer_effect(bad_before, bad_after, event, bad_effect, store=self.store)
        position_forgeries = {
            "node_id": "other-node",
            "visit_id": "other-visit",
            "entry_contract": "0" * 64,
            "start_checkpoint": "0" * 64,
            "loop_counts": {"other-loop": 1},
            "lineage_id": "other-lineage",
        }
        for field, value in position_forgeries.items():
            with self.subTest(position=field):
                forged = action_state.to_dict()
                forged["position"][field] = value
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(
                        self.runtime.state, EnvironmentStateV1.from_dict(forged), event, effect
                    )
        continuation_forgeries = {
            "author_request": "0" * 64,
            "check_requests": ["0" * 64],
            "external_requests": ["other-request"],
            "applied_responses": ["other-response"],
            "feedback_cursor": 1,
        }
        for field, value in continuation_forgeries.items():
            with self.subTest(continuation=field):
                forged = action_state.to_dict()
                forged["continuation"][field] = value
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(
                        self.runtime.state, EnvironmentStateV1.from_dict(forged), event, effect
                    )
        result = self.writer.step_tool(action.runtime)
        result_event = self.store.load_event(result.event_id)
        result_effect = self.store.get_artifact(result_event.payload_ref)
        result_state = self.store._apply_recorded_effect_body(
            action.runtime.state, result_event, result_effect
        )
        wrong_result = result_state.to_dict()
        wrong_result["position"]["phase"] = "checking"
        with self.assertRaises(ProjectionError):
            validate_writer_effect(
                action.runtime.state,
                EnvironmentStateV1.from_dict(wrong_result),
                result_event,
                result_effect,
            )
        for field in set(action.runtime.state.to_dict()) - {
            "continuation",
            "position",
            "budgets_ref",
            "external_inputs_ref",
        }:
            with self.subTest(result_field=field):
                forged = {
                    **result_effect,
                    "set": {**result_effect["set"], field: action.runtime.state.to_dict()[field]},
                }
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(action.runtime.state, result_state, result_event, forged)
        for field in set(action.runtime.state.history) - {"head", "seq", "tool_result_ids"}:
            with self.subTest(result_history=field):
                forged = {
                    **result_effect,
                    "history_set": {
                        **result_effect["history_set"],
                        field: action.runtime.state.history[field],
                    },
                }
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(action.runtime.state, result_state, result_event, forged)
        fixture = WriterFixture()
        fixture.setUp()
        try:
            writer, runtime, _ = fixture.entry_with_budget(consumed={"writer_turns": 5})
            stop = writer.stop_exhausted(runtime)
            stop_event = fixture.store.load_event(stop.event_id)
            stop_effect = fixture.store.get_artifact(stop_event.payload_ref)
            stop_state = fixture.store._apply_recorded_effect_body(
                runtime.state, stop_event, stop_effect
            )
            wrong_stop = stop_state.to_dict()
            wrong_stop["position"]["phase"] = "ready_writer"
            with self.assertRaises(ProjectionError):
                validate_writer_effect(
                    runtime.state,
                    EnvironmentStateV1.from_dict(wrong_stop),
                    stop_event,
                    stop_effect,
                    store=fixture.store,
                )
            for field in set(runtime.state.to_dict()) - {
                "position",
                "outcome_ref",
                "external_inputs_ref",
            }:
                with self.subTest(stop_field=field):
                    forged = {
                        **stop_effect,
                        "set": {**stop_effect["set"], field: runtime.state.to_dict()[field]},
                    }
                    with self.assertRaises(ProjectionError):
                        validate_writer_effect(runtime.state, stop_state, stop_event, forged)
            for field in set(runtime.state.history) - {"head", "seq"}:
                with self.subTest(stop_history=field):
                    forged = {**stop_effect, "history_set": {field: runtime.state.history[field]}}
                    with self.assertRaises(ProjectionError):
                        validate_writer_effect(runtime.state, stop_state, stop_event, forged)
        finally:
            fixture.doCleanups()

    def test_sampled_stop_ownership_and_all_post_phases(self):
        writer, runtime, _ = self.entry_with_budget(limits={"generated_tokens": 1})
        stop = writer.submit_action(
            runtime,
            self.action(content="overrun"),
            usage={"completion_tokens": 2, "total_tokens": 2},
        )
        event = self.store.load_event(stop.event_id)
        effect = self.store.get_artifact(event.payload_ref)
        after = self.store._apply_recorded_effect_body(runtime.state, event, effect)
        for field in set(runtime.state.to_dict()) - {
            "position",
            "budgets_ref",
            "outcome_ref",
            "external_inputs_ref",
        }:
            with self.subTest(field=field):
                forged = {**effect, "set": {**effect["set"], field: runtime.state.to_dict()[field]}}
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(runtime.state, after, event, forged, store=self.store)
        for field in set(runtime.state.history) - {"head", "seq"}:
            with self.subTest(history=field):
                forged = {**effect, "history_set": {field: runtime.state.history[field]}}
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(runtime.state, after, event, forged, store=self.store)
        for phase in ("checking", "awaiting_author", "ready_transition"):
            with self.subTest(post_phase=phase):
                changed = after.to_dict()
                changed["position"]["phase"] = phase
                with self.assertRaises(ProjectionError):
                    validate_writer_effect(
                        runtime.state,
                        EnvironmentStateV1.from_dict(changed),
                        event,
                        effect,
                        store=self.store,
                    )

    def test_trace_claims_are_bound_to_one_action(self):
        prepared_ref = self.writer.prepare_request(self.runtime, {"prompt": "pinned"})
        action = self.writer.submit_action(
            self.runtime,
            self.action(content="done"),
            prepared_request_ref=prepared_ref,
            raw_output=b"done",
            trace={"model": "test-model", "seed": 7, "generated_token_ids": [1]},
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )
        record = self.record(action)
        trace = self.store.get_artifact(record["trace_ref"])
        claims = {
            "record_type": "OtherTraceV1",
            "action_id": "forged:action:99",
            "exact_request_ref": None,
            "prepared_request_ref": None,
            "raw_output_ref": None,
            "logprob_ref": "0" * 64,
            "usage": {"total_tokens": 999},
            "model": "other-model",
            "seed": 8,
            "rendering": {"projection_version": "forged"},
            "context_revision_ref": "0" * 64,
            "context_content_hash": "0" * 64,
            "raw_output_evidence": "missing",
            "token_evidence": "missing",
        }
        for field, value in claims.items():
            with self.subTest(trace_field=field):
                forged = {**trace, field: value}
                with self.assertRaises(ProjectionError):
                    validate_action_trace(
                        self.store,
                        record,
                        forged,
                        record["action_id"],
                        self.runtime.context.content_hash,
                        self.runtime.context.identity(),
                        self.runtime.context.rendering,
                        action.runtime.context.messages[-1],
                    )
        # Presence is also a claim: an adapter cannot supply logprobs when the
        # owning record and outer trace explicitly say they are absent.
        forged = {
            **trace,
            "adapter_trace": {
                **trace["adapter_trace"],
                "per_token_logprobs_ref": self.store.put_bytes_artifact(b"unowned-logprobs"),
            },
        }
        with self.assertRaises(ProjectionError):
            validate_action_trace(
                self.store,
                record,
                forged,
                record["action_id"],
                self.runtime.context.content_hash,
                self.runtime.context.identity(),
                self.runtime.context.rendering,
                action.runtime.context.messages[-1],
            )

    def test_parent_file_conflict_commits_but_decode_corruption_interrupts(self):
        action = self.writer.submit_action(
            self.runtime,
            self.action(self.call("write_file", {"path": "draft.txt/child", "content": "x"})),
        )
        result = self.writer.step_tool(action.runtime)
        self.assertFalse(self.record(result)["observation"]["ok"])
        self.assertTrue(self.record(result)["observation"]["valid"])
        self.assertEqual(result.runtime.state.files, self.runtime.state.files)
        self.assertEqual(result.runtime.state.continuation["next_call"], 1)
        self.assertEqual(
            self.store.get_artifact(result.runtime.state.budgets_ref)["consumed"][
                "attempted_tool_calls"
            ],
            1,
        )
        second = self.writer.submit_action(
            result.runtime, self.action(self.call("read_file", {"path": "draft.txt"}, "backend-2"))
        )
        head = self.store.read_head("rollout-1")
        with patch.object(
            Workspace, "read_file", side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        ):
            with self.assertRaises(UnicodeDecodeError):
                self.writer.step_tool(second.runtime)
        self.assertEqual(self.store.read_head("rollout-1"), head)
        self.assertEqual(second.runtime.state.continuation["next_call"], 0)
        self.assertEqual(
            self.store.get_artifact(second.runtime.state.budgets_ref)["consumed"][
                "attempted_tool_calls"
            ],
            1,
        )
        fixture = WriterFixture()
        fixture.setUp()
        try:
            search = fixture.writer.submit_action(
                fixture.runtime,
                fixture.action(fixture.call("search", {"query": "alpha"})),
            )
            head = fixture.store.read_head("rollout-1")
            with patch.object(
                Workspace,
                "read_file",
                side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
            ):
                with self.assertRaises(UnicodeDecodeError):
                    fixture.writer.step_tool(search.runtime)
            self.assertEqual(fixture.store.read_head("rollout-1"), head)
            self.assertEqual(search.runtime.state.continuation["next_call"], 0)
        finally:
            fixture.doCleanups()
        legacy = Workspace(self.root / "legacy-search")
        (legacy.root / "corrupt.txt").write_bytes(b"\xff")
        self.assertEqual(legacy.search("alpha"), [])

    def test_decoder_depth_is_one_charged_nonmutating_result(self):
        for arguments in ("[" * 20_000 + "0" + "]" * 20_000, "[" * 70_000):
            with self.subTest(length=len(arguments)):
                fixture = WriterFixture()
                fixture.setUp()
                try:
                    action = fixture.writer.submit_action(
                        fixture.runtime,
                        fixture.action(fixture.call("write_file", arguments)),
                    )
                    result = fixture.writer.step_tool(action.runtime)
                    self.assertFalse(
                        fixture.store.get_artifact(result.record_ref)["observation"]["valid"]
                    )
                    self.assertEqual(result.runtime.state.files, fixture.runtime.state.files)
                    self.assertEqual(result.runtime.state.continuation["next_call"], 1)
                    self.assertEqual(
                        fixture.store.get_artifact(result.runtime.state.budgets_ref)["consumed"][
                            "attempted_tool_calls"
                        ],
                        1,
                    )
                finally:
                    fixture.doCleanups()
        nested: object = "leaf"
        for _ in range(100):
            nested = [nested]
        action = self.writer.submit_action(
            self.runtime,
            self.action(self.call("write_file", nested)),
        )
        result = self.writer.step_tool(action.runtime)
        self.assertFalse(self.record(result)["observation"]["valid"])
        self.assertEqual(result.runtime.state.files, self.runtime.state.files)
        self.assertEqual(result.runtime.state.continuation["next_call"], 1)

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

        with (
            patch.object(self.writer, "_publish", forged),
            patch.object(task_graph_writer, "project_writer_context"),
        ):
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
            "cursor",
            "message_origin",
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
                        elif field == "cursor":
                            kwargs["changes"]["continuation"]["next_call"] = 0
                        elif field == "message_origin":
                            message = kwargs["message"].to_dict()
                            message["origin"] = "unrelated:action:99"
                            kwargs["message"] = MessageV1.from_dict(message)
                        else:
                            record[field] = {"draft.txt": {"before": "alpha\n", "after": "forged"}}
                        return original(*args, **kwargs)

                    with (
                        patch.object(fixture.writer, "_publish", forged),
                        patch.object(task_graph_writer, "project_writer_context"),
                    ):
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

    def test_all_seven_result_forgeries_reject_before_head_publication(self):
        for field in (
            "action_id",
            "before_execution_hash",
            "after_execution_hash",
            "file_delta",
            "cursor",
            "budget_charge",
            "message_origin",
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
                        if field == "action_id":
                            record["action_id"] = "forged:action:99"
                        elif field in {"before_execution_hash", "after_execution_hash"}:
                            record[field] = "0" * 64
                        elif field == "file_delta":
                            record[field] = {"fake": {"before": None, "after": "x"}}
                        elif field == "cursor":
                            kwargs["changes"]["continuation"]["next_call"] = 0
                        elif field == "budget_charge":
                            record[field]["tool_calls"] = 0
                        else:
                            message = kwargs["message"].to_dict()
                            message["origin"] = "forged:action:99"
                            kwargs["message"] = MessageV1.from_dict(message)
                        return original(*args, **kwargs)

                    with patch.object(fixture.writer, "_publish", forged):
                        with self.assertRaises(ProjectionError):
                            fixture.writer.step_tool(action.runtime)
                    self.assertEqual(fixture.store.read_head("rollout-1"), action.commit_id)
                    self.assertEqual(action.runtime.state.continuation["next_call"], 0)
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
