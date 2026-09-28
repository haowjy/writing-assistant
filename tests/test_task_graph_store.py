import json
import os
import shutil
import stat
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from tests.task_graph_store_fixtures import PatchVerifier
from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    EnvironmentStateV1,
    EventV1,
    GraphInstanceV1,
    MaterializedContextV1,
    MessageV1,
    NodeSpecV1,
    canonical_bytes,
    canonical_json,
    domain_hash,
    tree_hash,
)
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_record_contracts import SEMANTICS_V1, ExecutionVersionsV1
from writing_agent.task_graph_records import ContextContentV1, ContextRevisionV1, OutcomeV1
from writing_agent.task_graph_store import (
    ConcurrentUpdateError,
    CorruptRecordError,
    MaterializationError,
    MissingReferenceError,
    TaskGraphStore,
    WrongRecordDomainError,
    _ClosureValidator,
)


class StoreFixture:
    def __init__(self, root: Path):
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.store = TaskGraphStore(root / "store", verifier=PatchVerifier())
        self.common = {}
        for name in (
            "entry",
            "template",
            "tokenizer",
            "tools",
            "decisions",
            "disclosures",
            "versions",
            "budgets",
            "rng",
            "external",
            "outcome",
            "provenance",
        ):
            self.common[name] = self.store.put_artifact({"fixture": name})
        self.common["versions"] = self.store.put_artifact(
            ExecutionVersionsV1(
                1,
                SEMANTICS_V1,
                self.common["entry"],
                {"max_file_bytes": 4096, "max_workspace_bytes": 8192},
            ).to_wire()
        )
        self.private = self.store.put_artifact({"fixture": "author-packet"}, private=True)
        self.common["requirements"] = self.store.put_artifact(
            {"fixture": "requirements"}, private=True
        )
        self.instance = GraphInstanceV1(
            template_ref=self.common["template"],
            entry_node="write",
            nodes=(NodeSpecV1(id="write", entry_contract=self.common["entry"]),),
        )
        self.store.persist(self.instance)
        self.context = self.make_context("Please revise the draft.")

    def make_context(self, text: str, *, event_head=None, provenance=()):
        node = ContextContentV1(
            parent_ref=None,
            messages=(MessageV1(content=(text,), origin="request:1"),),
            tools=(),
            rendering={
                "projection_version": "v1",
                "prefix_id": "root",
                "template_ref": self.common["template"],
                "tokenizer_ref": self.common["tokenizer"],
                "tool_schema_ref": self.common["tools"],
            },
        )
        self.store.persist(node)
        context = ContextRevisionV1(
            content_ref=node.identity(),
            event_head=event_head,
            provenance_refs=tuple(provenance),
        )
        self.store.persist(context)
        self.materialized_context = MaterializedContextV1(node.messages, node.tools, node.rendering)
        return context

    def state(
        self,
        *,
        lineage="seed",
        files=None,
        context=None,
        head=None,
        seq=0,
        start=None,
        branch_base=None,
        history_changes=None,
        position_changes=None,
        author_packet=True,
    ):
        files = {"draft.txt": "alpha"} if files is None else files
        history = {
            "head": head,
            "seq": seq,
            "branch_base": branch_base,
            "imported_refs": (),
            "action_ids": (),
            "tool_result_ids": (),
        }
        history.update(history_changes or {})
        position = {
            "node_id": "write",
            "visit_id": "visit-1",
            "phase": "ready_writer",
            "entry_contract": self.common["entry"],
            "start_checkpoint": start,
            "loop_counts": {},
            "lineage_id": lineage,
        }
        position.update(position_changes or {})
        return EnvironmentStateV1(
            instance_ref=self.instance.identity(),
            position=position,
            files=files,
            tree_hash=tree_hash(files),
            history=history,
            context_ref=(context or self.context).identity(),
            requirements_ref=self.common["requirements"],
            decisions_ref=self.common["decisions"],
            disclosures_ref=self.common["disclosures"],
            author_packet_ref=self.private if author_packet else None,
            versions_ref=self.common["versions"],
            budgets_ref=self.common["budgets"],
            rng_ref=self.common["rng"],
            external_inputs_ref=self.common["external"],
            outcome_ref=self.common["outcome"],
            provenance_ref=self.common["provenance"],
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

    def root(self, *, files=None, lineage="seed"):
        state = self.state(files=files, lineage=lineage)
        return self.store.save_checkpoint(state), state

    def event_state(
        self,
        before,
        *,
        lineage,
        file_delta=None,
        set_values=None,
        history_set=None,
        kind="tool_result",
    ):
        payload_ref = self.store.put_artifact({"store_test_event": kind})
        event = EventV1(
            previous=before.history["head"],
            seq=before.history["seq"] + 1,
            lineage_id=lineage,
            kind=kind,
            audience=("controller",),
            payload_ref=payload_ref,
            versions_ref=self.common["versions"],
            provenance_ref=self.common["provenance"],
        )
        body = json.loads(canonical_json(before.to_dict()))
        body.update(json.loads(canonical_json(set_values or {})))
        files = dict(before.files)
        for path, delta in (file_delta or {}).items():
            if delta["after"] is None:
                files.pop(path, None)
            else:
                files[path] = delta["after"]
        body["files"] = files
        body["tree_hash"] = tree_hash(files)
        history = dict(body["history"])
        history.update(json.loads(canonical_json(history_set or {})))
        history.update(head=event.identity(), seq=event.seq)
        body["history"] = history
        return event, EnvironmentStateV1.from_dict(body), payload_ref


class TaskGraphStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = StoreFixture(self.root)
        self.store = self.fixture.store

    def tearDown(self):
        self.temporary.cleanup()

    def test_store_requires_a_verifier_at_construction(self):
        root = self.root / "missing-verifier"
        with self.assertRaises(TypeError):
            TaskGraphStore(root)
        self.assertFalse(root.exists())
        with self.assertRaises(TypeError):
            TaskGraphStore(root, verifier=None)
        self.assertFalse(root.exists())

    def test_runtime_lineage_requires_transition_semantics_pin(self):
        state = self.fixture.state()
        old_versions = self.store.put_artifact(
            {
                "schema": 1,
                "admission_policy_ref": self.fixture.common["entry"],
                "tool_spec": {"max_file_bytes": 4096, "max_workspace_bytes": 8192},
            }
        )
        legacy_state = replace(state, versions_ref=old_versions)
        with self.assertRaisesRegex(
            ProjectionError,
            "state.versions_ref.transition_semantics: runtime lineages require "
            "task-graph-derive-v1",
        ):
            self.store.save_checkpoint(legacy_state)

    def test_operation_reuses_verified_artifacts_and_returns_deep_copies(self):
        value = {"nested": [{"items": ["original"]}]}
        identity = self.store.put_artifact(value)
        read_bytes = self.store._read_bytes

        with (
            mock.patch.dict(os.environ, {"CWA_TASK_GRAPH_AUDIT_SCOPE_EXIT": "1"}),
            mock.patch.object(self.store, "_read_bytes", wraps=read_bytes) as read,
        ):
            with self.store.operation():
                returned = self.store.get_artifact(identity)
                returned["nested"][0]["items"].append("caller mutation")
                self.assertEqual(self.store.get_artifact(identity), value)
            self.assertEqual(read.call_count, 1)
            self.assertIsNone(getattr(self.store._session, "validator", None))

            # Verification is deliberately operation-scoped: the next read
            # must read and hash the artifact again.
            self.assertEqual(self.store.get_artifact(identity), value)
            self.assertEqual(read.call_count, 2)

    def test_scope_exit_audit_does_not_replace_operation_error(self):
        with (
            mock.patch.dict(os.environ, {"CWA_TASK_GRAPH_AUDIT_SCOPE_EXIT": "1"}),
            mock.patch(
                "writing_agent.task_graph_store._ClosureValidator.audit_artifacts",
                side_effect=CorruptRecordError("audit failure"),
            ) as audit,
        ):
            with self.assertRaises(LookupError):
                with self.store.operation():
                    raise LookupError("operation failure")
        audit.assert_not_called()
        self.assertIsNone(getattr(self.store._session, "validator", None))

    def test_closure_validator_discards_rejected_candidate_state(self):
        validator = _ClosureValidator(self.store)
        keys = {("event", "1" * 64), ("commit", "2" * 64)}
        validator.virtual.update({key: object() for key in keys})
        validator.loaded.update({key: object() for key in keys})
        validator.completed.update(keys)
        validator.active.update(keys)
        validator.resolved[("artifact", "alias")] = ("event", "1" * 64)
        validator.resolved[("commit", "2" * 64)] = ("checkpoint", "3" * 64)

        validator.discard(keys)

        self.assertFalse(validator.virtual)
        self.assertFalse(validator.loaded)
        self.assertFalse(validator.completed)
        self.assertFalse(validator.active)
        self.assertFalse(validator.resolved)

    def publish_change(self, lineage="main", expected=None, parent=None, before=None, text="beta"):
        if before is None:
            parent, before = self.fixture.root()
        event, after, effect = self.fixture.event_state(
            before,
            lineage=lineage,
            file_delta={"draft.txt": {"before": before.files["draft.txt"], "after": text}},
            set_values={"position": {**before.position, "lineage_id": lineage}},
            history_set={"tool_result_ids": (*before.history["tool_result_ids"], f"{lineage}:r")},
        )
        commit = self.store.publish(
            lineage,
            expected,
            (event,),
            after,
            parent_checkpoint=parent if expected is None else None,
        )
        return commit, self.store.load_commit(commit).checkpoint, after, event, effect

    def test_missing_private_artifact_corrupt_bytes_and_wrong_domain_fail_closed(self):
        checkpoint, state = self.fixture.root()
        (self.store.root / "private" / state.author_packet_ref).unlink()
        with self.assertRaises(MissingReferenceError):
            self.store.load_checkpoint(checkpoint)

        wrong_payload = self.store.put_artifact(
            MessageV1(content=("x",), origin="a").to_dict(), domain="message"
        )
        event = EventV1(
            lineage_id="line",
            kind="tool_result",
            audience=("controller",),
            payload_ref=wrong_payload,
            versions_ref=self.fixture.common["versions"],
            provenance_ref=self.fixture.common["provenance"],
        )
        self.store.persist(event)
        with self.assertRaises(WrongRecordDomainError):
            self.store.load_event(event.identity())

        context_path = (
            self.store.root / "context_revisions" / f"{self.fixture.context.identity()}.json"
        )
        context_path.write_bytes(b"not-json")
        with self.assertRaises(CorruptRecordError):
            self.store.load_context_revision(self.fixture.context.identity())

    def test_event_record_byte_flip_is_corrupt_record(self):
        event = EventV1(
            lineage_id="flipped-event",
            kind="tool_result",
            audience=("controller",),
            payload_ref=self.fixture.common["entry"],
            versions_ref=self.fixture.common["versions"],
            provenance_ref=self.fixture.common["provenance"],
        )
        identity = self.store.persist(event)
        path = self.store.root / "events" / f"{identity}.json"
        damaged = bytearray(path.read_bytes())
        marker = damaged.index(b"tool_result")
        damaged[marker] = ord("x")
        path.write_bytes(damaged)

        with self.assertRaises(CorruptRecordError):
            self.store.load_event(identity)

    def test_hash_correct_forged_event_payload_is_projection_error(self):
        outcome = OutcomeV1(
            1,
            "unknown",
            "running",
            None,
            "pending",
            "pending",
            None,
            None,
            (),
            None,
            None,
            None,
            None,
        ).to_wire()
        outcome["forged_field"] = "not codec-owned"
        payload_ref = domain_hash("payload", outcome)
        self.store._write_immutable(
            self.store._artifact_path(payload_ref, False),
            canonical_bytes(
                {
                    "schema": 1,
                    "domain": "payload",
                    "encoding": "json",
                    "body": outcome,
                }
            ),
        )
        event = EventV1(
            lineage_id="forged-event",
            kind="tool_result",
            audience=("controller",),
            payload_ref=payload_ref,
            versions_ref=self.fixture.common["versions"],
            provenance_ref=self.fixture.common["provenance"],
        )
        identity = self.store.persist(event)

        with self.assertRaisesRegex(ProjectionError, "event.payload_ref.forged_field"):
            self.store.load_event(identity)

    def test_stored_symlink_and_unsafe_path_are_rejected(self):
        checkpoint, _ = self.fixture.root()
        checkpoint_path = self.store.root / "checkpoints" / f"{checkpoint}.json"
        backup = self.root / "checkpoint-copy"
        shutil.copyfile(checkpoint_path, backup)
        checkpoint_path.unlink()
        checkpoint_path.symlink_to(backup)
        with self.assertRaises(CorruptRecordError):
            self.store.load_checkpoint(checkpoint)

        state = self.fixture.state()
        body = state.to_dict()
        body["files"] = {"../escape": "x"}
        body["tree_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            EnvironmentStateV1.from_dict(body)

    def test_event_sequence_and_commit_order_closure(self):
        root, before = self.fixture.root()
        event, after, effect = self.fixture.event_state(
            before,
            lineage="line",
            file_delta={"draft.txt": {"before": "alpha", "after": "beta"}},
            set_values={"position": {**before.position, "lineage_id": "line"}},
        )
        self.store.persist(event)
        checkpoint = self.store.save_checkpoint(after, parent=root)
        bad = CommitV1(events=(event.identity(), event.identity()), checkpoint=checkpoint)
        self.store.persist(bad)
        with self.assertRaises(CorruptRecordError):
            self.store.load_commit(bad.identity())

        skipped = EventV1(
            previous=event.identity(),
            seq=event.seq + 2,
            lineage_id="line",
            kind="tool_result",
            audience=("controller",),
            payload_ref=effect,
            versions_ref=self.fixture.common["versions"],
            provenance_ref=self.fixture.common["provenance"],
        )
        self.store.persist(skipped)
        invalid_body = after.to_dict()
        invalid_body["history"].update(head=skipped.identity(), seq=skipped.seq)
        invalid_state = EnvironmentStateV1.from_dict(invalid_body)
        with self.assertRaises(CorruptRecordError):
            self.store.save_checkpoint(invalid_state, parent=checkpoint)

    def test_authority_is_exact_projection_and_cas_is_idempotent(self):
        root, before = self.fixture.root()
        event, after, effect = self.fixture.event_state(
            before,
            lineage="main",
            file_delta={"draft.txt": {"before": "alpha", "after": "beta"}},
            set_values={"position": {**before.position, "lineage_id": "main"}},
        )
        arguments = dict(
            lineage_id="main",
            expected_head=None,
            events=(event,),
            next_state=after,
            parent_checkpoint=root,
        )
        commit = self.store.publish(**arguments)
        self.assertEqual(self.store.publish(**arguments), commit)
        authority = json.loads((self.store.root / "refs" / "main.json").read_text())
        self.assertEqual(authority, {"head_commit": commit})
        self.assertNotIn("expected_head", authority)

        competing_event, competing_state, competing_effect = self.fixture.event_state(
            before,
            lineage="main",
            file_delta={"draft.txt": {"before": "alpha", "after": "other"}},
            set_values={"position": {**before.position, "lineage_id": "main"}},
        )
        with self.assertRaises(ConcurrentUpdateError):
            self.store.publish(
                "main",
                None,
                (competing_event,),
                competing_state,
                parent_checkpoint=root,
            )

    def test_competing_updates_allow_exactly_one_winner(self):
        first, first_checkpoint, state, _, _ = self.publish_change()
        barrier = threading.Barrier(2)
        results = []

        def contender(label):
            event, after, effect = self.fixture.event_state(
                state,
                lineage="main",
                file_delta={"draft.txt": {"before": "beta", "after": label}},
            )
            barrier.wait()
            try:
                result = self.store.publish("main", first, (event,), after)
            except ConcurrentUpdateError:
                result = "stale"
            results.append(result)

        threads = [threading.Thread(target=contender, args=(label,)) for label in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results.count("stale"), 1)
        self.assertIn(self.store.read_head("main"), results)
        self.assertEqual(self.store.load_checkpoint(first_checkpoint).state, state)

    def test_fault_boundaries_leave_old_or_new_authority_only(self):
        for stage, published in (
            ("before_immutable_writes", False),
            ("after_immutable_writes", False),
            ("before_head_publication", False),
            ("after_head_publication", True),
        ):
            with self.subTest(stage=stage):
                directory = self.root / stage
                fixture = StoreFixture(directory)
                root, before = fixture.root()
                event, after, effect = fixture.event_state(
                    before,
                    lineage="main",
                    file_delta={"draft.txt": {"before": "alpha", "after": "beta"}},
                    set_values={"position": {**before.position, "lineage_id": "main"}},
                )

                def fail(point, expected_stage=stage):
                    if point == expected_stage:
                        raise RuntimeError("power loss")

                with self.assertRaises(RuntimeError):
                    fixture.store.publish(
                        "main",
                        None,
                        (event,),
                        after,
                        parent_checkpoint=root,
                        fault=fail,
                    )
                head = fixture.store.read_head("main")
                self.assertEqual(head is not None, published)
                authoritative = root if head is None else fixture.store.load_commit(head).checkpoint
                expected = before if head is None else after
                self.assertEqual(fixture.store.load_checkpoint(authoritative).state, expected)

    def test_binary_artifact_and_missing_public_reference_validation(self):
        binary = self.store.put_bytes_artifact(b"\x00\xff", domain="payload")
        self.assertEqual(
            self.store.get_artifact(binary, expected_domain="payload:bytes"), b"\x00\xff"
        )
        checkpoint, state = self.fixture.root()
        (self.store.root / "artifacts" / state.budgets_ref).unlink()
        with self.assertRaises(MissingReferenceError):
            self.store.load_checkpoint(checkpoint)

    def test_supplemental_and_imported_refs_use_typed_transitive_closure(self):
        root, state = self.fixture.root()
        dangling = EventV1(
            lineage_id="seed",
            kind="tool_result",
            audience=("controller",),
            payload_ref="0" * 64,
            versions_ref=self.fixture.common["versions"],
            provenance_ref=self.fixture.common["provenance"],
        )
        self.store.persist(dangling)
        with self.assertRaises(MissingReferenceError):
            self.store.save_checkpoint(state, parent=root, artifact_refs=(dangling.identity(),))

        imported = self.fixture.make_context("Imported context")
        imported_state = self.fixture.state(
            history_changes={"imported_refs": (imported.identity(),)}
        )
        imported_checkpoint = self.store.save_checkpoint(imported_state)
        self.assertEqual(self.store.load_checkpoint(imported_checkpoint).state, imported_state)

        broken_import = self.fixture.make_context("Broken import", event_head="f" * 64)
        broken_state = self.fixture.state(
            history_changes={"imported_refs": (broken_import.identity(),)}
        )
        with self.assertRaises(MissingReferenceError):
            self.store.save_checkpoint(broken_state)

    def test_artifact_schemas_and_locations_fail_closed(self):
        _, state = self.fixture.root()
        malformed = self.store.put_artifact(
            {
                "artifact_type": "StoreTestEventV1",
                "before_state_ref": state.identity(),
                "file_delta": {},
                "set": {},
            }
        )
        with self.assertRaises(ProjectionError):
            self.store.save_checkpoint(state, artifact_refs=(malformed,))

        unsupported = self.store.put_artifact(
            {"artifact_type": "FutureTransitionV9", "some_ref": "0" * 64}
        )
        with self.assertRaises(ProjectionError):
            self.store.save_checkpoint(state, artifact_refs=(unsupported,))

        duplicate = self.store.put_artifact({"duplicate": True})
        self.store.put_artifact({"duplicate": True}, private=True)
        with self.assertRaises(WrongRecordDomainError):
            self.store.save_checkpoint(state, artifact_refs=(duplicate,))

        with self.assertRaises(ValueError):
            self.store.put_artifact(
                EventV1(
                    lineage_id="seed",
                    kind="tool_result",
                    audience=("controller",),
                    payload_ref=self.fixture.common["entry"],
                    versions_ref=self.fixture.common["versions"],
                    provenance_ref=self.fixture.common["provenance"],
                ).to_dict(),
                domain="event",
            )

        event_body = {"record_type": "EventV1", "foreign": True}
        event_ref = domain_hash("event", event_body)
        self.store._artifact_path(event_ref, False).write_bytes(
            canonical_bytes(
                {
                    "schema": 1,
                    "domain": "event",
                    "encoding": "json",
                    "body": event_body,
                }
            )
        )
        with self.assertRaises(WrongRecordDomainError):
            self.store.get_artifact(event_ref)
        with self.assertRaises(WrongRecordDomainError):
            self.store.save_checkpoint(state, artifact_refs=(event_ref,))

    def test_deep_branched_closure_is_iterative_and_reads_each_record_once(self):
        depth = 180
        previous = None
        for sequence in range(1, depth + 1):
            event = EventV1(
                previous=previous,
                seq=sequence,
                lineage_id="deep",
                kind="tool_result",
                audience=("controller",),
                payload_ref=self.fixture.common["entry"],
                versions_ref=self.fixture.common["versions"],
                provenance_ref=self.fixture.common["provenance"],
            )
            self.store.persist(event)
            previous = event.identity()
        state = self.fixture.state(lineage="deep", head=previous, seq=depth)
        first = CheckpointV1(state=state, event_head=previous)
        self.store.persist(first)
        ancestry = [first.identity()]
        for _ in range(depth):
            checkpoint = CheckpointV1(parents=(ancestry[-1],), state=state, event_head=previous)
            self.store.persist(checkpoint)
            ancestry.append(checkpoint.identity())
        branch_a = CheckpointV1(parents=(ancestry[-1],), state=state, event_head=previous)
        branch_b = CheckpointV1(parents=(ancestry[depth // 2],), state=state, event_head=previous)
        self.store.persist(branch_a)
        self.store.persist(branch_b)
        joined = CheckpointV1(
            parents=(branch_a.identity(),),
            state=state,
            event_head=previous,
            artifact_refs=(branch_b.identity(),),
        )
        self.store.persist(joined)

        reads = 0
        original = self.store._read_bytes

        def counted(path):
            nonlocal reads
            reads += 1
            return original(path)

        with mock.patch.object(self.store, "_read_bytes", side_effect=counted):
            loaded = self.store.load_checkpoint(joined.identity())
        self.assertEqual(loaded, joined)
        # Unique records are 184 checkpoints, 180 events, and a small fixed
        # state/context/artifact closure. A repeated traversal would greatly
        # exceed this linear bound.
        self.assertLessEqual(reads, 400)

    def test_durability_failures_are_repaired_by_reopen_and_retry(self):
        def reopen(store):
            return TaskGraphStore(
                store.root,
                verifier=PatchVerifier(),
                max_record_bytes=store.max_record_bytes,
            )

        # Regular-file fsync fails before link: retry writes and flushes anew.
        value = {"durability": "file-fsync"}
        real_fsync = os.fsync
        failed = False

        def fail_regular_once(descriptor):
            nonlocal failed
            if not failed and stat.S_ISREG(os.fstat(descriptor).st_mode):
                failed = True
                raise OSError("injected file fsync failure")
            return real_fsync(descriptor)

        with mock.patch("writing_agent.task_graph_store.os.fsync", side_effect=fail_regular_once):
            with self.assertRaises(OSError):
                self.store.put_artifact(value)
        reopened = reopen(self.store)
        identity = reopened.put_artifact(value)
        self.assertEqual(reopened.get_artifact(identity), value)

        # Link can fail before creation, or report failure after creating the
        # immutable name. Both reopen/retry paths must finish directory fsync.
        for label, after_link in (("link", False), ("uncertain-link", True)):
            item = {"durability": label}
            real_link = os.link
            tripped = False

            def fail_link_once(source, target, *, _after=after_link, _real=real_link):
                nonlocal tripped
                if not tripped:
                    tripped = True
                    if _after:
                        _real(source, target)
                    raise OSError("injected link failure")
                return _real(source, target)

            with mock.patch("writing_agent.task_graph_store.os.link", side_effect=fail_link_once):
                with self.assertRaises(OSError):
                    reopened.put_artifact(item)
            reopened = reopen(reopened)
            item_id = reopened.put_artifact(item)
            self.assertEqual(reopened.get_artifact(item_id), item)

        # The target link exists, but its containing-directory fsync fails.
        item = {"durability": "directory-fsync"}
        real_directory_fsync = reopened._fsync_directory
        tripped = False

        def fail_directory_once(path):
            nonlocal tripped
            if not tripped and path.name == "artifacts":
                tripped = True
                raise OSError("injected directory fsync failure")
            return real_directory_fsync(path)

        with mock.patch.object(reopened, "_fsync_directory", side_effect=fail_directory_once):
            with self.assertRaises(OSError):
                reopened.put_artifact(item)
        reopened = reopen(reopened)
        item_id = reopened.put_artifact(item)
        self.assertEqual(reopened.get_artifact(item_id), item)

        # Atomic replace failures are uncertain: test both before and after the
        # real replace, then retry the identical transaction after reopening.
        for label, after_replace in (("replace", False), ("uncertain-replace", True)):
            directory = self.root / label
            fixture = StoreFixture(directory)
            root, before = fixture.root()
            event, after, effect = fixture.event_state(
                before,
                lineage="main",
                set_values={"position": {**before.position, "lineage_id": "main"}},
            )
            real_replace = os.replace
            tripped = False

            def fail_replace_once(source, target, *, _after=after_replace, _real=real_replace):
                nonlocal tripped
                if not tripped:
                    tripped = True
                    if _after:
                        _real(source, target)
                    raise OSError("injected replace failure")
                return _real(source, target)

            arguments = dict(
                lineage_id="main",
                expected_head=None,
                events=(event,),
                next_state=after,
                parent_checkpoint=root,
            )
            with mock.patch(
                "writing_agent.task_graph_store.os.replace", side_effect=fail_replace_once
            ):
                with self.assertRaises(OSError):
                    fixture.store.publish(**arguments)
            resumed = reopen(fixture.store)
            commit = resumed.publish(**arguments)
            self.assertEqual(resumed.read_head("main"), commit)

        # Replace succeeds and refs fsync fails. An equal visible head must not
        # return success until the retry repeats the refs-directory barrier.
        directory = self.root / "post-replace-fsync"
        fixture = StoreFixture(directory)
        root, before = fixture.root()
        event, after, effect = fixture.event_state(
            before,
            lineage="main",
            set_values={"position": {**before.position, "lineage_id": "main"}},
        )
        arguments = dict(
            lineage_id="main",
            expected_head=None,
            events=(event,),
            next_state=after,
            parent_checkpoint=root,
        )
        original_barrier = fixture.store._fsync_directory
        tripped = False

        def fail_refs_once(path):
            nonlocal tripped
            if not tripped and path.name == "refs":
                tripped = True
                raise OSError("injected refs fsync failure")
            return original_barrier(path)

        with mock.patch.object(fixture.store, "_fsync_directory", side_effect=fail_refs_once):
            with self.assertRaises(OSError):
                fixture.store.publish(**arguments)
        resumed = reopen(fixture.store)
        with mock.patch.object(
            resumed, "_fsync_directory", wraps=resumed._fsync_directory
        ) as barrier:
            commit = resumed.publish(**arguments)
            self.assertTrue(any(call.args[0].name == "refs" for call in barrier.call_args_list))
        self.assertEqual(resumed.read_head("main"), commit)

    def test_lineage_authority_is_bound_to_target_checkpoint(self):
        parent, before = self.fixture.root()
        event, after, effect = self.fixture.event_state(
            before,
            lineage="child",
            set_values={
                "position": {
                    **before.position,
                    "lineage_id": "child",
                    "start_checkpoint": parent,
                }
            },
            history_set={"branch_base": before.history["head"]},
            kind="rollout_started",
        )
        commit = self.store.publish("child", None, (event,), after, parent_checkpoint=parent)
        shutil.copyfile(
            self.store.root / "refs" / "child.json",
            self.store.root / "refs" / "imposter.json",
        )
        with self.assertRaises(CorruptRecordError):
            self.store.read_head("imposter")
        imposter_event, imposter_state, _ = self.fixture.event_state(
            after,
            lineage="child",
        )
        imposter_event = replace(imposter_event, lineage_id="imposter", id=None)
        imposter_state = replace(
            imposter_state,
            history={**imposter_state.history, "head": imposter_event.identity()},
        )
        with self.assertRaises(ProjectionError):
            self.store.publish(
                "child",
                commit,
                (imposter_event,),
                imposter_state,
            )

        child_checkpoint = self.store.load_commit(commit).checkpoint
        foreign_event, foreign_state, _ = self.fixture.event_state(
            after,
            lineage="foreign",
            set_values={
                "position": {
                    **after.position,
                    "lineage_id": "foreign",
                }
            },
        )
        self.store.persist(foreign_event)
        foreign_checkpoint = self.store.save_checkpoint(
            foreign_state,
            parent=child_checkpoint,
        )
        foreign_commit = CommitV1(
            parent_commit=commit,
            events=(foreign_event.identity(),),
            checkpoint=foreign_checkpoint,
        )
        self.store.persist(foreign_commit)
        with self.assertRaises(CorruptRecordError):
            self.store.load_commit(foreign_commit.identity())

    def test_store_root_ancestry_and_durability(self):
        durable_root = self.root / "durable-store"
        with mock.patch.object(
            TaskGraphStore,
            "_fsync_directory",
            wraps=TaskGraphStore._fsync_directory,
        ) as barrier:
            TaskGraphStore(durable_root, verifier=PatchVerifier())
        synced = {call.args[0] for call in barrier.call_args_list}
        self.assertIn(durable_root.parent, synced)
        self.assertIn(durable_root, synced)

        real = self.root / "canonical-parent"
        real.mkdir()
        alias = self.root / "store-alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(MaterializationError):
            TaskGraphStore(alias / "store", verifier=PatchVerifier())
        self.assertFalse((real / "store").exists())

        with self.assertRaises(MaterializationError):
            TaskGraphStore(self.root / "missing" / "parent" / "store", verifier=PatchVerifier())
        self.assertFalse((self.root / "missing").exists())

        public_parent = self.root / "public-parent"
        public_parent.mkdir(mode=0o700)
        public_parent.chmod(0o755)
        with self.assertRaises(MaterializationError):
            TaskGraphStore(public_parent / "store", verifier=PatchVerifier())
        self.assertFalse((public_parent / "store").exists())

        with self.assertRaises(MaterializationError):
            TaskGraphStore(
                self.root / "component" / ".." / "lexical-store", verifier=PatchVerifier()
            )
        self.assertFalse((self.root / "lexical-store").exists())

        retry_parent = self.root / "retry-parent"
        retry_parent.mkdir(mode=0o700)
        retry_root = retry_parent / "store"
        real_barrier = TaskGraphStore._fsync_directory
        failed = False

        def fail_parent_once(path):
            nonlocal failed
            if not failed and path == retry_parent:
                failed = True
                raise OSError("injected root-name barrier failure")
            return real_barrier(path)

        with mock.patch.object(TaskGraphStore, "_fsync_directory", side_effect=fail_parent_once):
            with self.assertRaises(OSError):
                TaskGraphStore(retry_root, verifier=PatchVerifier())
        self.assertTrue(retry_root.is_dir())
        with mock.patch.object(TaskGraphStore, "_fsync_directory", wraps=real_barrier) as barrier:
            TaskGraphStore(retry_root, verifier=PatchVerifier())
        self.assertIn(retry_parent, (call.args[0] for call in barrier.call_args_list))


if __name__ == "__main__":
    unittest.main()
