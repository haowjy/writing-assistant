import json
import os
import shutil
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    ContextRevisionV1,
    EnvironmentStateV1,
    EventV1,
    GraphInstanceV1,
    MessageV1,
    NodeSpecV1,
    canonical_json,
    tree_hash,
)
from writing_agent.task_graph_store import (
    ConcurrentUpdateError,
    CorruptRecordError,
    MaterializationError,
    MissingReferenceError,
    ReplayError,
    TaskGraphStore,
    WrongRecordDomainError,
)


class StoreFixture:
    def __init__(self, root: Path, **limits):
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.store = TaskGraphStore(root / "store", **limits)
        self.common = {}
        for name in (
            "entry",
            "template",
            "tokenizer",
            "tools",
            "requirements",
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
        self.private = self.store.put_artifact({"fixture": "author-packet"}, private=True)
        self.instance = GraphInstanceV1(
            template_ref=self.common["template"],
            entry_node="write",
            nodes=(NodeSpecV1(id="write", entry_contract=self.common["entry"]),),
        )
        self.store.persist(self.instance)
        self.context = self.make_context("Please revise the draft.")

    def make_context(self, text: str, *, event_head=None, provenance=()):
        context = ContextRevisionV1(
            messages=(MessageV1(content=(text,), origin="request:1"),),
            event_head=event_head,
            provenance_refs=provenance,
            rendering={
                "projection_version": "v1",
                "prefix_id": "root",
                "template_ref": self.common["template"],
                "tokenizer_ref": self.common["tokenizer"],
                "tool_schema_ref": self.common["tools"],
            },
        )
        self.store.persist(context)
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

    def effect_event_state(
        self,
        before,
        *,
        lineage,
        file_delta=None,
        set_values=None,
        history_set=None,
        kind="tool_result",
    ):
        set_values = json.loads(canonical_json(set_values or {}))
        history_set = json.loads(canonical_json(history_set or {}))
        effect = {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": before.identity(),
            "file_delta": file_delta or {},
            "set": set_values,
            "history_set": history_set,
        }
        effect_ref = self.store.put_artifact(effect)
        event = EventV1(
            previous=before.history["head"],
            seq=before.history["seq"] + 1,
            lineage_id=lineage,
            kind=kind,
            audience=("controller",),
            payload_ref=effect_ref,
            versions_ref=self.common["versions"],
            provenance_ref=self.common["provenance"],
        )
        value = before.to_dict()
        value.update(set_values)
        files = dict(before.files)
        for path, delta in (file_delta or {}).items():
            if delta["after"] is None:
                files.pop(path, None)
            else:
                files[path] = delta["after"]
        value["files"] = files
        value["tree_hash"] = tree_hash(files)
        history = before.to_dict()["history"]
        history.update(history_set)
        history.update(head=event.identity(), seq=event.seq)
        value["history"] = history
        return event, EnvironmentStateV1.from_dict(value), effect_ref


class TaskGraphStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = StoreFixture(self.root)
        self.store = self.fixture.store

    def tearDown(self):
        self.temporary.cleanup()

    def publish_change(self, lineage="main", expected=None, parent=None, before=None, text="beta"):
        if before is None:
            parent, before = self.fixture.root()
        event, after, effect = self.fixture.effect_event_state(
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
            artifact_refs=(effect,),
        )
        return commit, self.store.load_commit(commit).checkpoint, after, event, effect

    def test_round_trip_restore_preserves_complete_state_context_and_files(self):
        files = {"empty.txt": "", "unicode/雪.txt": "café\n", "draft.txt": "alpha"}
        root, state = self.fixture.root(files=files)
        runtime = self.store.restore(root, self.root / "worker")

        self.assertEqual(runtime.state, state)
        self.assertEqual(runtime.context, self.fixture.context)
        self.assertEqual(runtime.checkpoint_id, root)
        self.assertEqual((runtime.workspace / "empty.txt").read_bytes(), b"")
        self.assertEqual((runtime.workspace / "unicode" / "雪.txt").read_text(), "café\n")
        self.assertEqual(
            sorted(
                path.relative_to(runtime.workspace).as_posix()
                for path in runtime.workspace.rglob("*")
            ),
            ["draft.txt", "empty.txt", "unicode", "unicode/雪.txt"],
        )
        self.assertEqual(stat.S_IMODE(runtime.workspace.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((runtime.workspace / "draft.txt").stat().st_mode), 0o600)

    def test_deletion_diff_and_non_file_diff(self):
        root, before = self.fixture.root(files={"gone.txt": "x", "same": "", "z": "old"})
        event, after, effect = self.fixture.effect_event_state(
            before,
            lineage="main",
            file_delta={
                "gone.txt": {"before": "x", "after": None},
                "new/é.txt": {"before": None, "after": "雪"},
                "z": {"before": "old", "after": "new"},
            },
            set_values={"position": {**before.position, "lineage_id": "main", "phase": "checking"}},
        )
        commit = self.store.publish(
            "main", None, (event,), after, parent_checkpoint=root, artifact_refs=(effect,)
        )
        target = self.store.load_commit(commit).checkpoint
        difference = self.store.diff(root, target, text=True)
        self.assertEqual(
            [(item.path, item.status) for item in difference.files],
            [("gone.txt", "deleted"), ("new/é.txt", "added"), ("z", "changed")],
        )
        self.assertIn("-old", difference.files[-1].text_diff)
        self.assertEqual([item.field for item in difference.state], ["history", "position"])
        restored = self.store.restore(target, self.root / "after")
        self.assertFalse((restored.workspace / "gone.txt").exists())
        self.assertEqual((restored.workspace / "new" / "é.txt").read_text(), "雪")

    def test_workspace_limits_freshness_symlinks_and_cleanup(self):
        small = StoreFixture(self.root / "small", max_workspace_bytes=4, max_file_bytes=3)
        too_large, _ = small.root(files={"a": "éé"})
        with self.assertRaises(MaterializationError):
            small.store.materialize(too_large, self.root / "too-large")
        self.assertFalse((self.root / "too-large").exists())

        checkpoint, _ = self.fixture.root(files={"a/b": "x"})
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            self.store.materialize(checkpoint, existing)
        link = self.root / "link"
        link.symlink_to(self.root / "elsewhere")
        with self.assertRaises(MaterializationError):
            self.store.materialize(checkpoint, link)

        with self.assertRaises(MaterializationError):
            self.store.materialize(checkpoint, self.store.root / "worker")
        self.assertFalse((self.store.root / "worker").exists())
        with self.assertRaises(MaterializationError):
            self.store.materialize(checkpoint, self.root)

        real = self.root / "real"
        real.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(MaterializationError):
            self.store.materialize(checkpoint, alias / "workspace")
        self.assertFalse((real / "workspace").exists())

        destination = self.root / "faulted"

        def fail(stage):
            if stage == "after_file:a/b":
                raise RuntimeError("induced")

        with self.assertRaises(MaterializationError):
            self.store.materialize(checkpoint, destination, fault=fail)
        self.assertFalse(destination.exists())

    def test_missing_private_artifact_corrupt_bytes_and_wrong_domain_fail_closed(self):
        checkpoint, state = self.fixture.root()
        (self.store.root / "private" / state.author_packet_ref).unlink()
        with self.assertRaises(MissingReferenceError):
            self.store.restore(checkpoint, self.root / "missing-private")
        self.assertFalse((self.root / "missing-private").exists())

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

        context_path = self.store.root / "contexts" / f"{self.fixture.context.identity()}.json"
        context_path.write_bytes(b"not-json")
        with self.assertRaises(CorruptRecordError):
            self.store.load_context(self.fixture.context.identity())

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
        event, after, effect = self.fixture.effect_event_state(
            before,
            lineage="line",
            file_delta={"draft.txt": {"before": "alpha", "after": "beta"}},
            set_values={"position": {**before.position, "lineage_id": "line"}},
        )
        self.store.persist(event)
        checkpoint = self.store.save_checkpoint(after, parent=root, artifact_refs=(effect,))
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
            self.store.save_checkpoint(invalid_state, parent=checkpoint, artifact_refs=(effect,))

    def test_authority_is_exact_projection_and_cas_is_idempotent(self):
        root, before = self.fixture.root()
        event, after, effect = self.fixture.effect_event_state(
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
            artifact_refs=(effect,),
        )
        commit = self.store.publish(**arguments)
        self.assertEqual(self.store.publish(**arguments), commit)
        authority = json.loads((self.store.root / "refs" / "main.json").read_text())
        self.assertEqual(authority, {"head_commit": commit})
        self.assertNotIn("expected_head", authority)

        competing_event, competing_state, competing_effect = self.fixture.effect_event_state(
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
                artifact_refs=(competing_effect,),
            )

    def test_competing_updates_allow_exactly_one_winner(self):
        first, first_checkpoint, state, _, _ = self.publish_change()
        barrier = threading.Barrier(2)
        results = []

        def contender(label):
            event, after, effect = self.fixture.effect_event_state(
                state,
                lineage="main",
                file_delta={"draft.txt": {"before": "beta", "after": label}},
            )
            barrier.wait()
            try:
                result = self.store.publish("main", first, (event,), after, artifact_refs=(effect,))
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
                event, after, effect = fixture.effect_event_state(
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
                        artifact_refs=(effect,),
                        fault=fail,
                    )
                head = fixture.store.read_head("main")
                self.assertEqual(head is not None, published)
                authoritative = root if head is None else fixture.store.load_commit(head).checkpoint
                expected = before if head is None else after
                self.assertEqual(fixture.store.load_checkpoint(authoritative).state, expected)

    def test_branch_isolated_full_state_and_parent_immutable(self):
        parent, parent_state = self.fixture.root(files={"draft.txt": "alpha", "empty": ""})
        parent_path = self.store.root / "checkpoints" / f"{parent}.json"
        original = parent_path.read_bytes()
        event, branch_state, effect = self.fixture.effect_event_state(
            parent_state,
            lineage="branch-a",
            set_values={
                "position": {
                    **parent_state.position,
                    "lineage_id": "branch-a",
                    "start_checkpoint": parent,
                }
            },
            history_set={"branch_base": parent_state.history["head"]},
            kind="rollout_started",
        )
        commit = self.store.branch(
            parent, "branch-a", (event,), branch_state, artifact_refs=(effect,)
        )
        child = self.store.load_commit(commit).checkpoint
        runtime = self.store.restore(child, self.root / "branch-worker")
        (runtime.workspace / "draft.txt").write_text("local mutation")
        self.assertEqual(parent_path.read_bytes(), original)
        self.assertEqual(self.store.load_checkpoint(parent).state, parent_state)
        self.assertEqual(self.store.load_checkpoint(child).parents, (parent,))
        self.assertEqual(runtime.state.position["start_checkpoint"], parent)
        self.assertEqual(self.store.replay("branch-a", parent, (commit,)), child)

    def test_recorded_replay_is_read_only_idempotent_and_verifies_post_state(self):
        root, before = self.fixture.root()
        first, state1, effect1 = self.fixture.effect_event_state(
            before,
            lineage="main",
            file_delta={"draft.txt": {"before": "alpha", "after": "beta"}},
            set_values={"position": {**before.position, "lineage_id": "main"}},
        )
        commit1 = self.store.publish(
            "main", None, (first,), state1, parent_checkpoint=root, artifact_refs=(effect1,)
        )
        checkpoint1 = self.store.load_commit(commit1).checkpoint
        second, state2, effect2 = self.fixture.effect_event_state(
            state1,
            lineage="main",
            file_delta={"draft.txt": {"before": "beta", "after": "gamma"}},
            history_set={"tool_result_ids": ("main:result:1",)},
        )
        commit2 = self.store.publish("main", commit1, (second,), state2, artifact_refs=(effect2,))
        checkpoint2 = self.store.load_commit(commit2).checkpoint
        authority_before = (self.store.root / "refs" / "main.json").read_bytes()
        self.assertEqual(self.store.replay("main", root, (commit1, commit2)), checkpoint2)
        self.assertEqual(self.store.replay("main", root, (commit1, commit2)), checkpoint2)
        self.assertEqual((self.store.root / "refs" / "main.json").read_bytes(), authority_before)
        self.assertEqual(self.store.replay("main", checkpoint1, (commit2,)), checkpoint2)

        bad_effect = {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": state2.identity(),
            "file_delta": {"draft.txt": {"before": "WRONG", "after": "delta"}},
            "set": {},
            "history_set": {},
        }
        bad_ref = self.store.put_artifact(bad_effect)
        bad_event = EventV1(
            previous=state2.history["head"],
            seq=state2.history["seq"] + 1,
            lineage_id="main",
            kind="tool_result",
            audience=("controller",),
            payload_ref=bad_ref,
            versions_ref=self.fixture.common["versions"],
            provenance_ref=self.fixture.common["provenance"],
        )
        bad_body = state2.to_dict()
        bad_body["files"] = {"draft.txt": "delta"}
        bad_body["tree_hash"] = tree_hash({"draft.txt": "delta"})
        bad_body["history"].update(head=bad_event.identity(), seq=bad_event.seq)
        bad_state = EnvironmentStateV1.from_dict(bad_body)
        with self.assertRaises(ReplayError):
            self.store.publish("main", commit2, (bad_event,), bad_state, artifact_refs=(bad_ref,))
        self.assertEqual(self.store.read_head("main"), commit2)

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

    def test_recorded_effect_reference_edges_are_typed_and_transient(self):
        root, state = self.fixture.root()
        missing = "0" * 64
        position = state.to_dict()["position"]
        continuation = state.to_dict()["continuation"]
        cases = {
            "instance_ref": ({"instance_ref": missing}, {}),
            "position.entry_contract": (
                {"position": {**position, "entry_contract": missing}},
                {},
            ),
            "position.start_checkpoint": (
                {"position": {**position, "start_checkpoint": missing}},
                {},
            ),
            "context_ref": ({"context_ref": missing}, {}),
            "requirements_ref": ({"requirements_ref": missing}, {}),
            "decisions_ref": ({"decisions_ref": missing}, {}),
            "disclosures_ref": ({"disclosures_ref": missing}, {}),
            "author_packet_ref": ({"author_packet_ref": missing}, {}),
            "versions_ref": ({"versions_ref": missing}, {}),
            "budgets_ref": ({"budgets_ref": missing}, {}),
            "rng_ref": ({"rng_ref": missing}, {}),
            "external_inputs_ref": ({"external_inputs_ref": missing}, {}),
            "outcome_ref": ({"outcome_ref": missing}, {}),
            "provenance_ref": ({"provenance_ref": missing}, {}),
            "continuation.author_request": (
                {"continuation": {**continuation, "author_request": missing}},
                {},
            ),
            "continuation.check_requests": (
                {"continuation": {**continuation, "check_requests": (missing,)}},
                {},
            ),
            "history.branch_base": ({}, {"branch_base": missing}),
            "history.imported_refs": ({}, {"imported_refs": (missing,)}),
        }
        for label, (set_values, history_set) in cases.items():
            with self.subTest(edge=label):
                effect = self.store.put_artifact(
                    {
                        "artifact_type": "Phase2RecordedEffectV1",
                        "before_state_ref": state.identity(),
                        "file_delta": {},
                        "set": set_values,
                        "history_set": history_set,
                    }
                )
                with self.assertRaises(MissingReferenceError):
                    self.store.save_checkpoint(state, parent=root, artifact_refs=(effect,))

        wrong_domain_cases = {
            "instance": ({"instance_ref": state.budgets_ref}, {}),
            "context": ({"context_ref": state.budgets_ref}, {}),
            "private": ({"author_packet_ref": state.budgets_ref}, {}),
            "checkpoint": (
                {"position": {**position, "start_checkpoint": state.budgets_ref}},
                {},
            ),
            "event": ({}, {"branch_base": state.budgets_ref}),
        }
        for label, (set_values, history_set) in wrong_domain_cases.items():
            with self.subTest(wrong_domain=label):
                effect = self.store.put_artifact(
                    {
                        "artifact_type": "Phase2RecordedEffectV1",
                        "before_state_ref": state.identity(),
                        "file_delta": {},
                        "set": set_values,
                        "history_set": history_set,
                    }
                )
                with self.assertRaises(WrongRecordDomainError):
                    self.store.save_checkpoint(state, parent=root, artifact_refs=(effect,))

        private_effect = self.store.put_artifact(
            {
                "artifact_type": "Phase2RecordedEffectV1",
                "before_state_ref": state.identity(),
                "file_delta": {"private-probe": {"before": None, "after": "x"}},
                "set": {"budgets_ref": missing},
                "history_set": {},
            },
            private=True,
        )
        private_body = state.to_dict()
        private_body["continuation"]["check_requests"] = [private_effect]
        with self.assertRaises(MissingReferenceError):
            self.store.save_checkpoint(EnvironmentStateV1.from_dict(private_body))

        # before_state_ref is a replay assertion over an EnvironmentStateV1
        # identity, not a reference to a separately persisted state object.
        assertion_only = self.store.put_artifact(
            {
                "artifact_type": "Phase2RecordedEffectV1",
                "before_state_ref": missing,
                "file_delta": {},
                "set": {},
                "history_set": {},
            }
        )
        self.store.save_checkpoint(state, parent=root, artifact_refs=(assertion_only,))

        first, transient, first_effect = self.fixture.effect_event_state(
            state,
            lineage="main",
            set_values={
                "position": {**state.position, "lineage_id": "main"},
                "budgets_ref": missing,
            },
        )
        second, final, second_effect = self.fixture.effect_event_state(
            transient,
            lineage="main",
            set_values={"budgets_ref": state.budgets_ref},
        )
        with self.assertRaises(MissingReferenceError):
            self.store.publish(
                "main",
                None,
                (first, second),
                final,
                parent_checkpoint=root,
                artifact_refs=(first_effect, second_effect),
            )
        self.assertIsNone(self.store.read_head("main"))

    def test_artifact_schemas_and_locations_fail_closed(self):
        _, state = self.fixture.root()
        malformed = self.store.put_artifact(
            {
                "artifact_type": "Phase2RecordedEffectV1",
                "before_state_ref": state.identity(),
                "file_delta": {},
                "set": {},
            }
        )
        with self.assertRaises(CorruptRecordError):
            self.store.save_checkpoint(state, artifact_refs=(malformed,))

        unsupported = self.store.put_artifact(
            {"artifact_type": "FutureTransitionV9", "some_ref": "0" * 64}
        )
        with self.assertRaises(WrongRecordDomainError):
            self.store.save_checkpoint(state, artifact_refs=(unsupported,))

        duplicate = self.store.put_artifact({"duplicate": True})
        self.store.put_artifact({"duplicate": True}, private=True)
        with self.assertRaises(WrongRecordDomainError):
            self.store.save_checkpoint(state, artifact_refs=(duplicate,))

        event_envelope = self.store.put_artifact(
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
        with self.assertRaises(WrongRecordDomainError):
            self.store.save_checkpoint(state, artifact_refs=(event_envelope,))

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

    def test_publication_reduces_actual_parent_and_rejects_unsupported_effects(self):
        root, before = self.fixture.root()
        event, after, effect = self.fixture.effect_event_state(
            before,
            lineage="main",
            set_values={"position": {**before.position, "lineage_id": "main"}},
        )
        body = after.to_dict()
        body["budgets_ref"] = self.store.put_artifact({"fixture": "other-budget"})
        contradictory = EnvironmentStateV1.from_dict(body)
        with self.assertRaises(ReplayError):
            self.store.publish(
                "main",
                None,
                (event,),
                contradictory,
                parent_checkpoint=root,
                artifact_refs=(effect,),
            )
        self.assertIsNone(self.store.read_head("main"))

        payload = self.store.put_artifact({"transition": "future"})
        unsupported = EventV1(
            lineage_id="main",
            kind="tool_result",
            audience=("controller",),
            payload_ref=payload,
            versions_ref=self.fixture.common["versions"],
            provenance_ref=self.fixture.common["provenance"],
        )
        unsupported_body = before.to_dict()
        unsupported_body["position"]["lineage_id"] = "main"
        unsupported_body["history"].update(head=unsupported.identity(), seq=1)
        unsupported_state = EnvironmentStateV1.from_dict(unsupported_body)
        with self.assertRaises(ReplayError):
            self.store.publish(
                "main",
                None,
                (unsupported,),
                unsupported_state,
                parent_checkpoint=root,
                artifact_refs=(payload,),
            )

    def test_durability_failures_are_repaired_by_reopen_and_retry(self):
        def reopen(store):
            return TaskGraphStore(
                store.root,
                max_workspace_bytes=store.max_workspace_bytes,
                max_file_bytes=store.max_file_bytes,
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
            event, after, effect = fixture.effect_event_state(
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
                artifact_refs=(effect,),
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
        event, after, effect = fixture.effect_event_state(
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
            artifact_refs=(effect,),
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

    def test_branch_retry_repairs_post_replace_barrier_and_conflicts_stay_distinct(self):
        for reopen in (False, True):
            with self.subTest(reopen=reopen):
                fixture = StoreFixture(self.root / f"branch-retry-{reopen}")
                parent, before = fixture.root()
                event, after, effect = fixture.effect_event_state(
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
                arguments = (parent, "child", (event,), after)
                keywords = {"artifact_refs": (effect,)}
                real_barrier = fixture.store._fsync_directory
                failed = False

                def fail_refs_once(path, _barrier=real_barrier):
                    nonlocal failed
                    if not failed and path.name == "refs":
                        failed = True
                        raise OSError("injected refs fsync failure")
                    return _barrier(path)

                with mock.patch.object(
                    fixture.store, "_fsync_directory", side_effect=fail_refs_once
                ):
                    with self.assertRaises(OSError):
                        fixture.store.branch(*arguments, **keywords)

                store = TaskGraphStore(fixture.store.root) if reopen else fixture.store
                with mock.patch.object(
                    store, "_fsync_directory", wraps=store._fsync_directory
                ) as barrier:
                    commit = store.branch(*arguments, **keywords)
                    self.assertTrue(
                        any(call.args[0].name == "refs" for call in barrier.call_args_list)
                    )
                self.assertEqual(store.read_head("child"), commit)

                other_event, other_state, other_effect = fixture.effect_event_state(
                    before,
                    lineage="child",
                    set_values={
                        "position": {
                            **before.position,
                            "lineage_id": "child",
                            "start_checkpoint": parent,
                            "phase": "checking",
                        }
                    },
                    history_set={"branch_base": before.history["head"]},
                    kind="rollout_started",
                )
                with self.assertRaises(ConcurrentUpdateError):
                    store.branch(
                        parent,
                        "child",
                        (other_event,),
                        other_state,
                        artifact_refs=(other_effect,),
                    )

    def test_lineage_authority_is_bound_to_target_checkpoint(self):
        parent, before = self.fixture.root()
        event, after, effect = self.fixture.effect_event_state(
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
        commit = self.store.branch(parent, "child", (event,), after, artifact_refs=(effect,))
        shutil.copyfile(
            self.store.root / "refs" / "child.json",
            self.store.root / "refs" / "imposter.json",
        )
        with self.assertRaises(CorruptRecordError):
            self.store.read_head("imposter")
        with self.assertRaises(CorruptRecordError):
            self.store.replay("imposter", parent, (commit,))
        imposter_event, imposter_state, imposter_effect = self.fixture.effect_event_state(
            after,
            lineage="imposter",
            set_values={
                "position": {
                    **after.position,
                    "lineage_id": "imposter",
                }
            },
        )
        with self.assertRaises(CorruptRecordError):
            self.store.publish(
                "imposter",
                commit,
                (imposter_event,),
                imposter_state,
                artifact_refs=(imposter_effect,),
            )

        child_checkpoint = self.store.load_commit(commit).checkpoint
        foreign_event, foreign_state, foreign_effect = self.fixture.effect_event_state(
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
            artifact_refs=(foreign_effect,),
        )
        foreign_commit = CommitV1(
            parent_commit=commit,
            events=(foreign_event.identity(),),
            checkpoint=foreign_checkpoint,
        )
        self.store.persist(foreign_commit)
        with self.assertRaises(CorruptRecordError):
            self.store.load_commit(foreign_commit.identity())

    def test_store_root_ancestry_and_restore_cleanup(self):
        durable_root = self.root / "durable-store"
        with mock.patch.object(
            TaskGraphStore,
            "_fsync_directory",
            wraps=TaskGraphStore._fsync_directory,
        ) as barrier:
            TaskGraphStore(durable_root)
        synced = {call.args[0] for call in barrier.call_args_list}
        self.assertIn(durable_root.parent, synced)
        self.assertIn(durable_root, synced)

        real = self.root / "canonical-parent"
        real.mkdir()
        alias = self.root / "store-alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(MaterializationError):
            TaskGraphStore(alias / "store")
        self.assertFalse((real / "store").exists())

        with self.assertRaises(MaterializationError):
            TaskGraphStore(self.root / "missing" / "parent" / "store")
        self.assertFalse((self.root / "missing").exists())

        public_parent = self.root / "public-parent"
        public_parent.mkdir(mode=0o700)
        public_parent.chmod(0o755)
        with self.assertRaises(MaterializationError):
            TaskGraphStore(public_parent / "store")
        self.assertFalse((public_parent / "store").exists())

        with self.assertRaises(MaterializationError):
            TaskGraphStore(self.root / "component" / ".." / "lexical-store")
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
                TaskGraphStore(retry_root)
        self.assertTrue(retry_root.is_dir())
        with mock.patch.object(TaskGraphStore, "_fsync_directory", wraps=real_barrier) as barrier:
            TaskGraphStore(retry_root)
        self.assertIn(retry_parent, (call.args[0] for call in barrier.call_args_list))

        checkpoint, _ = self.fixture.root()
        inside = self.store.root / "inside"
        inside.mkdir(mode=0o700)
        jump = self.root / "jump"
        jump.symlink_to(inside, target_is_directory=True)
        with self.assertRaises(MaterializationError):
            self.store.materialize(checkpoint, jump / ".." / "worker")
        self.assertFalse((self.store.root / "worker").exists())

        workspace = self.root / "restore-cleanup"
        with mock.patch.object(
            self.store,
            "load_context",
            side_effect=CorruptRecordError("induced late context corruption"),
        ):
            with self.assertRaises(CorruptRecordError):
                self.store.restore(checkpoint, workspace)
        self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()
