import json
import shutil
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from writing_agent.task_graph import (
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
        with self.assertRaises(FileExistsError):
            self.store.materialize(checkpoint, link)

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
        branch_base = self.fixture.state(
            lineage="branch-a",
            files=dict(parent_state.files),
            start=parent,
            branch_base=parent_state.history["head"],
        )
        event, branch_state, effect = self.fixture.effect_event_state(
            branch_base,
            lineage="branch-a",
            set_values={"position": dict(branch_base.position)},
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
        bad_commit = self.store.publish(
            "main", commit2, (bad_event,), bad_state, artifact_refs=(bad_ref,)
        )
        with self.assertRaises(ReplayError):
            self.store.replay("main", checkpoint2, (bad_commit,))

    def test_binary_artifact_and_missing_public_reference_validation(self):
        binary = self.store.put_bytes_artifact(b"\x00\xff", domain="payload")
        self.assertEqual(
            self.store.get_artifact(binary, expected_domain="payload:bytes"), b"\x00\xff"
        )
        checkpoint, state = self.fixture.root()
        (self.store.root / "artifacts" / state.budgets_ref).unlink()
        with self.assertRaises(MissingReferenceError):
            self.store.load_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
