"""Context-operation and member-start derive contracts."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

from tests.task_graph_fixtures import make_entry_fixture
from writing_agent.task_graph import (
    CheckpointV1,
    MessageV1,
    canonical_bytes,
    domain_hash,
)
from writing_agent.task_graph_compaction import (
    CompactionError,
    ContextPolicyV1,
    completed_exchanges,
)
from writing_agent.task_graph_derive_context import derive_context_operation, derive_member_start
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import (
    ContextContentV1,
    ContextOperationInputV1,
    ContextRevisionV1,
    MemberStartV1,
)
from writing_agent.task_graph_transition import (
    CheckpointChain,
    ContextView,
    LineageMode,
    LineageView,
    ToolSpec,
)


def _context(messages, tools, rendering, content_ref, revision_ref):
    return ContextView(
        messages=messages,
        sources=(None,) * len(messages),
        tools=tools,
        rendering=rendering,
        content_ref=content_ref,
        revision_ref=revision_ref,
    )


def _view(fixture) -> LineageView:
    state = fixture.state
    materialized = fixture.reader.context(state.context_ref)
    context = _context(
        materialized.messages,
        materialized.tools,
        materialized.rendering,
        ContextRevisionV1.from_dict(
            fixture.reader.artifact(state.context_ref, domain="context_revision")
        ).content_ref,
        state.context_ref,
    )
    checkpoint_id = CheckpointV1(state=state, event_head=None).identity()
    ancestry = CheckpointChain(checkpoint_id, context)
    node = fixture.graph.node(state.position["node_id"])
    return LineageView(
        root_checkpoint_id=checkpoint_id,
        checkpoint_id=checkpoint_id,
        head_event_id=None,
        state=state,
        budget=fixture.reader.artifact(state.budgets_ref),
        outcome=None,
        check_statuses={},
        context=context,
        raw_call_ids=frozenset(),
        call_sources={},
        samples=(),
        ancestry=ancestry,
        node=node,
        mode=LineageMode.for_node(node),
        tool_spec=ToolSpec(128_000, 4096),
    )


def _append_messages(view: LineageView, messages: tuple[MessageV1, ...]) -> LineageView:
    context_messages = (*view.context.messages, *messages)
    content = ContextContentV1(
        parent_ref=None,
        messages=context_messages,
        tools=view.context.tools,
        rendering=view.context.rendering,
    )
    revision = ContextRevisionV1(
        content_ref=content.identity(),
        event_head="c" * 64,
        provenance_refs=("c" * 64,),
    )
    state = replace(
        view.state,
        context_ref=revision.identity(),
        history={**view.state.history, "head": "c" * 64, "seq": 2},
    )
    context = _context(
        context_messages,
        view.context.tools,
        view.context.rendering,
        content.identity(),
        revision.identity(),
    )
    checkpoint_id = CheckpointV1(
        parents=(view.checkpoint_id,), state=state, event_head="c" * 64
    ).identity()
    return replace(
        view,
        checkpoint_id=checkpoint_id,
        head_event_id="c" * 64,
        state=state,
        context=context,
        ancestry=CheckpointChain(
            checkpoint_id,
            context,
            CheckpointChain(view.checkpoint_id, view.context, view.ancestry),
        ),
    )


def _assert_same_transition(test: unittest.TestCase, left, right) -> None:
    def record_bytes(value):
        if isinstance(value, bytes):
            return value
        if hasattr(value, "to_wire"):
            return canonical_bytes(value.to_wire())
        if hasattr(value, "to_dict"):
            return canonical_bytes(value.to_dict())
        return canonical_bytes(value)

    def body(transition):
        return (
            canonical_bytes(transition.event.to_dict()),
            canonical_bytes(transition.state.to_dict()),
            tuple((item.ref, record_bytes(item.value), item.kind) for item in transition.artifacts),
        )

    test.assertEqual(body(left), body(right))


class ContextDeriveTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_entry_fixture()
        self.view = _append_messages(
            _view(self.fixture),
            (
                MessageV1(role="assistant", content=("old draft",), origin="action:1"),
                MessageV1(role="user", content=("Continue",), origin="feedback:1"),
                MessageV1(role="assistant", content=("new draft",), origin="action:2"),
            ),
        )
        self.reader = self.fixture.reader

    def _operation(self, policy: ContextPolicyV1) -> ContextOperationInputV1:
        return ContextOperationInputV1(policy_ref=self.reader.add(policy.to_wire()))

    def test_unmatched_tool_observation_is_not_a_complete_exchange(self):
        unmatched = MessageV1(
            role="tool",
            origin="action:missing",
            call_id="missing-call",
            content=({"type": "tool_result", "call_id": "missing-call", "content": "x"},),
        )

        with self.assertRaisesRegex(CompactionError, "unmatched observation"):
            completed_exchanges((*self.view.context.messages, unmatched))

    def test_carry_seed_drop_compact_and_wire_fixed_points(self):
        cases = (
            ("carry", ContextPolicyV1("carry")),
            (
                "seed",
                ContextPolicyV1(
                    "seed",
                    seed_name="entry",
                    seed_checkpoint_ref=self.view.ancestry.parent.checkpoint_id,
                ),
            ),
            ("drop", ContextPolicyV1("drop")),
            (
                "compact",
                ContextPolicyV1(
                    "compact",
                    retained_exchanges=1,
                    summarizer_version="visible-text-v1",
                    max_summary_chars=64,
                ),
            ),
        )
        for name, policy in cases:
            with self.subTest(operation=name):
                operation = self._operation(policy)
                transition = derive_context_operation(self.view, operation, self.reader)
                roundtrip = ContextOperationInputV1.from_dict(operation.to_wire())
                _assert_same_transition(
                    self,
                    transition,
                    derive_context_operation(self.view, roundtrip, self.reader),
                )
                self.assertEqual(transition.event.kind, "context_changed")
                self.assertEqual(transition.view.context.revision_ref, transition.state.context_ref)
                if name == "carry":
                    self.assertEqual(
                        transition.view.context.content_ref, self.view.context.content_ref
                    )
                else:
                    self.assertNotEqual(
                        transition.view.context.content_ref, self.view.context.content_ref
                    )
                if name == "drop":
                    self.assertEqual(len(transition.view.context.messages), 2)
                if name == "compact":
                    self.assertIn(
                        "assistant: old draft",
                        transition.view.context.messages[2].content[0]["text"],
                    )
                    self.assertEqual(
                        {artifact.kind for artifact in transition.artifacts},
                        {"artifact", "context_node", "context_revision"},
                    )
                    self.assertNotIn(
                        "ContextOperationV1",
                        {
                            getattr(artifact.value, "record_type", None)
                            for artifact in transition.artifacts
                        },
                    )

    def test_seed_uses_ancestor_context_without_reprojection(self):
        seed_messages = (
            *self.view.context.messages[:2],
            MessageV1(role="assistant", content=("ancestor draft",), origin="action:1"),
        )
        seed_content = ContextContentV1(
            parent_ref=None,
            messages=seed_messages,
            tools=self.view.context.tools,
            rendering=self.view.context.rendering,
        )
        seed_checkpoint = "a" * 64
        seed_context = _context(
            seed_messages,
            self.view.context.tools,
            self.view.context.rendering,
            seed_content.identity(),
            "b" * 64,
        )
        current = replace(
            self.view,
            ancestry=CheckpointChain(
                self.view.checkpoint_id,
                self.view.context,
                CheckpointChain(seed_checkpoint, seed_context, self.view.ancestry.parent),
            ),
        )
        reader = _NoProjectionReader(self.reader)
        operation = self._operation(
            ContextPolicyV1(
                "seed", seed_name="approved-prefix", seed_checkpoint_ref=seed_checkpoint
            )
        )
        transition = derive_context_operation(current, operation, reader)
        self.assertEqual(reader.projection_reads, [])
        self.assertIn("ancestor draft", str(transition.view.context.messages))
        self.assertNotIn("new draft", str(transition.view.context.messages))
        self.assertTrue(
            all(not message.loss_eligible for message in transition.view.context.messages[2:])
        )

    def test_group_policy_must_match_the_sealed_spec(self):
        sealed_ref = "d" * 64
        grouped = replace(self.view, group=_GroupBinding({"context_policy_ref": sealed_ref}))
        operation = self._operation(ContextPolicyV1("drop"))
        with self.assertRaises(ProjectionError):
            derive_context_operation(grouped, operation, self.reader)


@dataclass(frozen=True)
class _GroupBinding:
    policy: dict[str, str]


class _NoProjectionReader:
    def __init__(self, reader):
        self.reader = reader
        self.projection_reads = []

    def __getattr__(self, name):
        if name in {"checkpoint", "context"}:
            self.projection_reads.append(name)
            raise AssertionError("seed derivation must use view.ancestry")
        return getattr(self.reader, name)


class MemberStartDeriveTests(unittest.TestCase):
    def setUp(self):
        from tests.test_task_graph_group import GroupCoordinatorTests

        self.group_fixture = GroupCoordinatorTests("test_full_contract_drift_and_start_isolation")
        self.group_fixture.setUp()
        self.addCleanup(self.group_fixture.doCleanups)
        self.spec = self.group_fixture.group()
        self.reader = _StoreReader(self.group_fixture.store)
        self.reader.public[self.spec.identity()] = self.spec.to_wire()
        self.view = _group_entry_view(self.group_fixture)

    def test_member_start_seed_payload_matches_coordinator_and_roundtrips(self):
        spec_ref = self.spec.identity()
        operation = MemberStartV1(group_spec_ref=spec_ref, ordinal=0)
        transition = derive_member_start(self.view, operation, self.reader)
        roundtrip = MemberStartV1.from_dict(operation.to_wire())
        _assert_same_transition(
            self,
            transition,
            derive_member_start(self.view, roundtrip, self.reader),
        )
        runtime = self.group_fixture.coordinator.start(
            self.spec, 0, policy=self.group_fixture.policy
        )
        seeds_artifact = next(
            item for item in transition.artifacts if item.ref == transition.state.rng_ref
        )
        self.assertEqual(seeds_artifact.ref, runtime.state.rng_ref)
        self.assertEqual(
            seeds_artifact.value,
            canonical_bytes(self.group_fixture.store.get_artifact(runtime.state.rng_ref)),
        )
        self.assertEqual(transition.state.history["branch_base"], self.view.head_event_id)
        self.assertEqual(transition.view.group.identity(), spec_ref)

    def test_changed_seed_ordinal_and_nonentry_view_are_rejected(self):
        spec_ref = self.spec.identity()
        body = self.spec.to_wire()
        members = list(body["members"])
        members[0] = {**members[0], "writer_seed": members[0]["writer_seed"] + 1}
        body["members"] = members
        forged_ref = domain_hash("payload", body)
        self.reader.public[forged_ref] = body
        with self.assertRaises(ProjectionError):
            derive_member_start(
                self.view,
                MemberStartV1(group_spec_ref=forged_ref, ordinal=0),
                self.reader,
            )
        with self.assertRaises(ProjectionError):
            derive_member_start(
                self.view,
                MemberStartV1(group_spec_ref=spec_ref, ordinal=len(self.spec.members)),
                self.reader,
            )
        started = derive_member_start(
            self.view, MemberStartV1(group_spec_ref=spec_ref, ordinal=0), self.reader
        )
        with self.assertRaises(ProjectionError):
            derive_member_start(
                started.view, MemberStartV1(group_spec_ref=spec_ref, ordinal=1), self.reader
            )
        unsampled = replace(
            self.view,
            state=replace(
                self.view.state,
                history={**self.view.state.history, "seq": 1, "head": "e" * 64},
            ),
        )
        with self.assertRaises(ProjectionError):
            derive_member_start(
                unsampled, MemberStartV1(group_spec_ref=spec_ref, ordinal=0), self.reader
            )


class _StoreReader:
    def __init__(self, store):
        self.store = store
        self.public = {}

    def artifact(self, ref, *, domain="payload", private=False):
        if domain == "payload" and ref in self.public and not private:
            return self.public[ref]
        return self.store.get_artifact(ref, expected_domain=domain, private=private)


def _group_entry_view(fixture) -> LineageView:
    return fixture.fixture.env.verify(fixture.runtime)
