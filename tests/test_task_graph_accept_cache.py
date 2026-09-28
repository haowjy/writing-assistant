"""View-cache soundness, closure ordering, eviction, and copy-on-read acceptance."""

from __future__ import annotations

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from tests.task_graph_rollout_fixtures import build_rollout_fixture, run_slice
from writing_agent.task_graph import CheckpointV1
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_errors import DriverBudgetError, ProjectionError
from writing_agent.task_graph_gate import LineageGate, derive_input


class CacheAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_warm_gate_rejects_same_event_sibling_after_a_failed_publish(self) -> None:
        fixture = build_rollout_fixture(self.root / "warm-sibling")
        runtime = run_slice(fixture, until=lambda directive: directive.kind == "execute_tool")
        honest_checkpoint = fixture.store.load_checkpoint(runtime.checkpoint_id)
        cached = fixture.gate.view(fixture.store, runtime.checkpoint_id)
        self.assertEqual(cached.head_event_id, honest_checkpoint.event_head)

        forged_budget = fixture.store.get_artifact(runtime.state.budgets_ref)
        forged_budget["consumed"]["storage_bytes"] += 1
        forged_state = replace(
            runtime.state,
            budgets_ref=fixture.store.put_artifact(forged_budget),
        )
        sibling_id = fixture.store.persist(
            CheckpointV1(
                parents=honest_checkpoint.parents,
                state=forged_state,
                event_head=honest_checkpoint.event_head,
            )
        )
        sibling = fixture.store.load_checkpoint(sibling_id)
        self.assertEqual(sibling.event_head, honest_checkpoint.event_head)
        self.assertNotEqual(sibling.state.identity(), honest_checkpoint.state.identity())

        def assert_sibling_rejected() -> None:
            with self.assertRaises(ProjectionError) as caught:
                fixture.gate.view(fixture.store, sibling_id)
            self.assertIn("state.budgets_ref", str(caught.exception))

        assert_sibling_rejected()
        head = fixture.store.read_head(fixture.lineage_id)
        original_publish = fixture.store.publish

        def fail_after_verifier(*args, **kwargs):
            def fail(stage: str) -> None:
                if stage == "after_immutable_writes":
                    raise RuntimeError("failed after candidate verification")

            kwargs["fault"] = fail
            return original_publish(*args, **kwargs)

        fixture.store.publish = fail_after_verifier
        with self.assertRaisesRegex(RuntimeError, "candidate verification"):
            fixture.driver().run(runtime, max_steps=1)
        fixture.store.publish = original_publish
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)
        assert_sibling_rejected()

    def test_unpublished_candidate_never_enters_the_view_cache(self) -> None:
        fixture = build_rollout_fixture(self.root / "candidate")
        view = fixture.env.verify(fixture.runtime)
        turn = fixture.gatherers.sampler.turn(fixture.env.port_input(view, next_step(view)))
        candidate = derive_input(view, turn, fixture.env.reader)
        candidate_id = candidate.view.checkpoint_id
        original_publish = fixture.store.publish

        def fail_after_verifier(*args, **kwargs):
            def fail(stage: str) -> None:
                if stage == "after_immutable_writes":
                    raise RuntimeError("do not publish candidate")

            kwargs["fault"] = fail
            return original_publish(*args, **kwargs)

        fixture.store.publish = fail_after_verifier
        with self.assertRaisesRegex(RuntimeError, "do not publish"):
            fixture.env.commit(fixture.runtime, turn)
        fixture.store.publish = original_publish
        self.assertIsNone(fixture.gate.cache.get(view.root_checkpoint_id, candidate_id))
        self.assertIsNone(fixture.store.read_head(fixture.lineage_id))

    def test_cache_lookup_occurs_only_after_the_store_reloads_checkpoint_ancestry(self) -> None:
        fixture = build_rollout_fixture(self.root / "closure-order")
        runtime = run_slice(fixture, until=lambda directive: directive.kind == "execute_tool")
        target_id = runtime.checkpoint_id
        chain = []
        current = fixture.store.load_checkpoint(target_id)
        identity = target_id
        while True:
            chain.append(identity)
            if not current.parents:
                break
            identity = current.parents[0]
            current = fixture.store.load_checkpoint(identity)
        trace = []
        original_load = fixture.store.load_checkpoint
        original_get = fixture.gate.cache.get

        def load_checkpoint(identity):
            checkpoint = original_load(identity)
            trace.append(("closure", identity))
            return checkpoint

        def cache_get(root_id, checkpoint_id):
            trace.append(("cache", checkpoint_id))
            return original_get(root_id, checkpoint_id)

        with patch.object(fixture.store, "load_checkpoint", side_effect=load_checkpoint):
            with patch.object(fixture.gate.cache, "get", side_effect=cache_get):
                with fixture.store.operation():
                    fixture.gate.view(fixture.store, target_id)

        first_cache = next(index for index, item in enumerate(trace) if item[0] == "cache")
        loaded_before_cache = {
            identity for kind, identity in trace[:first_cache] if kind == "closure"
        }
        self.assertTrue(set(chain) <= loaded_before_cache)

    def test_lru_evicts_a_lineage_under_load_of_sixty_five_published_views(self) -> None:
        gate = LineageGate()
        fixtures = []
        roots = []
        published = []
        for index in range(65):
            fixture = build_rollout_fixture(self.root / f"lineage-{index}", mode="none")
            fixture.gate = gate
            fixture.store._verifier = gate
            fixture.env.gate = gate
            runtime = fixture.env.enter(
                fixture.entry.node_id,
                replace(
                    fixture.entry.params,
                    lineage_id=f"cache-lineage-{index}",
                    visit_id=f"cache-visit-{index}",
                ),
            )
            fixture.runtime = runtime
            roots.append(runtime.checkpoint_id)
            fixtures.append(fixture)
            with self.assertRaises(DriverBudgetError):
                fixture.driver().run(runtime, max_steps=1)
            published.append(fixture.checkpoint_ids[-1])

        self.assertEqual(len(gate.cache._views), 64)
        self.assertIsNone(gate.cache.get(roots[0], published[0]))
        self.assertIsNotNone(gate.cache.get(roots[-1], published[-1]))

    def test_two_threads_can_verify_one_warm_head_through_the_same_gate(self) -> None:
        fixture = build_rollout_fixture(self.root / "concurrent")
        runtime = run_slice(fixture, until=lambda directive: directive.kind == "execute_tool")
        barrier = Barrier(2)

        def verify():
            barrier.wait(timeout=5)
            return fixture.env.verify(runtime)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(verify)
            second = executor.submit(verify)
            self.assertEqual(first.result(timeout=10), second.result(timeout=10))

    def test_full_slice_passes_scope_exit_audit_and_returned_mutations_do_not_escape(self) -> None:
        with patch.dict(os.environ, {"CWA_TASK_GRAPH_AUDIT_SCOPE_EXIT": "1"}):
            fixture = build_rollout_fixture(self.root / "scope-audit")
            final = run_slice(fixture)
            self.assertEqual(final.state.position["phase"], "terminal")
            original = fixture.store.get_artifact(final.state.requirements_ref, private=True)
            returned_copy = fixture.store.get_artifact(final.state.requirements_ref, private=True)
            returned_copy["active"]["mutated_after_scope"] = "poison"
            self.assertEqual(
                fixture.store.get_artifact(final.state.requirements_ref, private=True),
                original,
            )


if __name__ == "__main__":
    unittest.main()
