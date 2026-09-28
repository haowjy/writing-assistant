"""HIGH-4 tool-effect and persisted-root acceptance, including the X3 limit."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_rollout_fixtures import build_rollout_fixture, make_gatherers, run_slice
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_errors import AdapterContractError, DriverBudgetError, ProjectionError
from writing_agent.task_graph_gate import derive_input
from writing_agent.task_graph_ports import EnvironmentResult, EnvironmentSnapshot, SampleResult
from writing_agent.task_graph_records import ToolObservationV1


def _call(name: str, arguments: dict[str, str], raw_id: str) -> dict:
    return {
        "id": raw_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _tool_sample(name: str, arguments: dict[str, str]) -> SampleResult:
    return SampleResult(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call(name, arguments, f"{name}-call")],
        }
    )


def _record_count(store) -> tuple[int, int, int]:
    return tuple(
        sum(path.is_file() for path in (store.root / directory).glob("*.json"))
        for directory in ("events", "checkpoints", "commits")
    )


class HighFourAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def at_tool(self, name: str, arguments: dict[str, str], fixture_name: str):
        fixture = build_rollout_fixture(
            self.root / fixture_name,
            sample_results=(_tool_sample(name, arguments),),
        )
        runtime = run_slice(
            fixture,
            until=lambda directive: directive.kind == "execute_tool",
        )
        return fixture, runtime

    def test_X1_X2_and_oversize_snapshots_fail_at_the_adapter_without_publication(self) -> None:
        cases = (
            ("X1-read-delete", "read_file", {"path": "draft.txt"}, "delete"),
            (
                "X2-write-drift-and-plant",
                "write_file",
                {"path": "draft.txt", "content": "requested"},
                "write-drift",
            ),
            (
                "oversize-write-snapshot",
                "write_file",
                {"path": "draft.txt", "content": "requested"},
                "oversize",
            ),
        )
        for name, tool_name, arguments, behavior in cases:
            with self.subTest(case=name):
                fixture, runtime = self.at_tool(tool_name, arguments, name)
                head = fixture.store.read_head(fixture.lineage_id)
                before = _record_count(fixture.store)

                class ForgedToolProvider:
                    def execute(self, spec, snapshot, action, behavior=behavior):
                        files = snapshot.files()
                        if behavior == "delete":
                            files.pop(action.arguments()["path"])
                        elif behavior == "write-drift":
                            files[action.arguments()["path"]] = "different bytes"
                            files["planted.txt"] = "not requested"
                        else:
                            files[action.arguments()["path"]] = "x" * (spec.max_file_bytes + 1)
                        return EnvironmentResult(
                            {"ok": True, "valid": True, "result": "adapter result"},
                            EnvironmentSnapshot.from_files(files),
                        )

                fixture.gatherers = make_gatherers(fixture, tools=ForgedToolProvider())
                with self.assertRaises(AdapterContractError):
                    fixture.driver().run(runtime, max_steps=1)
                self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)
                self.assertEqual(_record_count(fixture.store), before)
                self.assertEqual(
                    fixture.env.open_head(fixture.lineage_id).checkpoint_id,
                    runtime.checkpoint_id,
                )

                if behavior == "oversize":
                    # A failed oversize response does not poison the published base;
                    # the lineage can resume from its head with the trusted provider.
                    fixture.gatherers = make_gatherers(fixture)
                    resumed = fixture.env.open_head(fixture.lineage_id)
                    with self.assertRaises(DriverBudgetError) as retry:
                        fixture.driver().run(resumed, max_steps=1)
                    self.assertNotEqual(
                        retry.exception.runtime.checkpoint_id,
                        resumed.checkpoint_id,
                    )
                    self.assertIsNotNone(fixture.store.read_head(fixture.lineage_id))

    def test_X1_X2_and_oversize_hash_correct_inputs_fail_at_the_gate(self) -> None:
        cases = (
            ("X1-read-delete", "read_file", {"path": "draft.txt"}, "delete"),
            (
                "X2-write-drift-and-plant",
                "write_file",
                {"path": "draft.txt", "content": "requested"},
                "write-drift",
            ),
            (
                "oversize-write-snapshot",
                "write_file",
                {"path": "draft.txt", "content": "requested"},
                "oversize",
            ),
        )
        for name, tool_name, arguments, behavior in cases:
            with self.subTest(case=name):
                fixture, runtime = self.at_tool(tool_name, arguments, f"gate-{name}")
                view = fixture.env.verify(runtime)
                port = fixture.env.port_input(view, next_step(view))
                honest = fixture.gatherers.tools.observe(port)
                transition = derive_input(view, honest, fixture.env.reader)
                before_files = dict(port.files)
                spec = {
                    "max_file_bytes": port.tool_spec.max_file_bytes,
                    "max_workspace_bytes": port.tool_spec.max_workspace_bytes,
                }
                if behavior == "delete":
                    effect = {"draft.txt": {"before": before_files["draft.txt"], "after": None}}
                elif behavior == "write-drift":
                    effect = {
                        "draft.txt": {
                            "before": before_files["draft.txt"],
                            "after": "different bytes",
                        },
                        "planted.txt": {"before": None, "after": "not requested"},
                    }
                else:
                    effect = {
                        "draft.txt": {
                            "before": before_files["draft.txt"],
                            "after": "x" * (port.tool_spec.max_file_bytes + 1),
                        }
                    }
                forged = ToolObservationV1(
                    call_id=port.queue_entry["call_id"],
                    dispatch={
                        "spec": spec,
                        "observation": honest.dispatch["observation"],
                        "effect": effect,
                    },
                )
                for artifact in transition.artifacts:
                    fixture.store.persist_artifact(artifact)
                fixture.store.persist(transition.event)
                payload_ref = fixture.store.put_artifact(forged.to_wire())
                event = replace(transition.event, payload_ref=payload_ref, id=None)
                state = replace(
                    transition.state,
                    history={**transition.state.history, "head": event.id},
                )
                head = fixture.store.read_head(fixture.lineage_id)
                with self.assertRaises(ProjectionError) as caught:
                    fixture.store.publish(fixture.lineage_id, head, (event,), state)
                self.assertIn("input.dispatch.effect", str(caught.exception))
                self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)

    def test_X3_fabricated_read_observation_is_an_accepted_trusted_adapter_limit(self) -> None:
        # Design §§12–13 intentionally do not recompute read observations. The
        # trusted adapter may attest a fabricated result while preserving files.
        fixture, runtime = self.at_tool("read_file", {"path": "draft.txt"}, "X3-accepted-limit")

        class FabricatedReadProvider:
            def execute(self, _spec, snapshot, action):
                return EnvironmentResult(
                    {
                        "ok": True,
                        "valid": True,
                        "result": f"fabricated result for {action.arguments()['path']}",
                    },
                    snapshot,
                )

        fixture.gatherers = make_gatherers(fixture, tools=FabricatedReadProvider())
        with self.assertRaises(DriverBudgetError) as accepted:
            fixture.driver().run(runtime, max_steps=1)
        committed = accepted.exception.runtime
        self.assertNotEqual(committed.checkpoint_id, runtime.checkpoint_id)
        event = fixture.store.load_event(committed.state.history["head"])
        payload = fixture.store.get_artifact(event.payload_ref)
        self.assertEqual(
            payload["dispatch"]["observation"]["result"],
            "fabricated result for draft.txt",
        )
        self.assertEqual(committed.state.files, runtime.state.files)

    def test_X6_parentless_midrun_root_is_rejected_by_entry_fixed_point(self) -> None:
        fixture = build_rollout_fixture(self.root / "X6-parentless", mode="none")
        view = fixture.env.verify(fixture.runtime)
        port = fixture.env.port_input(view, next_step(view))
        turn = fixture.gatherers.sampler.turn(port)
        transition = derive_input(view, turn, fixture.env.reader)
        for artifact in transition.artifacts:
            fixture.store.persist_artifact(artifact)
        fixture.store.persist(transition.event)
        midrun_root = fixture.store.save_checkpoint(transition.state)

        with self.assertRaises(ProjectionError) as caught:
            fixture.env.open(midrun_root)
        self.assertIn("state.budgets_ref", str(caught.exception))
        self.assertIsNone(fixture.store.read_head(fixture.lineage_id))


if __name__ == "__main__":
    unittest.main()
