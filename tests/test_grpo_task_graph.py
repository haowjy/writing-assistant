"""Native task-graph GRPO profile and fail-closed resume/step admission."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from writing_agent.grpo_checkpoint import seal_directory
from writing_agent.grpo_config import TaskGraphGRPOSettings, trainer_config
from writing_agent.grpo_task_graph import (
    TaskGraphGroupPending,
    TaskGraphResumeLocationRefused,
    TaskGraphResumeRefused,
    TaskGraphRollouts,
    TaskGraphTrainingError,
    _reserve_step,
    task_graph_behavior_policy_ref,
    task_graph_resume_preflight,
    train_task_graph,
)
from writing_agent.native_audit import TrainingAuditError
from writing_agent.native_gemma import assert_active_adapter
from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_records import TrainingAdmissionV1
from writing_agent.task_graph_store import TaskGraphStore


def _settings(**overrides) -> TaskGraphGRPOSettings:
    values = {
        "model_id": "caller-owned/tiny-gemma4",
        "revision": "a" * 40,
        "runtime_profile": "task-graph-v1",
        "loss_type": "dapo",
        "enable_thinking": False,
        "group_size": 4,
        "microbatch_size": 1,
        "max_steps": 3,
        "max_tokens": 512,
        "max_generated_tokens": 1536,
        "context_tokens": 4096,
    }
    values.update(overrides)
    return TaskGraphGRPOSettings(**values)


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    required = (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_config.json",
        "adapter_model.safetensors",
    )
    for name in required:
        path = checkpoint / name
        if name == "trainer_state.json":
            path.write_text(json.dumps({"global_step": step}))
        else:
            path.write_bytes(b"checkpoint fixture")
    seal_directory(checkpoint, "fixture-identity", "trainer", group_files={})
    return checkpoint


def _rollouts_scaffold(root: Path, status: str):
    store = TaskGraphStore(root, verifier=LineageGate())
    pin = store.put_artifact({"fixture": "behavior-policy"})
    group_id = "b" * 64
    spec = SimpleNamespace(group_id=group_id, members=(), policy={})
    decision = SimpleNamespace(
        status=status,
        to_wire=lambda: {"fixture": "decision", "group_id": group_id, "status": status},
    )

    class Coordinator:
        def __init__(self):
            self.seal_calls = 0

        def seal(self, *_args, **_kwargs):
            self.seal_calls += 1
            return spec

        def finalize(self, _spec):
            return decision

        def record_training_admission(self, _spec, admission):
            admission_ref = store.put_artifact(admission.to_wire())
            directory = root / "groups" / group_id
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "training-batch.json").write_text(
                json.dumps({"batch_ref": admission.batch_ref})
            )
            (directory / "training-admission.json").write_text(
                json.dumps({"admission_ref": admission_ref})
            )

        def record_training_consumed(self, _spec, receipt):
            directory = root / "groups" / group_id
            (directory / "trainer-consumed.json").write_text(json.dumps(receipt))

    coordinator = Coordinator()
    task = SimpleNamespace(
        environment=SimpleNamespace(store=store, session=None),
        entry_checkpoint_id="a" * 64,
    )
    rollouts = object.__new__(TaskGraphRollouts)
    rollouts.task_ids = ("task-1",)
    rollouts.task_entries = {"task-1": task}
    rollouts.environments = {"task-1": task.environment}
    rollouts.settings = SimpleNamespace(group_size=2, max_steps=1, seed=5)
    rollouts.output = root
    rollouts.runtime_manifest = object()
    rollouts.manifest_descriptors = ()
    rollouts.model_ref = store.put_artifact({"fixture": "model"})
    rollouts.experiment_identity = "c" * 64
    rollouts.base_revision = "d" * 40
    rollouts.adapter_name = "default"
    rollouts.tokenizer_root = None
    rollouts.sample_backend_factory = None
    rollouts.observer = None
    rollouts.audit_function = None
    rollouts.invocation_id = None
    rollouts.store = store
    rollouts.session = None
    rollouts._context_policy_ref = store.put_artifact({"fixture": "context-policy"})
    rollouts._ensure_runtime = lambda _trainer: None
    rollouts._behavior_policy_ref = lambda _model, _step: pin
    rollouts._gatherers = lambda _task: None
    rollouts.last_spec = None
    rollouts.last_decision = None
    rollouts.last_admission = None
    return rollouts, coordinator, decision, store


class TaskGraphSettingsTests(unittest.TestCase):
    def test_profile_is_dapo_unscaled_nonthinking_and_keeps_every_checkpoint(self):
        settings = _settings()
        settings.validate()
        values = trainer_config(settings, "unused", use_cpu=True, bf16=False)
        self.assertEqual(values["loss_type"], "dapo")
        self.assertEqual(values["scale_rewards"], "none")
        self.assertIsNone(values["save_total_limit"])
        for override in (
            {"loss_type": "grpo"},
            {"enable_thinking": True},
            {"runtime_profile": "probe"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                _settings(**override).validate()

    def test_behavior_policy_binds_base_adapter_step_and_experiment(self):
        baseline = task_graph_behavior_policy_ref("base-r1", "a" * 64, 2, "experiment")
        self.assertEqual(
            baseline,
            task_graph_behavior_policy_ref("base-r1", "a" * 64, 2, "experiment"),
        )
        self.assertEqual(
            len(
                {
                    baseline,
                    task_graph_behavior_policy_ref("base-r2", "a" * 64, 2, "experiment"),
                    task_graph_behavior_policy_ref("base-r1", "b" * 64, 2, "experiment"),
                    task_graph_behavior_policy_ref("base-r1", "a" * 64, 3, "experiment"),
                    task_graph_behavior_policy_ref("base-r1", "a" * 64, 2, "other"),
                }
            ),
            5,
        )

    def test_checkpoint_resume_refuses_any_group_reservation_at_or_after_global_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            groups = output / "groups"
            groups.mkdir()
            checkpoint = _checkpoint(output, 2)
            self.assertEqual(task_graph_resume_preflight(output, checkpoint), 2)

            (groups / "step-000002").write_text('{"status":"halted"}')
            with self.assertRaises(TaskGraphResumeRefused):
                task_graph_resume_preflight(output, checkpoint)

    def test_resume_preflight_refuses_another_runs_checkpoint_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run-b"
            output.mkdir()
            (output / "sentinel").write_text("unchanged")
            checkpoint = root / "run-a" / "checkpoint-2"
            checkpoint.mkdir(parents=True)
            (checkpoint / "complete.json").write_text(json.dumps({"identity": "fixture"}))
            before = {
                path.relative_to(root): (
                    "directory" if path.is_dir() else "file",
                    path.read_bytes() if path.is_file() else None,
                )
                for path in sorted(root.rglob("*"))
            }

            with (
                patch("writing_agent.grpo_task_graph.verify_checkpoint") as verify,
                self.assertRaises(TaskGraphResumeLocationRefused),
            ):
                task_graph_resume_preflight(output, checkpoint)
            verify.assert_not_called()

            after = {
                path.relative_to(root): (
                    "directory" if path.is_dir() else "file",
                    path.read_bytes() if path.is_file() else None,
                )
                for path in sorted(root.rglob("*"))
            }
            self.assertEqual(after, before)

    def test_resume_preflight_runs_before_trainer_api_or_model_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = _checkpoint(output, 2)
            groups = output / "groups"
            groups.mkdir()
            (groups / "step-000002").write_text('{"status":"invalid"}')
            model_factory = Mock(side_effect=AssertionError("model loaded before refusal"))
            with patch("writing_agent.grpo_task_graph.load_trainer_api") as load_api:
                with self.assertRaises(TaskGraphResumeRefused):
                    train_task_graph(
                        (),
                        output,
                        settings=_settings(),
                        model=None,
                        model_factory=model_factory,
                        tokenizer=None,
                        manifest_descriptors=(),
                        runtime_identity={"fixture": True},
                        resume_from_checkpoint=checkpoint,
                    )
                load_api.assert_not_called()
            model_factory.assert_not_called()

    def test_external_resume_is_refused_before_model_api_or_output_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run-b"
            output.mkdir()
            (output / "sentinel").write_text("unchanged")
            checkpoint = root / "run-a" / "checkpoint-2"
            checkpoint.mkdir(parents=True)
            (checkpoint / "complete.json").write_text(json.dumps({"identity": "fixture"}))
            model_factory = Mock(side_effect=AssertionError("model loaded before refusal"))
            before = {
                path.relative_to(root): (
                    "directory" if path.is_dir() else "file",
                    path.read_bytes() if path.is_file() else None,
                )
                for path in sorted(root.rglob("*"))
            }

            with (
                patch("writing_agent.grpo_task_graph.verify_checkpoint") as verify,
                patch("writing_agent.grpo_task_graph.verify_runtime"),
                patch("writing_agent.grpo_task_graph.load_trainer_api") as load_api,
            ):
                with self.assertRaises(TaskGraphResumeLocationRefused):
                    train_task_graph(
                        (),
                        output,
                        settings=_settings(),
                        model=None,
                        model_factory=model_factory,
                        tokenizer=object(),
                        manifest_descriptors=(),
                        runtime_identity={"fixture": True},
                        resume_from_checkpoint=checkpoint,
                    )
                verify.assert_not_called()
                load_api.assert_not_called()
            model_factory.assert_not_called()
            after = {
                path.relative_to(root): (
                    "directory" if path.is_dir() else "file",
                    path.read_bytes() if path.is_file() else None,
                )
                for path in sorted(root.rglob("*"))
            }
            self.assertEqual(after, before)

    def test_step_reservation_prevents_a_second_attempt_even_before_group_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            groups = Path(temporary) / "groups"
            reservation = _reserve_step(groups, 1, "task-1")
            self.assertEqual(json.loads(reservation.read_text())["status"], "sealing")
            with self.assertRaises(TaskGraphTrainingError):
                _reserve_step(groups, 1, "task-1")


class TaskGraphRolloutFailureTests(unittest.TestCase):
    def test_pending_and_invalid_groups_halt_with_durable_decision_and_no_reseal(self):
        for status in ("pending", "invalid"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "trainer-output"
                rollouts, coordinator, decision, store = _rollouts_scaffold(root, status)
                trainer = SimpleNamespace(
                    state=SimpleNamespace(global_step=0),
                    model=object(),
                    processing_class=None,
                    optimizer_steps=0,
                )
                with (
                    patch(
                        "writing_agent.grpo_task_graph.GroupCoordinatorV1",
                        return_value=coordinator,
                    ),
                    patch("writing_agent.grpo_task_graph._native_policy", return_value={}),
                    patch("writing_agent.grpo_task_graph.assert_active_adapter"),
                ):
                    with self.assertRaises(TaskGraphGroupPending):
                        rollouts(["task-1", "task-1"], trainer)
                    receipt = json.loads((root / "groups" / "step-000000").read_text())
                    self.assertEqual(receipt["status"], status)
                    self.assertEqual(
                        store.get_artifact(receipt["decision_ref"]), decision.to_wire()
                    )
                    self.assertFalse((root / "batches" / "step-000000.json").exists())
                    self.assertEqual(trainer.optimizer_steps, 0)
                    with self.assertRaises(TaskGraphTrainingError):
                        rollouts(["task-1", "task-1"], trainer)
                    self.assertEqual(coordinator.seal_calls, 1)

    def test_disabled_adapter_halts_before_seal_and_keeps_the_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trainer-output"
            rollouts, coordinator, _decision, _store = _rollouts_scaffold(root, "invalid")
            disabled_layer = SimpleNamespace(disable_adapters=True)
            trainer = SimpleNamespace(
                state=SimpleNamespace(global_step=0),
                model=SimpleNamespace(
                    active_adapters=("default",), modules=lambda: [disabled_layer]
                ),
                processing_class=None,
                optimizer_steps=0,
            )
            with patch(
                "writing_agent.grpo_task_graph.GroupCoordinatorV1", return_value=coordinator
            ):
                with self.assertRaisesRegex(TaskGraphTrainingError, "disabled"):
                    rollouts(["task-1", "task-1"], trainer)
            receipt = json.loads((root / "groups" / "step-000000").read_text())
            self.assertEqual(receipt["status"], "halted")
            self.assertIn("disabled", receipt["failure"])
            self.assertEqual(coordinator.seal_calls, 0)
            self.assertEqual(trainer.optimizer_steps, 0)

    def test_audit_refusal_is_persisted_but_never_returns_a_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trainer-output"
            rollouts, coordinator, decision, store = _rollouts_scaffold(root, "ready")
            before = rollouts._behavior_policy_ref(None, 0)
            batch_ref = store.put_artifact({"fixture": "batch"})
            decision_ref = store.put_artifact(decision.to_wire())
            admission = TrainingAdmissionV1(
                group_id="b" * 64,
                decision_ref=decision_ref,
                batch_ref=batch_ref,
                audit_version="fixture-audit-v1",
                renderer_ref="1" * 64,
                tokenizer_descriptor_ref="2" * 64,
                adapter_hash_before=before,
                adapter_hash_after=before,
                members=[
                    {
                        "member_id": "member-0",
                        "status": "refused",
                        "failed_check": "policy_and_adapter_binding",
                    }
                ],
            )
            rollouts.audit_function = Mock(return_value=admission)
            rollouts._training_rows = lambda _batch: []
            trainer = SimpleNamespace(
                state=SimpleNamespace(global_step=0),
                model=object(),
                processing_class=object(),
                optimizer_steps=0,
            )
            with (
                patch(
                    "writing_agent.grpo_task_graph.GroupCoordinatorV1",
                    return_value=coordinator,
                ),
                patch("writing_agent.grpo_task_graph._native_policy", return_value={}),
                patch("writing_agent.grpo_task_graph.assert_active_adapter"),
                patch(
                    "writing_agent.grpo_task_graph.TrainingBatchV1.from_dict",
                    return_value=SimpleNamespace(members=()),
                ),
            ):
                with self.assertRaises(TrainingAuditError):
                    rollouts(["task-1", "task-1"], trainer)
            reservation = json.loads((root / "groups" / "step-000000").read_text())
            receipt = json.loads(
                (root / "groups" / ("b" * 64) / "trainer-consumed.json").read_text()
            )
            self.assertEqual(reservation["status"], "audit-refused")
            self.assertEqual(receipt["admission_status"], "refused")
            self.assertEqual(
                store.get_artifact(reservation["training_admission_ref"]), admission.to_wire()
            )
            self.assertEqual(trainer.optimizer_steps, 0)

    def test_adapter_drift_is_recorded_and_never_returns_a_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trainer-output"
            rollouts, coordinator, _decision, store = _rollouts_scaffold(root, "ready")
            before = store.put_artifact({"policy": "before"})
            after = store.put_artifact({"policy": "after"})
            rollouts._behavior_policy_ref = Mock(side_effect=(before, after))
            batch_ref = store.put_artifact({"fixture": "batch"})
            decision_ref = store.put_artifact(
                {"fixture": "decision", "group_id": "b" * 64, "status": "ready"}
            )
            admission = TrainingAdmissionV1(
                group_id="b" * 64,
                decision_ref=decision_ref,
                batch_ref=batch_ref,
                audit_version="fixture-audit-v1",
                renderer_ref="1" * 64,
                tokenizer_descriptor_ref="2" * 64,
                adapter_hash_before=before,
                adapter_hash_after=after,
                members=[{"member_id": "member-0", "status": "admitted", "failed_check": None}],
            )
            rollouts.audit_function = Mock(return_value=admission)
            rollouts._training_rows = lambda _batch: []
            trainer = SimpleNamespace(
                state=SimpleNamespace(global_step=0),
                model=object(),
                processing_class=object(),
                optimizer_steps=0,
            )
            with (
                patch(
                    "writing_agent.grpo_task_graph.GroupCoordinatorV1",
                    return_value=coordinator,
                ),
                patch("writing_agent.grpo_task_graph._native_policy", return_value={}),
                patch("writing_agent.grpo_task_graph.assert_active_adapter"),
                patch(
                    "writing_agent.grpo_task_graph.TrainingBatchV1.from_dict",
                    return_value=SimpleNamespace(members=()),
                ),
            ):
                with self.assertRaisesRegex(TaskGraphTrainingError, "adapter changed"):
                    rollouts(["task-1", "task-1"], trainer)
            reservation = json.loads((root / "groups" / "step-000000").read_text())
            self.assertEqual(reservation["status"], "adapter-drift")
            rollouts.audit_function.assert_called_once()
            self.assertEqual(
                rollouts.audit_function.call_args.kwargs["adapter_hash_before"], before
            )
            self.assertEqual(rollouts.audit_function.call_args.kwargs["adapter_hash_after"], after)
            self.assertEqual(trainer.optimizer_steps, 0)

    def test_adapter_must_be_active_and_not_disabled(self):
        active = SimpleNamespace(active_adapters=("default",), modules=lambda: [])
        assert_active_adapter(active, "default")
        with self.assertRaises(AdapterContractError):
            assert_active_adapter(active, "other")
        disabled = SimpleNamespace(
            active_adapters=("default",),
            modules=lambda: [SimpleNamespace(disable_adapters=True)],
        )
        with self.assertRaises(AdapterContractError):
            assert_active_adapter(disabled, "default")


if __name__ == "__main__":
    unittest.main()
