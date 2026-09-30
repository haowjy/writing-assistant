"""Native task-graph rollout admission for the shared public-TRL DAPO trainer."""

from __future__ import annotations

import fcntl
import hashlib
import json
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from writing_agent.atomic_io import atomic_write_json
from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo import GRPOSettings, trainer_config
from writing_agent.grpo_checkpoint import verify_checkpoint
from writing_agent.grpo_identity import adapter_tensor_hash, base_tensor_identity
from writing_agent.grpo_runtime import STREAMING, verify_runtime
from writing_agent.grpo_task_graph_errors import (
    TaskGraphGroupPending,
    TaskGraphResumeRefused,
    TaskGraphTrainingError,
)
from writing_agent.grpo_task_graph_observer import TaskGraphLossObserver
from writing_agent.grpo_trainer import load_trainer_api, run_trainer
from writing_agent.native_audit import (
    audit_training_batch,
    require_training_admission,
)
from writing_agent.native_gemma import NativeGemmaSampleBackend, assert_active_adapter
from writing_agent.task_graph import domain_hash, thaw
from writing_agent.task_graph_composition import RuntimeSession
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_errors import AdapterContractError, DriverBudgetError
from writing_agent.task_graph_gate import StoreArtifactReader
from writing_agent.task_graph_gatherers import (
    CheckRunner,
    Gatherers,
    SamplingRunner,
    ScriptedAuthorSource,
    ToolRunner,
)
from writing_agent.task_graph_group import GroupCoordinatorV1
from writing_agent.task_graph_group_contract import derive_group_seed
from writing_agent.task_graph_local import DeterministicEvaluator, LocalTextToolProvider
from writing_agent.task_graph_record_contracts import ContextPolicyV1, GroupError
from writing_agent.task_graph_records import RuntimeManifestV2, WriterTurnV2
from writing_agent.task_graph_rollout import RolloutDriver
from writing_agent.task_graph_token_ledger import decode_u32_token_ids
from writing_agent.task_graph_training_export import read_advantage_f64
from writing_agent.task_graph_training_records import TrainingBatchV1


@dataclass(frozen=True)
class TaskGraphTaskV1:
    """One admitted writer entry available to the task-graph trainer schedule."""

    task_id: str
    environment: RolloutEnvironment
    entry_checkpoint_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for char in self.task_id
            )
        ):
            raise ValueError("task-graph task ID must be path safe")
        if not isinstance(self.environment, RolloutEnvironment):
            raise TypeError("task-graph task requires a RolloutEnvironment")
        if not isinstance(self.entry_checkpoint_id, str) or len(self.entry_checkpoint_id) != 64:
            raise ValueError("task-graph task requires a checkpoint identity")


def task_graph_resume_preflight(output: Path | str, checkpoint: Path | str) -> int:
    """Verify the checkpoint and refuse groups at/after its global step, model-free."""
    checkpoint = Path(checkpoint)
    try:
        marker = json.loads((checkpoint / "complete.json").read_text())
        checkpoint_identity = marker["identity"]
        step = verify_checkpoint(checkpoint, checkpoint_identity)
    except (OSError, TypeError, KeyError, json.JSONDecodeError, ValueError) as exc:
        raise TaskGraphResumeRefused("resume checkpoint is incomplete or invalid") from exc
    groups = Path(output) / "groups"
    try:
        existing = GroupCoordinatorV1.groups_by_sequence(groups)
    except (GroupError, OSError) as exc:
        raise TaskGraphResumeRefused("unrecognized task-graph group evidence") from exc
    if any(sequence >= step for sequence in existing):
        raise TaskGraphResumeRefused("task-graph evidence exists at or after the checkpoint step")
    return step


def _reserve_step(groups_root: Path, step: int, task_id: str) -> Path:
    """Atomically claim a step before sealing, closing the crash-before-receipt window."""
    groups_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = groups_root / f".step-{step:06d}.lock"
    reservation = groups_root / f"step-{step:06d}"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if reservation.exists():
            raise TaskGraphTrainingError("a task-graph group was already attempted for this step")
        atomic_write_json(
            reservation,
            {"schema": 1, "step": step, "task_id": task_id, "status": "sealing"},
            canonical=True,
            create_parent=True,
        )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return reservation


def task_graph_behavior_policy_ref(
    base_revision: str, adapter_sha256: str, global_step: int, experiment_identity: str
) -> str:
    """Bind the behavior policy to base, live adapter, training step and experiment."""
    return domain_hash(
        "payload",
        [
            "TaskGraphBehaviorPolicyV1",
            base_revision,
            adapter_sha256,
            global_step,
            experiment_identity,
        ],
    )


def task_graph_experiment_manifest(
    tasks: Sequence[TaskGraphTaskV1],
    settings: GRPOSettings,
    runtime_manifest: RuntimeManifestV2,
    *,
    base_identity: Mapping[str, Any],
    implementation: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable source/environment/graph/reward/manifest/recipe identity."""
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task-graph schedule needs unique admitted task IDs")
    environments = []
    all_sources: set[str] = set()
    for task in tasks:
        env = task.environment
        runtime = env.open(task.entry_checkpoint_id)
        view = env.verify(runtime)
        instance = env.store.load_instance(view.state.instance_ref)
        sources = tuple(sorted(instance.source_refs))
        all_sources.update(sources)
        node = view.node
        environments.append(
            {
                "task_id": task.task_id,
                "entry_checkpoint_id": task.entry_checkpoint_id,
                "graph_instance_ref": instance.identity(),
                "source_refs": list(sources),
                "environment_digest": fingerprint(
                    {
                        "environment": view.state.to_dict(),
                        "context": {
                            "content_ref": view.context.content_ref,
                            "revision_ref": view.context.revision_ref,
                            "rendering": thaw(view.context.rendering),
                            "tools": thaw(view.context.tools),
                        },
                        "contract": node.contract.to_dict(),
                    }
                ),
                "reward_contract_ref": node.reward_contract.identity(),
            }
        )
    recipe = {
        # Preserve the recipe shape while reward scaling is profile-derived, not an input.
        "settings": {**asdict(settings), "scale_rewards": "none"},
        "loss_type": "dapo",
        "beta": 0,
        "num_iterations": 1,
        "scale_rewards": "none",
        "schedule": "ordered task IDs by global_step modulo task count",
        "group_seed": "derive_group_seed(experiment_seed, task-graph-step, global_step)",
        "checkpoints": "keep-all",
    }
    return {
        "schema": "task-graph-experiment-v1",
        "sources": sorted(all_sources),
        "task_graphs": environments,
        "runtime_manifest": runtime_manifest.to_wire(),
        "runtime_manifest_ref": runtime_manifest.identity(),
        "runtime_identity": dict(runtime_identity),
        "base_identity": dict(base_identity),
        "recipe": recipe,
        "implementation": dict(implementation),
    }


def _task_graph_identity(experiment: Mapping[str, Any], *, source_root: Path | None = None):
    """Bind the frozen task-graph experiment to its trainer-side source files."""
    source_root = Path(source_root) if source_root is not None else Path(__file__).parent
    manifest = {
        "identity_version": "task-graph-experiment-v1",
        "experiment": experiment,
        "code": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                {
                    *source_root.glob("grpo*.py"),
                    *source_root.glob("native_*.py"),
                    *source_root.glob("task_graph*.py"),
                }
            )
        },
    }
    manifest = json.loads(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return fingerprint(manifest), manifest


def _native_policy(
    task: TaskGraphTaskV1,
    manifest: RuntimeManifestV2,
    *,
    model_ref: str,
    behavior_policy_ref: str,
    context_policy_ref: str,
) -> dict[str, str]:
    env = task.environment
    runtime = env.open(task.entry_checkpoint_id)
    view = env.verify(runtime)
    contract = view.node.contract
    script_ref = contract.interaction_contract.script_ref
    simulator_ref = env.store.put_artifact(
        {"implementation": "scripted-author-v1", "script_ref": script_ref}
    )
    return {
        "model_ref": model_ref,
        "behavior_policy_ref": behavior_policy_ref,
        "tokenizer_ref": manifest.tokenizer.identity(),
        "template_ref": manifest.renderer.template_ref,
        "adapter_ref": manifest.identity(),
        "decoding_ref": manifest.decoding.identity(),
        "simulator_ref": simulator_ref,
        "context_policy_ref": context_policy_ref,
        "controller_ref": view.state.versions_ref,
        "rng_derivation_version": "sha256-domain-v1",
    }


class TaskGraphRollouts:
    """TRL rollout callback: one sealed task-graph group per step, never resampled."""

    def __init__(
        self,
        invocation_id: str,
        *,
        tasks,
        settings,
        output,
        task_entries: Sequence[TaskGraphTaskV1],
        runtime_manifest: RuntimeManifestV2,
        manifest_descriptors: tuple,
        model_ref: str,
        experiment_identity: str,
        base_revision: str,
        adapter_name: str = "default",
        tokenizer_root: Path | str | None = None,
        sample_backend_factory: Callable[..., Any] = NativeGemmaSampleBackend,
        observer=None,
        audit_function=audit_training_batch,
    ) -> None:
        self.task_ids = tuple(item["id"] for item in tasks)
        self.task_entries = {entry.task_id: entry for entry in task_entries}
        self.environments = {
            task_id: entry.environment for task_id, entry in self.task_entries.items()
        }
        if set(self.task_ids) != set(self.task_entries):
            raise ValueError("TRL task IDs differ from task-graph entry registry")
        self.settings = settings
        self.output = Path(output)
        self.runtime_manifest = runtime_manifest
        self.manifest_descriptors = manifest_descriptors
        self.model_ref = model_ref
        self.experiment_identity = experiment_identity
        self.base_revision = base_revision
        self.adapter_name = adapter_name
        self.tokenizer_root = tokenizer_root
        self.sample_backend_factory = sample_backend_factory
        self.observer = observer
        self.audit_function = audit_function
        self.invocation_id = invocation_id
        stores = {entry.environment.store.root.resolve() for entry in task_entries}
        if stores != {self.output.resolve()}:
            raise ValueError(
                "task-graph store root must equal trainer output for checkpoint binding"
            )
        self.store = task_entries[0].environment.store
        self._context_policy_ref = self.store.put_artifact(ContextPolicyV1("carry").to_wire())
        self.store.get_artifact(model_ref)
        self.backend = None
        self.session = None
        self._tools = None
        self._evaluator = None
        self._manifest_checked = False
        self.last_spec = None
        self.last_decision = None
        self.last_admission = None

    def __call__(self, prompts, trainer):
        if len(prompts) != self.settings.group_size or len(set(prompts)) != 1:
            raise TaskGraphTrainingError("expected exactly one complete same-task group")
        self.last_spec = None
        self.last_decision = None
        self.last_admission = None
        step = trainer.state.global_step
        if not 0 <= step < self.settings.max_steps:
            raise TaskGraphTrainingError("task-graph schedule exceeded its frozen step budget")
        task_id = prompts[0]
        expected = self.task_ids[step % len(self.task_ids)]
        if task_id != expected:
            raise TaskGraphTrainingError("rollout task differs from the frozen ordered schedule")
        self._refuse_existing_step(step)
        self._ensure_runtime(trainer)
        task = self.task_entries[task_id]
        environment = self.environments[task_id]
        reservation = _reserve_step(self.output / "groups", step, task_id)
        spec = None
        try:
            try:
                assert_active_adapter(trainer.model, self.adapter_name)
            except AdapterContractError as exc:
                raise TaskGraphTrainingError(str(exc)) from exc
            adapter_before = self._behavior_policy_ref(trainer.model, step)
            group_seed = derive_group_seed(self.settings.seed, "task-graph-step", step)
            coordinator = GroupCoordinatorV1(environment, session=self.session)
            policy = _native_policy(
                task,
                self.runtime_manifest,
                model_ref=self.model_ref,
                behavior_policy_ref=adapter_before,
                context_policy_ref=self._context_policy_ref,
            )
            spec = coordinator.seal(
                task.entry_checkpoint_id,
                policy=policy,
                group_seed=group_seed,
                group_sequence=step,
                member_count=self.settings.group_size,
                runner_mode="real",
                training_mode="native",
            )
            self.last_spec = spec
            atomic_write_json(
                reservation,
                {
                    "schema": 1,
                    "step": step,
                    "task_id": task_id,
                    "group_id": spec.group_id,
                    "status": "sealed",
                },
                canonical=True,
                create_parent=True,
            )
            gatherers = self._gatherers(task)
            failure = None
            for ordinal in range(len(spec.members)):
                try:
                    runtime = coordinator.start(spec, ordinal, policy=spec.policy)
                    run = RolloutDriver(environment, gatherers).run(runtime, max_steps=256)
                    if run.directive.kind != "done":
                        raise TaskGraphTrainingError(
                            f"task-graph member halted with directive {run.directive.kind}"
                        )
                    coordinator.collect_completed(spec, ordinal, run.runtime)
                except Exception as exc:
                    failure = exc
                    reason = (
                        f"DriverBudgetError: {exc.max_steps}"
                        if isinstance(exc, DriverBudgetError)
                        else f"{type(exc).__name__}: {exc}"
                    )
                    coordinator.collect_invalid(
                        spec,
                        ordinal,
                        reason=reason[:512],
                    )
                    break

            decision = coordinator.finalize(spec)
            self.last_decision = decision
            decision_ref = self.store.put_artifact(decision.to_wire())
            if failure is not None or decision.status in {"pending", "invalid"}:
                atomic_write_json(
                    reservation,
                    {
                        "schema": 1,
                        "step": step,
                        "task_id": task_id,
                        "group_id": spec.group_id,
                        "decision_ref": decision_ref,
                        "status": decision.status,
                        "failure": None
                        if failure is None
                        else f"{type(failure).__name__}: {failure}",
                    },
                    canonical=True,
                    create_parent=True,
                )
                raise TaskGraphGroupPending(
                    f"task-graph group {decision.status}; resampling is forbidden"
                ) from failure
            if decision.status not in {"ready", "tie"}:
                raise TaskGraphTrainingError("unknown finalized task-graph group status")

            adapter_after = self._behavior_policy_ref(trainer.model, step)
            try:
                assert_active_adapter(trainer.model, self.adapter_name)
            except AdapterContractError as exc:
                raise TaskGraphTrainingError(str(exc)) from exc
            admission = self.audit_function(
                self.store,
                spec,
                decision,
                trainer.processing_class,
                adapter_hash_before=adapter_before,
                adapter_hash_after=adapter_after,
                tokenizer_root=self.tokenizer_root,
            )
            coordinator.record_training_admission(spec, admission)
            self.last_admission = admission
            batch = TrainingBatchV1.from_dict(self.store.get_artifact(admission.batch_ref))
            members = self._training_rows(batch)
            self._save_consumed_batch(
                step,
                spec,
                decision,
                decision_ref,
                admission,
                members,
                reservation,
                adapter_before,
                adapter_after,
                task_id,
                coordinator,
            )
            require_training_admission(admission)
            if adapter_before != adapter_after:
                raise TaskGraphTrainingError("adapter changed while task-graph group was sampled")
            if self.observer is not None:
                self.observer.register_batch(step, members)
            return {
                "prompt_ids": [member["prompt_ids"] for member in members],
                "completion_ids": [member["completion_ids"] for member in members],
                "env_mask": [member["env_mask"] for member in members],
                "logprobs": None,
                "rollout_rewards": [member["advantage_f64"] for member in members],
            }
        except BaseException as exc:
            if reservation.exists():
                try:
                    receipt = json.loads(reservation.read_text())
                except (OSError, json.JSONDecodeError):
                    receipt = {"schema": 1, "step": step, "task_id": task_id}
                if receipt.get("status") in {"sealing", "sealed"}:
                    receipt["status"] = "halted"
                receipt["failure"] = f"{type(exc).__name__}: {exc}"[:1024]
                if spec is not None:
                    receipt["group_id"] = spec.group_id
                atomic_write_json(reservation, receipt, canonical=True, create_parent=True)
            raise

    def _ensure_runtime(self, trainer) -> None:
        if self._manifest_checked:
            if self.backend.model is not trainer.model:
                raise TaskGraphTrainingError("trainer replaced its live model during the run")
            return
        backend = self.sample_backend_factory(
            trainer.model,
            trainer.processing_class,
            manifest_descriptors=self.manifest_descriptors,
            adapter_name=self.adapter_name,
            model_ref=self.model_ref,
        )
        tools = LocalTextToolProvider()
        evaluator = DeterministicEvaluator()
        from writing_agent.task_graph_local import LocalWorkspaceEnvironment
        from writing_agent.task_graph_ports import RuntimeDependenciesV1

        dependencies = RuntimeDependenciesV1(
            backend,
            LocalWorkspaceEnvironment(tools),
            tools,
            evaluator,
        )
        manifest = dependencies.manifest()
        if not isinstance(manifest, RuntimeManifestV2) or manifest.identity() != (
            self.runtime_manifest.identity()
        ):
            raise TaskGraphTrainingError("live native runtime differs from experiment manifest")
        session = RuntimeSession.create(self.store, dependencies).bind(
            self.store, self.runtime_manifest.identity()
        )
        self.environments = {
            task_id: task.environment.with_session(session)
            for task_id, task in self.task_entries.items()
        }
        self.backend, self.session = backend, session
        self._tools, self._evaluator = tools, evaluator
        self._manifest_checked = True

    def _gatherers(self, task: TaskGraphTaskV1) -> Gatherers:
        checks = {
            check.identity(): check
            for node in task.environment.graph.nodes.values()
            for check in node.checks.values()
        }
        return Gatherers(
            SamplingRunner(self.store, self.backend),
            ToolRunner(self._tools),
            ScriptedAuthorSource(),
            CheckRunner(self.store, self._evaluator, checks),
        )

    def _behavior_policy_ref(self, model: Any, step: int) -> str:
        try:
            adapter = adapter_tensor_hash(model, self.adapter_name)
        except ValueError as exc:
            raise TaskGraphTrainingError(str(exc)) from exc
        value = [
            "TaskGraphBehaviorPolicyV1",
            self.base_revision,
            adapter,
            step,
            self.experiment_identity,
        ]
        reference = self.store.put_artifact(value)
        if reference != task_graph_behavior_policy_ref(
            self.base_revision, adapter, step, self.experiment_identity
        ):
            raise TaskGraphTrainingError("behavior-policy artifact identity differs")
        return reference

    def _refuse_existing_step(self, step: int) -> None:
        groups = self.output / "groups"
        try:
            existing = GroupCoordinatorV1.groups_by_sequence(groups)
        except (GroupError, OSError) as exc:
            raise TaskGraphTrainingError("task-graph group evidence is unrecognized") from exc
        if step in existing:
            raise TaskGraphTrainingError("a task-graph group already exists for this step")

    def _training_rows(self, batch: TrainingBatchV1) -> list[dict[str, Any]]:
        reader = StoreArtifactReader(self.store)
        rows = []
        for member in batch.members:
            prompt_bytes = reader.bytes_artifact(member["prompt_ids_ref"])
            completion_bytes = reader.bytes_artifact(member["completion_ids_ref"])
            prompt = decode_u32_token_ids(prompt_bytes, len(prompt_bytes) // 4)
            completion = decode_u32_token_ids(completion_bytes, len(completion_bytes) // 4)
            mask_bytes = reader.bytes_artifact(member["env_mask_ref"])
            sampled: list[float | None] = [None] * len(completion)
            for span in member["turn_spans"]:
                turn = WriterTurnV2.from_dict(self.store.get_artifact(span["turn_ref"]))
                evidence = reader.bytes_artifact(turn.logprobs["ref"])
                values = struct.unpack(f"<{len(evidence) // 4}f", evidence)
                start, end = span["completion_start"], span["completion_end"]
                generated = [value for value in mask_bytes[start:end] if value]
                if len(values) != len(generated):
                    raise TaskGraphTrainingError("sampled logprobs differ from exported turn span")
                cursor = 0
                for offset in range(start, end):
                    if mask_bytes[offset]:
                        sampled[offset] = float(values[cursor])
                        cursor += 1
            rows.append(
                {
                    "member_id": member["member_id"],
                    "prompt_ids": list(prompt),
                    "completion_ids": list(completion),
                    "env_mask": list(mask_bytes),
                    "advantage_f64": read_advantage_f64(member, reader),
                    "sampled_logprobs": sampled,
                    "turn_spans": [dict(span) for span in member["turn_spans"]],
                    "ledger_hash": member["ledger_hash"],
                }
            )
        return rows

    def _save_consumed_batch(
        self,
        step,
        spec,
        decision,
        decision_ref,
        admission,
        members,
        reservation,
        adapter_before,
        adapter_after,
        task_id,
        coordinator,
    ) -> None:
        batch_ref = admission.batch_ref
        admission_ref = admission.identity()
        receipt = {
            "schema": 1,
            "step": step,
            "group_id": spec.group_id,
            "decision_ref": decision_ref,
            "training_batch_ref": batch_ref,
            "training_admission_ref": admission_ref,
            "adapter_hash_before": adapter_before,
            "adapter_hash_after": adapter_after,
            "status": decision.status,
            "admission_status": (
                "admitted"
                if all(member["status"] == "admitted" for member in admission.members)
                else "refused"
            ),
        }
        if adapter_before != adapter_after:
            reservation_status = "adapter-drift"
        elif receipt["admission_status"] == "refused":
            reservation_status = "audit-refused"
        else:
            reservation_status = "consumed"
        coordinator.record_training_consumed(spec, receipt)
        save_json(
            self.output / "batches" / f"step-{step:06d}.json",
            {**receipt, "members": members},
        )
        atomic_write_json(
            reservation,
            {
                "schema": 1,
                "step": step,
                "task_id": task_id,
                "group_id": spec.group_id,
                "decision_ref": decision_ref,
                "training_batch_ref": batch_ref,
                "training_admission_ref": admission_ref,
                "status": reservation_status,
            },
            canonical=True,
            create_parent=True,
        )


def train_task_graph(
    task_entries: Sequence[TaskGraphTaskV1],
    output: Path | str,
    *,
    settings: GRPOSettings,
    model: Any | None,
    tokenizer: Any,
    manifest_descriptors: tuple,
    runtime_identity: Mapping[str, Any],
    model_factory: Callable[[], Any] | None = None,
    implementation: str = STREAMING,
    resume_from_checkpoint: Path | str | None = None,
    stop_after_steps: int | None = None,
    resume_checkpoint_identity: str | None = None,
    sample_backend_factory: Callable[..., Any] = NativeGemmaSampleBackend,
    trainer_callback_factory: Callable[[Any], Any] | None = None,
    tokenizer_root: Path | str | None = None,
    adapter_name: str = "default",
) -> dict[str, Any]:
    """Run an identity-bound task-graph experiment through the shared TRL lifecycle."""
    output = Path(output).resolve()
    if resume_from_checkpoint is not None:
        # Must precede API/model construction. The checkpoint's own sealed identity
        # is verified here; run_trainer later binds it to the current full identity.
        task_graph_resume_preflight(output, resume_from_checkpoint)
        if model_factory is None or model is not None:
            raise ValueError("resume requires a lazy model_factory and no preloaded model")
    settings.validate()
    if settings.runtime_profile != "task-graph-v1":
        raise ValueError("task-graph trainer requires runtime_profile='task-graph-v1'")
    if implementation != STREAMING:
        raise ValueError("task-graph DAPO requires the pinned public streaming TRL runtime")
    if tokenizer is None:
        raise ValueError("task-graph training requires the pinned Gemma tokenizer")
    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        raise ValueError("task-graph training needs explicit model/runtime identity")
    runtime = verify_runtime(implementation)
    api = load_trainer_api()
    if model is not None and model_factory is not None:
        raise ValueError("supply a base model or model_factory, not both")
    if model is None and model_factory is None:
        raise ValueError("task-graph training requires a base model or lazy model_factory")
    if model is None:
        model = model_factory()
    if isinstance(model, api.PeftModel) or getattr(model, "peft_config", None):
        raise ValueError("supply a fresh base model, not a PEFT model")
    if not isinstance(manifest_descriptors, tuple) or len(manifest_descriptors) != 3:
        raise TypeError(
            "task-graph trainer requires native renderer/tokenizer/decoding descriptors"
        )
    stores = {entry.environment.store.root.resolve() for entry in task_entries}
    if stores != {output}:
        raise ValueError("task-graph store must use the training output root")
    if len({entry.task_id for entry in task_entries}) != len(task_entries):
        raise ValueError("task-graph schedule task IDs must be unique")

    store = task_entries[0].environment.store
    model_ref = store.put_artifact({"model_id": settings.model_id, "revision": settings.revision})
    native_descriptor_backend = NativeGemmaSampleBackend(
        None,
        tokenizer,
        manifest_descriptors=manifest_descriptors,
        adapter_name=adapter_name,
        model_ref=model_ref,
    )
    tools = LocalTextToolProvider()
    evaluator = DeterministicEvaluator()
    from writing_agent.task_graph_local import LocalWorkspaceEnvironment
    from writing_agent.task_graph_ports import RuntimeDependenciesV1

    placeholder_dependencies = RuntimeDependenciesV1(
        native_descriptor_backend,
        LocalWorkspaceEnvironment(tools),
        tools,
        evaluator,
    )
    runtime_manifest = placeholder_dependencies.manifest()
    if not isinstance(runtime_manifest, RuntimeManifestV2):
        raise ValueError("task-graph native sampling requires RuntimeManifestV2")
    if runtime_manifest.tokenizer.model_id != settings.model_id or (
        runtime_manifest.tokenizer.revision != settings.revision
    ):
        raise ValueError("native tokenizer pin differs from GRPO settings")
    if runtime_manifest.decoding.max_tokens_per_decision != settings.max_tokens:
        raise ValueError("native per-decision cap differs from GRPO settings")
    for entry in task_entries:
        runtime_handle = entry.environment.open(entry.entry_checkpoint_id)
        view = entry.environment.verify(runtime_handle)
        budget = view.budget["limits"]
        if (
            budget.get("context_tokens") != settings.context_tokens
            or budget.get("generated_tokens") != settings.max_generated_tokens
        ):
            raise ValueError("task-graph budgets differ from the frozen trainer recipe")
        for field, expected in (
            ("template_ref", runtime_manifest.renderer.template_ref),
            ("tokenizer_ref", runtime_manifest.renderer.tokenizer_ref),
        ):
            if view.context.rendering.get(field) != expected:
                raise ValueError(f"task graph {field} differs from native runtime manifest")

    base_identity = base_tensor_identity(model)
    experiment = task_graph_experiment_manifest(
        task_entries,
        settings,
        runtime_manifest,
        base_identity=base_identity,
        implementation=runtime,
        runtime_identity=runtime_identity,
    )
    identity, manifest = _task_graph_identity(experiment)
    plan = {
        "settings": asdict(settings),
        "runtime": runtime,
        "experiment": experiment,
        "retention": {
            "trainer_checkpoints": settings.max_steps,
            "policy": "keep every task-graph checkpoint",
        },
    }
    if next(model.parameters()).device.type not in {"cpu", "cuda"}:
        raise ValueError("task-graph model must reside on CPU or CUDA")
    trainer_config_values = trainer_config(
        settings,
        output,
        use_cpu=next(model.parameters()).device.type == "cpu",
        bf16=next(model.parameters()).dtype == api.torch.bfloat16,
        implementation_config=runtime["config"],
    )
    observer = TaskGraphLossObserver(output)
    factory = partial(
        TaskGraphRollouts,
        tasks=[{"id": entry.task_id} for entry in task_entries],
        settings=settings,
        output=output,
        task_entries=task_entries,
        runtime_manifest=runtime_manifest,
        manifest_descriptors=manifest_descriptors,
        model_ref=model_ref,
        experiment_identity=identity,
        base_revision=settings.revision,
        adapter_name=adapter_name,
        tokenizer_root=tokenizer_root,
        sample_backend_factory=sample_backend_factory,
        observer=observer,
    )

    def run():
        return run_trainer(
            api=api,
            tasks=[{"id": entry.task_id} for entry in task_entries],
            output=output,
            settings=settings,
            plan=plan,
            identity=identity,
            manifest=manifest,
            model=model,
            tokenizer=tokenizer,
            lora_config=api.LoraConfig(
                task_type="CAUSAL_LM",
                r=settings.lora_rank,
                lora_alpha=2 * settings.lora_rank,
                lora_dropout=0.0,
                target_modules="all-linear",
                bias="none",
            ),
            trainer_config_values=trainer_config_values,
            make_rollouts=factory,
            resume_from_checkpoint=resume_from_checkpoint,
            resume_checkpoint_identity=resume_checkpoint_identity,
            stop_after_steps=stop_after_steps,
            trainer_callback_factory=trainer_callback_factory,
        )

    result = observer.run(run)
    summary = observer.verify(cpu_gate=next(model.parameters()).device.type == "cpu")
    result["task_graph_observer"] = summary
    complete = Path(result["adapter"]).parent / "complete.json"
    if complete.exists():
        body = json.loads(complete.read_text())
        body["task_graph_observer"] = summary
        save_json(complete, body)
    return result


__all__ = [
    "TaskGraphGroupPending",
    "TaskGraphLossObserver",
    "TaskGraphResumeRefused",
    "TaskGraphRollouts",
    "TaskGraphTaskV1",
    "TaskGraphTrainingError",
    "task_graph_behavior_policy_ref",
    "task_graph_experiment_manifest",
    "task_graph_resume_preflight",
    "train_task_graph",
]
