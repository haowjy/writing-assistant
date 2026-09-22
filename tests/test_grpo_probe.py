"""Runner identity, execution-gate, and process-containment contracts."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from writing_agent import grpo_probe as probe
from writing_agent.backends import Completion, ScriptedBackend
from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo_probe import (
    DEVELOPMENT_SEEDS,
    GPU_BUDGET_SECONDS,
    SETTINGS,
    evaluation_config,
    frozen_plan,
    inspect_probe,
    supervise,
)
from writing_agent.grpo_probe_data import (
    build_probe_release,
    golden_messages,
    load_probe_release,
)
from writing_agent.grpo_rollout import RolloutGroups
from writing_agent.inference import TransformersBackend
from writing_agent.reward import Reward
from writing_agent.suite import run_selected


class ProbeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)

    def test_frozen_settings_and_matched_evaluation_conditions(self):
        assert SETTINGS.group_size == 4
        assert SETTINGS.max_steps == 3
        assert SETTINGS.max_tokens == 768
        assert SETTINGS.max_generated_tokens == 1536
        assert SETTINGS.context_tokens == 4096
        assert SETTINGS.lora_rank == 8
        assert SETTINGS.learning_rate == 1e-5
        assert SETTINGS.enable_thinking
        assert SETTINGS.gradient_checkpointing
        assert not SETTINGS.gradient_checkpointing_use_reentrant

        adapter = self.root / "adapter"
        for seed in DEVELOPMENT_SEEDS:
            base = evaluation_config(seed)
            trained = evaluation_config(seed, adapter)
            assert {**trained, "adapter": None} == {**base, "adapter": None}
        assert (
            evaluation_config(DEVELOPMENT_SEEDS[0])["seed"]
            != evaluation_config(DEVELOPMENT_SEEDS[1])["seed"]
        )

    def test_inspection_is_read_only_and_import_safe(self):
        run = self.root / "absent"
        result = inspect_probe(run)
        assert result["status"] == "blocked"
        assert not run.exists()
        code = (
            "import sys; import writing_agent.grpo_probe; "
            "assert not {'torch','transformers','peft','trl'} & sys.modules.keys()"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_cli_requires_execute_without_creating_run(self):
        run = self.root / "planned"
        script = Path(__file__).resolve().parents[1] / "scripts/run_grpo_probe.py"
        process = subprocess.run(
            [sys.executable, str(script), str(run), "--phase", "prepare"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(process.stdout)
        assert result["status"] == "planned"
        assert not run.exists()

    def test_watchdog_kills_only_child_process_group_and_records_timeout(self):
        run = self.root / "timed"
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            supervise(
                run,
                "preflight",
                [sys.executable, "-c", "import time; time.sleep(60)"],
                timeout_seconds=0.05,
                grace_seconds=0.05,
            )
        ledger = json.loads((run / "resource-ledger.json").read_text())
        stage = ledger["stages"][0]
        assert stage["status"] == "timeout"
        assert stage["returncode"] is None
        assert stage["elapsed_seconds"] < 5
        assert ledger["gpu_budget_seconds"] == GPU_BUDGET_SECONDS

    def child_command(self, run):
        return [
            sys.executable,
            "-c",
            (
                "import time; from pathlib import Path; "
                "from writing_agent.grpo_probe import worker_record; "
                f"worker_record(Path({str(run)!r}), 'preflight', lambda: time.sleep(60))"
            ),
        ]

    def test_running_worker_cannot_override_timeout_and_no_retry(self):
        run = self.root / "timeout"
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            supervise(
                run, "preflight", self.child_command(run), timeout_seconds=0.5, grace_seconds=0.1
            )
        entry = probe._load_ledger(run)["stages"][0]
        assert entry["status"] == "timeout"
        assert entry["worker"]["status"] == "running"
        assert entry["process_peak_rss_bytes"] is None
        assert entry["whole_run_disk_growth_bytes"] >= 0
        with self.assertRaises(ProcessLookupError):
            os.kill(entry["child_pid"], 0)
        with self.assertRaisesRegex(RuntimeError, "already attempted"):
            supervise(run, "preflight", self.child_command(run))

    def test_parent_interrupt_reaps_actual_child_and_accounts_cost(self):
        run = self.root / "interrupt"
        original = subprocess.Popen.wait
        interrupted = False

        def wait(process, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt("parent interruption")
            return original(process, *args, **kwargs)

        with patch.object(subprocess.Popen, "wait", wait):
            with self.assertRaises(KeyboardInterrupt):
                supervise(run, "preflight", self.child_command(run), grace_seconds=0.1)
        entry = probe._load_ledger(run)["stages"][0]
        assert entry["status"] == "interrupted" and entry["elapsed_seconds"] > 0
        with self.assertRaises(ProcessLookupError):
            os.kill(entry["child_pid"], 0)

    def test_stale_running_gpu_stage_reserves_allowance_and_blocks(self):
        ledger = {"stages": [{"phase": "train", "status": "running", "timeout_seconds": 900}]}
        save_json(self.root / "resource-ledger.json", ledger)
        assert probe._gpu_seconds(ledger) == 900
        with patch.object(probe, "_gpu_preflight") as gpu:
            with self.assertRaisesRegex(RuntimeError, "Unreconciled"):
                supervise(self.root, "resume", [sys.executable, "-c", "pass"])
            gpu.assert_not_called()
        ledger["stages"][0].update(status="failed", elapsed_seconds=3600)
        save_json(self.root / "resource-ledger.json", ledger)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            supervise(self.root, "resume", [sys.executable, "-c", "pass"])

    def test_display_policy_accepts_bounded_known_consumers_only(self):
        allowed = probe._display_consumers(
            [
                "12, /usr/bin/cosmic-comp, 145",
                "13, /opt/chrome/chrome --type=gpu-process --enable-features=A,B, 36",
                "14, /usr/share/cursor/cursor --type=gpu-process, 64",
                "15, cosmic-app-library, 39",
                "16, /usr/libexec/xdg-desktop-portal-cosmic, 36",
                "17, cosmic-edit, 44",
                "18, cosmic-settings, 76",
                "19, cosmic-files, 76",
            ],
            23000,
        )
        assert len(allowed) == 8
        for rows, free in [
            (["14, /opt/llama-server, 21620"], 23000),
            (["15, /usr/bin/cursor, 300"], 23000),
            ([f"{i}, chrome, 200" for i in range(4)], 23000),
            ([], 21000),
        ]:
            with self.assertRaises(RuntimeError):
                probe._display_consumers(rows, free)

    def test_worker_initialization_failure_has_terminal_record(self):
        fake = SimpleNamespace(
            cuda=SimpleNamespace(
                reset_peak_memory_stats=lambda: (_ for _ in ()).throw(RuntimeError("CUDA init")),
                is_initialized=lambda: False,
            )
        )
        with patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaisesRegex(RuntimeError, "CUDA init"):
                probe.worker_record(self.root, "train", lambda: None)
        record = json.loads(next((self.root / "stages").glob("*.json")).read_text())
        assert record["status"] == "failed"
        assert record["torch_peak_allocated_bytes"] is None
        assert record["process_peak_rss_bytes"] > 0

    def test_all_twelve_evaluation_slots_are_scored_and_paired(self):
        release = build_probe_release(self.root / "release")
        root = self.root / "eval"
        task = release["development"][0]
        config = {**evaluation_config(DEVELOPMENT_SEEDS[0]), "kind": "scripted"}
        run_selected(
            [task],
            config,
            lambda: ScriptedBackend(golden_messages(task)),
            root / f"seed-{DEVELOPMENT_SEEDS[0]}",
            execute=True,
        )
        scores = probe.score_evaluation(release["development"], root)
        assert scores["planned_attempts"] == 12
        assert scores["completed"] == 1 and scores["unfinished"] == 11
        assert scores["unavailable_rewards"] == 11
        assert scores["attempts"][0]["reward"]["value"] == 1
        scores["run_identity"] = "same"
        paired = probe.pair_evaluations(scores, scores)
        assert paired["planned_pairs"] == 12
        assert paired["pairs"][0]["mechanical_delta"] == 0
        assert paired["pairs"][1]["mechanical_delta"] is None

    def test_plan_has_stable_paths_and_explicit_policy_limit(self):
        first = frozen_plan(self.root / "run")
        second = frozen_plan(self.root / "run")
        assert first == second
        assert first["identity"] == second["identity"]
        assert first["paths"]["training"].endswith("/run/training")
        assert "re-render" in first["evaluation"]["policy_limit"]
        assert "appends masked" in first["evaluation"]["policy_limit"]

    def test_training_reward_receives_real_trace_events(self):
        class Backend:
            failure = None

            def complete(self, messages, tools, *, emit):
                emit({"type": "model_input", "sentinel": "real-event"})
                emit({"type": "model_output", "output_ids": [2], "text": "done"})
                return Completion({"role": "assistant", "content": "done"})

            def evidence(self):
                return {
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "env_mask": [1],
                    "boundaries": [{"input_ids": [1], "completion_offset": 0, "output_ids": [2]}],
                }

        task = {
            "id": "trace-fixture",
            "role": "train",
            "source_groups": ["fixture"],
            "labels": {"checks": []},
            "visible": {
                "brief": "finish",
                "initial_files": {},
                "followups": [],
                "tools": [],
                "prose": [],
                "budgets": {
                    "max_steps": 1,
                    "max_tool_calls": 0,
                    "max_read_tokens": 0,
                    "max_total_bytes": 1024,
                },
            },
        }
        seen = []

        def reward(_task, result):
            seen.append(result["trace"])
            return Reward("ok", float(result["seed"] > 42))

        settings = SETTINGS.__class__(
            model_id="fixture/model",
            revision="a" * 40,
            context_tokens=16,
            max_tokens=2,
            max_generated_tokens=4,
            max_steps=1,
            group_size=2,
        )
        trainer = SimpleNamespace(
            model=None, processing_class=None, state=SimpleNamespace(global_step=0)
        )
        groups = RolloutGroups(
            [task], settings, self.root / "groups", reward, lambda *_args: Backend(), "system"
        )
        groups([task["id"], task["id"]], trainer)
        assert len(seen) == 2
        assert all(any(event.get("sentinel") == "real-event" for event in trace) for trace in seen)

    def test_evaluation_enforces_total_generated_token_budget(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"Optional CPU dependency unavailable: {exc}")

        class Inputs(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, messages, **_kwargs):
                return str(messages)

            def __call__(self, _text, **_kwargs):
                return Inputs(input_ids=torch.tensor([[3, 4]]))

            def decode(self, _output, **_kwargs):
                return "done"

            def parse_response(self, _text, *, prefix):
                return {"role": "assistant", "content": "done"}

        class Model(torch.nn.Module):
            device = torch.device("cpu")
            generation_config = SimpleNamespace(eos_token_id=1)

            def generate(self, input_ids, max_new_tokens, **_kwargs):
                generated = torch.tensor([[7, 1]]) if max_new_tokens > 1 else torch.tensor([[1]])
                return torch.cat([input_ids, generated], dim=-1)

        backend = TransformersBackend(
            Model(),
            Tokenizer(),
            {
                "protocol": "gemma-native-v1",
                "prompt_format": "chat",
                "max_tokens": 2,
                "max_generated_tokens": 3,
                "context_tokens": 32,
                "temperature": 0,
                "top_p": 1,
                "seed": 1,
            },
        )
        backend.complete([{"role": "user", "content": "one"}], [])
        backend.complete([{"role": "user", "content": "two"}], [])
        with self.assertRaisesRegex(ValueError, "Total generated-token budget exhausted"):
            backend.complete([{"role": "user", "content": "three"}], [])
        # Older evaluation callers declare only a per-call limit: preserve that
        # behavior instead of exhausting a new implicit trajectory budget.
        legacy_config = dict(backend.config)
        del legacy_config["max_generated_tokens"]
        legacy = TransformersBackend(Model(), Tokenizer(), legacy_config)
        for text in ("one", "two", "three"):
            legacy.complete([{"role": "user", "content": text}], [])
        assert legacy.generated_tokens == 6


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        probe.prepare_probe(self.root)
        preparation = probe._validate_preparation(self.root)
        release = load_probe_release(self.root / "release")
        # Synthetic tokenizer evidence isolates the file-binding tests; the final
        # real cached-tokenizer CLI run supplies independent runtime fit evidence.
        preflight = {
            "status": "completed",
            "preparation_identity": fingerprint(preparation),
            "cached_metadata_files": {},
            "tasks": [
                {"task_id": t["id"], "fits": True, "labels_excluded": True}
                for t in release["tasks"]
            ],
        }
        save_json(self.root / "preflight.json", preflight)
        binding = {"schema_version": 2, "preparation": preparation, "preflight": preflight}
        binding["identity"] = fingerprint(binding)
        save_json(self.root / "plan.json", binding)

    def test_positive_binding_and_model_admission_without_ml(self):
        assert inspect_probe(self.root)["status"] == "offline-ready"
        assert probe.admit_phase(self.root, "base-eval")["status"] == "admitted"
        with self.assertRaises(FileNotFoundError):
            probe.admit_phase(self.root, "train")

    def test_killed_evaluation_still_scores_all_planned_slots_offline(self):
        (self.root / "base-eval").mkdir()
        with patch.object(probe, "_gpu_preflight", return_value={"test": "no GPU query"}):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                supervise(
                    self.root,
                    "base-eval",
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    timeout_seconds=0.05,
                    grace_seconds=0.05,
                )
        scores = json.loads((self.root / "base-eval/scores.json").read_text())
        assert scores["planned_attempts"] == scores["unfinished"] == 12
        assert scores["unavailable_rewards"] == 12
        assert not (self.root / "base-eval/complete.json").exists()
        assert inspect_probe(self.root)["status"] == "blocked"

    def test_missing_or_changed_evidence_blocks_inspection_and_admission(self):
        for relative in (
            "prepare.json",
            "preflight.json",
            "fixture-evidence/evidence.json",
            "release/freeze.json",
            "plan.json",
        ):
            path = self.root / relative
            original = path.read_bytes()
            path.unlink()
            try:
                assert inspect_probe(self.root)["status"] == "blocked"
                with self.assertRaises((ValueError, FileNotFoundError)):
                    probe.admit_phase(self.root, "base-eval")
            finally:
                path.write_bytes(original)
        path = self.root / "preflight.json"
        value = json.loads(path.read_text())
        value["new-field"] = "altered-but-still-completed"
        save_json(path, value)
        assert inspect_probe(self.root)["status"] == "blocked"

    def test_internally_valid_replaced_release_and_source_changes_block(self):
        path = self.root / "release" / "freeze.json"
        original = path.read_bytes()
        value = json.loads(path.read_text())
        value["limitations"].append("replacement release")
        save_json(path, value)
        load_probe_release(self.root / "release")  # Still internally valid.
        with self.assertRaisesRegex(ValueError, "match release"):
            probe.admit_phase(self.root, "base-eval")
        path.write_bytes(original)
        assert inspect_probe(self.root)["status"] == "offline-ready"
        with patch.object(probe, "_code_hashes", return_value={"changed.py": "changed"}):
            assert inspect_probe(self.root)["status"] == "blocked"

    def test_changed_fixture_reward_and_missing_workspace_are_rejected(self):
        evidence = self.root / "fixture-evidence" / "evidence.json"
        value = json.loads(evidence.read_text())
        value["tasks"][0]["cases"][0]["reward"]["value"] = 0
        save_json(evidence, value)
        with self.assertRaisesRegex(ValueError, "reward/check"):
            probe.admit_phase(self.root, "base-eval")

    def test_foreign_adapter_is_not_admitted(self):
        adapter = self.root / "foreign"
        adapter.mkdir()
        from writing_agent.grpo import seal_directory

        seal_directory(adapter, "foreign", "inference-adapter")
        with self.assertRaises((ValueError, FileNotFoundError)):
            probe.admit_phase(self.root, "adapter-eval", adapter=adapter)

    def test_adapter_admission_requires_exact_resumed_export(self):
        expected = self.root / "training/invocations/resume/adapter"
        with (
            patch.object(probe, "_require_base"),
            patch.object(
                probe, "_completed_training", return_value={"adapter": str(expected)}
            ) as completed,
            patch.object(probe, "_verify_adapter", return_value={"identity": "expected"}),
        ):
            for other in (self.root / "foreign", self.root / "training/invocations/first/adapter"):
                with self.assertRaisesRegex(ValueError, "step-3 export"):
                    probe.admit_phase(self.root, "adapter-eval", adapter=other)
            probe.admit_phase(self.root, "adapter-eval", adapter=expected)
            completed.assert_called_with(self.root, 3, 1)

    def test_resume_uses_experiment_identity_and_initial_invocation_path(self):
        checkpoint = self.root / "training/checkpoint-1"
        save_json(self.root / "training/experiment.json", {"identity": "experiment-not-marker"})
        with (
            patch.object(probe, "_require_base"),
            patch.object(
                probe, "_completed_training", return_value={"checkpoint": str(checkpoint)}
            ) as completed,
            patch("writing_agent.grpo.verify_checkpoint", return_value=1) as verify,
        ):
            probe.admit_phase(self.root, "resume", resume=checkpoint)
            completed.assert_called_once_with(self.root, 1, 0)
            verify.assert_called_once_with(checkpoint, "experiment-not-marker")
            with self.assertRaisesRegex(ValueError, "initial completed invocation"):
                probe.admit_phase(self.root, "resume", resume=self.root / "training/checkpoint-2")

    def test_retention_comparison_never_reads_pruned_checkpoint(self):
        before = {"tensor_count": 2, "sha256": "step1", "lora_B_l1": 1.0}
        after = {"tensor_count": 2, "sha256": "step3", "lora_B_l1": 2.0}
        result = {
            "trainable_before": before,
            "trainable_after": after,
            "global_step": 3,
            "resumed_step": 1,
        }
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No checkpoint reads")):
            assert probe.training_change_evidence(result)["changed"]
        result["trainable_after"] = before
        with self.assertRaisesRegex(RuntimeError, "further change"):
            probe.training_change_evidence(result)


if __name__ == "__main__":
    unittest.main()
