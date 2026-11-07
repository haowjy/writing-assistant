"""Read-only observer for the pinned TRL task-graph DAPO loss."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from writing_agent.catalog import save_json
from writing_agent.grpo_task_graph_errors import TaskGraphTrainingError


class TaskGraphLossObserver:
    """Read-only observer for actual pinned-TRL DAPO tensors and denominator."""

    def __init__(self, output: Path | str) -> None:
        from trl import GRPOTrainer

        self.output = Path(output)
        self.loss_code = GRPOTrainer._compute_loss.__code__
        self.expected: dict[int, list[dict[str, Any]]] = {}
        self.expected_denominators: dict[int, int] = {}
        self.records: list[dict[str, Any]] = []
        self._previous = None

    def register_batch(self, step: int, members: list[dict[str, Any]]) -> None:
        if step in self.expected:
            raise TaskGraphTrainingError("observer received duplicate task-graph step evidence")
        self.expected[step] = members
        self.expected_denominators[step] = sum(sum(member["env_mask"]) for member in members)

    def __call__(self, frame, event, result) -> None:
        if event != "return" or frame.f_code is not self.loss_code or result is None:
            return
        local = frame.f_locals
        trainer, inputs = local["self"], local["inputs"]
        self.records.append(
            {
                "step": trainer.state.global_step,
                "loss": float(result.detach()),
                "normalizer": float(local["normalizer"]),
                "loss_mask": local["mask"].detach().cpu().tolist(),
                "advantages": inputs["advantages"].detach().cpu().tolist(),
                "prompt_ids": inputs["prompt_ids"].detach().cpu().tolist(),
                "completion_ids": inputs["completion_ids"].detach().cpu().tolist(),
                "completion_mask": inputs["completion_mask"].detach().cpu().tolist(),
                "recomputed_logprobs": local["per_token_logps"].detach().cpu().tolist(),
            }
        )

    def run(self, operation: Callable[[], Any]) -> Any:
        previous = sys.getprofile()
        self._previous = previous
        try:
            sys.setprofile(self)
            return operation()
        finally:
            sys.setprofile(previous)
            self._write()

    def _write(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        by_step: dict[int, list[dict[str, Any]]] = {}
        for record in self.records:
            by_step.setdefault(record["step"], []).append(record)
        for step, values in by_step.items():
            save_json(
                self.output / "observer" / f"step-{step:06d}.json",
                {"schema": 1, "step": step, "microbatches": values},
            )

    def verify(self, *, cpu_gate: bool = True) -> dict[str, Any]:
        if not self.records:
            raise TaskGraphTrainingError("TRL loss observer saw no DAPO loss calls")
        records_by_step: dict[int, list[dict[str, Any]]] = {}
        tie_steps = {
            step
            for step, rows in self.expected.items()
            if rows and all(row["advantage_f64"] == 0.0 for row in rows)
        }
        max_drift = 0.0
        masks_match = advantages_match = denominator_match = tie_loss_zero = True
        for record in self.records:
            step = record["step"]
            rows = self.expected.get(step)
            if rows is None:
                raise TaskGraphTrainingError("TRL consumed a step without admitted batch evidence")
            completion_rows = record["completion_ids"]
            mask_rows = record["loss_mask"]
            advantage_rows = record["advantages"]
            recomputed_rows = record["recomputed_logprobs"]
            completion_masks = record["completion_mask"]
            if not (
                len(completion_rows)
                == len(mask_rows)
                == len(advantage_rows)
                == len(recomputed_rows)
                == len(completion_masks)
                == 1
            ):
                raise TaskGraphTrainingError("expected one-row task-graph microbatches")
            actual_completion = [
                token
                for token, valid in zip(completion_rows[0], completion_masks[0], strict=False)
                if valid
            ]
            match_index = next(
                (
                    index
                    for index, row in enumerate(rows)
                    if row["completion_ids"] == actual_completion
                ),
                None,
            )
            if match_index is None:
                raise TaskGraphTrainingError("TRL completion differs from admitted export")
            row = rows.pop(match_index)
            expected_mask = row["env_mask"]
            actual_mask = [
                value
                for value, valid in zip(mask_rows[0], completion_masks[0], strict=False)
                if valid
            ]
            masks_match = masks_match and actual_mask == expected_mask
            expected_advantage = row["advantage_f64"]
            advantage_row = advantage_rows[0]
            if not isinstance(advantage_row, list):
                advantage_row = [advantage_row]
            actual_advantage = float(advantage_row[0])
            advantages_match = (
                advantages_match and abs(actual_advantage - expected_advantage) <= 1e-7
            )
            recomputed = recomputed_rows[0]
            sampled = [value for value in row["sampled_logprobs"] if value is not None]
            recomputed_active = [
                float(value)
                for value, valid, active in zip(
                    recomputed, completion_masks[0], mask_rows[0], strict=False
                )
                if valid and active
            ]
            if len(sampled) != len(recomputed_active):
                raise TaskGraphTrainingError("sampled and recomputed logprob counts differ")
            max_drift = max(
                max_drift,
                max(
                    (
                        abs(left - right)
                        for left, right in zip(sampled, recomputed_active, strict=True)
                    ),
                    default=0.0,
                ),
            )
            denominator_match = (
                denominator_match and record["normalizer"] == self.expected_denominators[step]
            )
            if step in tie_steps:
                tie_loss_zero = tie_loss_zero and record["loss"] == 0.0
            records_by_step.setdefault(step, []).append(record)
        if any(rows for rows in self.expected.values()):
            raise TaskGraphTrainingError("admitted batch has members TRL did not consume")
        if not (masks_match and advantages_match and denominator_match and tie_loss_zero):
            raise TaskGraphTrainingError(
                "TRL loss inputs differ from export: "
                f"masks={masks_match}, advantages={advantages_match}, "
                f"denominator={denominator_match}, tie_loss={tie_loss_zero}"
            )
        if cpu_gate and max_drift > 1e-4:
            raise TaskGraphTrainingError(
                f"CPU sampled/recomputed logprob drift {max_drift} exceeds 1e-4"
            )
        return {
            "loss_calls": len(self.records),
            "steps": sorted(records_by_step),
            "masks_match_export": masks_match,
            "advantages_match_export": advantages_match,
            "dapodenominator_matches_active_tokens": denominator_match,
            "tie_loss_zero": tie_loss_zero,
            "max_sampled_recomputed_logprob_drift": max_drift,
        }
