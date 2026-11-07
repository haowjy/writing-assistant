"""Settings and public-TRL arguments for task-graph GRPO training."""

from dataclasses import dataclass
from pathlib import Path

from writing_agent.inference import checkpoint_identity


@dataclass(frozen=True)
class TaskGraphGRPOSettings:
    """The frozen DAPO trainer recipe composed with task-graph rollouts."""

    model_id: str = "google/gemma-4-E2B-it"
    revision: str = ""  # Caller must select an immutable checkpoint explicitly.
    context_tokens: int = 4096
    max_tokens: int = 256
    max_generated_tokens: int = 1024
    max_steps: int = 2
    group_size: int = 2
    microbatch_size: int | None = None
    learning_rate: float = 1e-5
    lora_rank: int = 8
    seed: int = 42
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    runtime_profile: str = "task-graph-v1"
    loss_type: str = "dapo"
    enable_thinking: bool = False

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.group_size // (self.microbatch_size or self.group_size)

    def validate(self) -> None:
        if self.runtime_profile != "task-graph-v1":
            raise ValueError("Only the task-graph training profile is supported")
        if self.loss_type != "dapo" or self.enable_thinking:
            raise ValueError("Task-graph training requires DAPO with thinking disabled")
        checkpoint_identity(self.model_id, self.revision)
        if not 2 <= self.group_size <= 8 or not 1 <= self.max_steps <= 20:
            raise ValueError("Task-graph training requires bounded groups and steps")
        if self.microbatch_size is not None and (
            type(self.microbatch_size) is not int
            or not 1 <= self.microbatch_size <= self.group_size
            or self.group_size % self.microbatch_size != 0
        ):
            raise ValueError("Training microbatch must divide the group size")
        if not 1 <= self.max_tokens <= self.max_generated_tokens < self.context_tokens <= 4096:
            raise ValueError("Task-graph token budgets must fit the 4,096-token ceiling")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must fit unsigned 32-bit range")
        if self.lora_rank < 1 or not 0 < self.learning_rate <= 0.01:
            raise ValueError("Invalid optimizer/LoRA settings")
        if not isinstance(self.gradient_checkpointing, bool) or not isinstance(
            self.gradient_checkpointing_use_reentrant, bool
        ):
            raise ValueError("Activation-checkpointing settings must be booleans")


def trainer_config(
    settings: TaskGraphGRPOSettings,
    output: Path | str,
    *,
    use_cpu: bool,
    bf16: bool,
    implementation_config: dict | None = None,
    report_to: str | list[str] = "none",
    run_name: str | None = None,
) -> dict:
    """Build public TRL args without importing its optional runtime."""
    if settings.loss_type != "dapo" or settings.runtime_profile != "task-graph-v1":
        raise ValueError("Only task-graph DAPO training is supported")
    return dict(
        **(implementation_config or {}),
        output_dir=str(output),
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.microbatch_size or settings.group_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        generation_batch_size=settings.group_size,
        num_generations=settings.group_size,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        seed=settings.seed,
        data_seed=settings.seed,
        use_cpu=use_cpu,
        bf16=bf16,
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": settings.gradient_checkpointing_use_reentrant
        },
        optim="adamw_torch",
        beta=0.0,
        num_iterations=1,
        disable_dropout=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_completion_length=settings.max_generated_tokens,
        scale_rewards="none",
        loss_type="dapo",
        mask_truncated_completions=False,
        shuffle_dataset=False,
        report_to=report_to,
        run_name=run_name,
        logging_steps=1,
        save_steps=1,
        save_total_limit=None,
        eval_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
