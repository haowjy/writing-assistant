"""TRL configuration shared by legacy and task-graph GRPO profiles."""


def trainer_config(
    settings,
    output,
    *,
    use_cpu,
    bf16,
    implementation_config=None,
    report_to="none",
    run_name=None,
    scale_rewards="group",
):
    """Build public TRL args without importing its optional runtime."""
    if settings.runtime_profile == "task-graph-v1" and scale_rewards != "none":
        raise ValueError("Task-graph training requires scale_rewards='none'")
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
        scale_rewards=scale_rewards,
        loss_type=settings.loss_type,
        mask_truncated_completions=False,
        shuffle_dataset=False,
        report_to=report_to,
        run_name=run_name,
        logging_steps=1,
        save_steps=1,
        save_total_limit=(
            None if settings.runtime_profile in {"intact-full48-v1", "task-graph-v1"} else 2
        ),
        eval_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
