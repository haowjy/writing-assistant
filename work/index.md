# Research work

- [TODO — now](TODO.md): near-term tasks for the current pilot.
- [FUTURE](FUTURE.md): deferred work, including the final data-generation coverage review.
- [Custom evaluation suite](custom-eval-suite/index.md): implementation, data, metrics, review materials, and [current coverage](custom-eval-suite/coverage.md).
- [Precision is not a detail](custom-eval-suite/precision.md): the nf4 baseline's unmeasured confound, and why the bf16 arm shows a *less* diverse model than the 4-bit one suggested.
- [Research plan](research-plan/index.md): broader experiments and source research.

Maintain task status in TODO and FUTURE. Keep designs, inventories, and verification
evidence in their work-item directories, linked from the task lists. Move an item
between lists when its priority changes; remove completed tasks once their outcome
is recorded in the relevant work item.

- [Creative Writing v3](creative-writing-v3/cost-plan.md): primary external prose benchmark, first-run scope and Anthropic cost estimate.
- [External checks](external-benchmarks/plan.md): automatic instruction-following/coding runs and paid-subset decisions.

- [Grok / OpenCode pilot](grok-pilot/plan.md): five-case comparison using native xAI tools and Astra ratings.

- [SFT preparation](sft/plan.md): native loss masks, training-data requirements, and the bounded QLoRA feasibility plan.
- [Training-distribution axes](sft/training-distribution-axes.md): task vs nuisance variables, reward invariance, and underspecification as the primary case.
- [Distribution fine-tuning](sft/distribution-finetuning.md): matching the human writing distribution rather than a good answer; the conditioning vector as the constraint channel, and the human-target construction we do not yet have.
- [Simulated author](sft/simulated-author.md): the controller and user model that run multi-turn sessions without a human.
- [Training data and task-generation research](sft/training-data-research.md): paper evidence,
  source-backed generation workflow, and the first 100 task assignments.
