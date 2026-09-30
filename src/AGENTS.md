# Authoring research framework

This package supports experiments with conversational creative-writing agents:
project-file interaction, model execution, evaluation, and trajectory preparation.

Apply [dev-principles](../.codex/skills/dev-principles/SKILL.md) when changing code.

Development and experiments run remotely through a terminal. Keep core workflows
usable from Python research scripts and the existing CLI, with results inspectable
without a browser or notebook.

For model comparisons, hold tasks and budgets constant unless they are the variable
under study. Report agent behavior and prose quality separately.

The task-graph runtime uses typed input records and one pure derive per input kind. Put
runtime behavior in the new core and keep imports downward. `TaskGraphStore` requires a
semantic verifier at construction and uses it for runtime publication; runtime lineages
must pin `task-graph-derive-v1`. Code steps a lineage only through `RolloutEnvironment`, and gives
ports only typed inputs built from published views, never a `LineageView`. Each rule the
gate enforces has one owner in its derive: producer-side code calls that exported owner
and never copies the check. After a failure, resume a published lineage only through
`open_head`, never from an in-memory handle; before the first head, rerun `enter` with the
same parameters or open the known entry checkpoint. View values are frozen all the way
down: read them through `Mapping`, and `thaw` them before mutating or serializing. A shallow
`dict(...)` copy, or a `dict` type check, silently mishandles nested values. Size budgets
apply to total `task_graph*` source, not to single files. Read
[.context/transition-seam.md](.context/transition-seam.md) before adding a record or a
derive, [.context/gate-and-rollout.md](.context/gate-and-rollout.md) before changing gate,
store-verifier or environment behavior, [.context/rollout-execution.md](.context/rollout-execution.md)
before changing driver, gatherer, native-sampler, resume or rollout-test behavior, and
[.context/group-coordination.md](.context/group-coordination.md) before changing how a
group starts, collects, credits or trains members.

Only an all-admitted `TrainingAdmissionV1` makes a native member trainable: structural
eligibility alone never reaches a trainer. Keep `torch`, `transformers` and trainer code
out of `task_graph*` modules; the tokenizer-backed audit and sampler live in `native_*`.

See [.context/CONTEXT.md](.context/CONTEXT.md) for implementation contracts and limits, and
[.context/TODO.md](.context/TODO.md) and [.context/FUTURE.md](.context/FUTURE.md) for
deferred work.
