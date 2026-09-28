# Authoring research framework

This package supports experiments with conversational creative-writing agents:
project-file interaction, model execution, evaluation, and trajectory preparation.

Apply [dev-principles](../.codex/skills/dev-principles/SKILL.md) when changing code.

Development and experiments run remotely through a terminal. Keep core workflows
usable from Python research scripts and the existing CLI, with results inspectable
without a browser or notebook.

For model comparisons, hold tasks and budgets constant unless they are the variable
under study. Report agent behavior and prose quality separately.

The task-graph runtime is mid-rewrite. A new core (typed input records and one pure derive
per input kind) is being built beside the old runtime, which stays as the behavior oracle
until a single switch deletes it. Put new task-graph behavior in the new core, never in
old-runtime modules, and keep its imports downward. New-core code steps a lineage only
through `RolloutEnvironment`, and gives ports only the typed inputs it builds from
published views, never a `LineageView`. Each rule the gate enforces has one owner, in its
derive: producer-side code calls that exported owner and never copies the check. After a
failure, resume a published lineage only through `open_head`, never from an in-memory handle;
before the first head, rerun `enter` with the same parameters or open the known entry
checkpoint. View values are frozen all the way down: read them through `Mapping`, and
`thaw` them before mutating or serializing. A shallow `dict(...)` copy, or a `dict` type
check, silently mishandles the nested values. Size budgets apply to total `task_graph*`
source, not to single files. Read
[.context/transition-seam.md](.context/transition-seam.md) before adding a record or a
derive, [.context/gate-and-rollout.md](.context/gate-and-rollout.md) before writing code or
tests that call the gate, the environment or the driver, and
[.context/group-coordination.md](.context/group-coordination.md) before changing how a
group starts, collects or credits members.

See [.context/CONTEXT.md](.context/CONTEXT.md) for implementation contracts and limits.
