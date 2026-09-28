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
old-runtime modules, and keep its imports downward. Size budgets apply to total
`task_graph*` source, not to single files. Read
[.context/transition-seam.md](.context/transition-seam.md) before adding a record or derive.

See [.context/CONTEXT.md](.context/CONTEXT.md) for implementation contracts and limits.
