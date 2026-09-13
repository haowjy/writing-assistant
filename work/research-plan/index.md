# Research work plan

These documents describe experiments to consider. Choose models, data sizes,
metrics, and execution order for the active research task.

The [custom evaluation suite](../custom-eval-suite/index.md) owns the current
five-family evaluation design, data inventory, and delivery order. Start with its
small development pilot before expanding the branching dataset or baseline runs.

- [Research proposal](research-proposal.md): research questions and candidate responsibilities.
- [Data experiments](data-experiments.md): construction methods, sample mixtures, and ablations.
- [Dataset and metric research](dataset-and-metric-research.md): existing sources, evaluation protocols, access limits, and a proposed shortlist.
- [Branching authorship](branching-authorship.md): Gutenberg checkpoints, alternate plots and styles, and recorded writing sessions.
- [Training experiments](training-experiments.md): SFT, specialization, preferences, RL, and diffusion.
- [Evaluation experiments](evaluation-experiments.md): metric candidates and failure diagnostics.
- [Execution checklist](execution-checklist.md): proposed end-to-end tasks; unchecked boxes are not a source-code audit.
- [Implementation milestones](implementation-milestones.md): scaffold status and suggested next experiments.

“DFT-like”, model judging (“JMQ”), and diversity `D(n)` need operational definitions.
The data-generation policy must also distinguish author plans from hindsight
based on the target passage.

Shared project intent lives in the [wiki](../../wiki/index.md). Implementation
contracts live beside the [source](../../src/.context/CONTEXT.md).
