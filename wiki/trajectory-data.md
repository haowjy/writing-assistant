# Trajectory data

Training examples need to capture authoring interactions: the conversation,
available project state, tool calls and observations, and the requested artifact.

The research includes continuation, brainstorming and selection, revision,
planning, critique, continuity repair, and state maintenance. Longer examples
should contain changing decisions rather than padded history.

## Supervision and provenance

Human-written passages are preferred prose targets where appropriate material is
available. Synthetic supervision can supply project files, instructions, local
plans, tool interactions, and editing corruptions. Reconstructed plans are training
scaffolds, not evidence of the original author's thinking.

Record provenance and keep related works and authors separated between training
and evaluation. Check that instructions, tool observations, targets, and state
updates agree. Exclude target content that would have been unavailable when the
instruction was given.

Representation variation should preserve meaning while changing file names,
formats, layouts, noise, or tool schemas. This tests whether the agent learns to
use project knowledge beyond one prescribed organization.

Dataset sizes, conversation-length mixtures, and ablations belong in the
[data experiments](../work/research-plan/data-experiments.md). The implemented record
format and export behavior are documented in the [data contract](../docs/data-contract.md).
