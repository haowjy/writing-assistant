# Project goals

Build a conversational authoring agent that can collaborate with a writer over a
long-running creative project. The project may exceed the model's context window,
so the agent must work with an [external authoring workspace](authoring-workspace.md).

The agent should:

- Follow changing instructions and distinguish tentative ideas from committed decisions.
- Brainstorm alternatives, plan passages, and write only the requested amount.
- Keep information implicit or deferred when the author requires it.
- Revise locally while preserving unrelated text and story continuity.
- Retrieve relevant project facts and maintain state as decisions become committed.

Assess collaboration and writing separately under the [evaluation principles](evaluation.md).

## Research questions

The research asks how adaptation changes authoring behavior, whether external state
improves long-horizon coherence, and when planning between passages helps. It also
examines whether planning, writing, and editing benefit from separate models or
adapters, and how to improve creative diversity without losing quality or control.

Model size, training objectives, specialist architecture, and experiment order are
choices to investigate in the [research work plan](../work/research-plan/index.md).
