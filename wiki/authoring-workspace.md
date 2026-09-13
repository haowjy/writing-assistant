# Authoring workspace

The workspace carries project knowledge beyond the conversation window: prior
prose, character and world notes, timelines, plans, and author decisions. The agent
must interpret their meaning across different file formats and layouts.

## State and decisions

A discussion can contain alternatives that the author has not selected. Notes can
also be stale or conflict with a later decision. The agent must distinguish those
cases before relying on a fact or updating project state.

Specify when prose becomes canon in each experiment. The agent should keep
unaccepted drafts separate from committed story facts.

The writing process separates what belongs explicitly on the page, what should
remain implicit, and what is deferred. These distinctions must survive retrieval,
planning, and prose generation.

## Authoring loop

The agent inspects the request, retrieves relevant state when needed, plans a local
passage, writes it, and observes the result before deciding what state to update.
How often to repeat that loop is an experimental variable: whole chapters, fixed
intervals, and semantic beats are candidates.

Planning, retrieval, writing, observation, and editing can share a model or use
specialists. The [training experiments](../work/research-plan/training-experiments.md)
compare these arrangements.

See [trajectory data](trajectory-data.md) for recording these interactions and
[evaluation](evaluation.md) for assessing their outcomes.
