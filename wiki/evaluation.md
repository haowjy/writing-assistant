# Evaluation principles

Evaluate the agent's collaboration and its writing separately.

## Agent behavior

Assess whether the agent selects the requested operation, follows constraints,
retrieves relevant facts, preserves decisions across turns, and updates project
state correctly. Revision tasks should assess the requested change and preservation
of text outside its scope.

## Writing artifacts

Assess coherence, characterization, pacing, dialogue, subtext, originality, and
redundancy in the resulting prose. Distributional similarity and model judgments
are candidate evidence; neither defines literary quality by itself.

## Comparisons

Keep tasks and generation conditions comparable, varying the factor under study.
Separate held-out evaluation from training, retain outputs and experiment settings,
and diagnose failures at retrieval, reasoning, realization, and state update.

The [evaluation experiments](../work/research-plan/evaluation-experiments.md) contain
candidate metrics, ablations, and diagnostic procedures. Their formulas, judge
rubrics, and decision thresholds must be specified before using them as evidence.
