# Training-distribution axes

Proposed design, 2026-09-21. Extends the content coverage sampled by
[task generation](rl-task-generation.md) into a distribution over tasks *and*
environments. This is a design, not an implemented configuration.

## Two kinds of axis

Each axis is either a **task variable** or a **nuisance variable**, and the reward
must treat them differently.

- A **task variable** changes what a good answer is. The model should adapt.
- A **nuisance variable** must not change what a good answer is. The model should be
  invariant, and the reward must be identical across its values.

If a nuisance variable moves the reward, the policy can learn the nuisance instead of
the task. If a task variable does not move the reward, the model is never taught to
adapt to it.

## The axes

| Axis | Varies | Kind | Evidence today |
|---|---|---|---|
| **A. Task content** | genre, style, trope, situation, continuity challenge, transformation | task | [variation catalog](../../data/training/variation-catalog-v1.json); `TRANSFORMATIONS` |
| **B. Starting point** | family F1–F5, initial files, stage depth | task | `FAMILIES`, `initial_files`, `stage_families` |
| **C. Instruction specificity** | which decision points the brief states, and how explicitly | task — primary | partial: "prompt specificity" is a listed coverage dimension |
| **D. Thinking level** | none / brief / extended reasoning before a turn | conditioning | none; DeepSeek exposes only thinking on/off |
| **E. Instruction phrasing** | meaning-preserving rewrites of the same brief | nuisance | none |
| **F. Tool envelope** | python five-tool harness vs pi native tools | nuisance | none; see [runtime boundary](#runtime-boundary-f) |
| **G. Partner identity** | behavior and phrasing of the [simulated author](simulated-author.md) | nuisance | none |

Axes A and B are already sampled by `_sample_options` in
[`task_generation.py`](../../src/writing_agent/task_generation.py) and summarized by
the coverage `Counter` in the same module. The rest are new.

## Reward rules

Three rules follow from the split.

1. **Invariance over nuisance.** The same task and the same outcome receive the same
   reward regardless of phrasing (E), envelope (F), or partner (G). This requires the
   reward to be computed from **outcomes and artifacts**, not from tool-call sequences.
   The current version-0 reward already does this: it reads delivered files, extracted
   prose, link results, and judge scores.
2. **Conditioning over task.** Axes A, B, C, and D change the applicable rubric. The
   task generator already emits per-task checks; the rubric set is what varies with the
   axis, not the reward formula.
3. **Group purity.** Each distinct prompt forms its own GRPO group, and all members of a
   group share the same axis values. Nuisance variation produces *different prompts for
   the same underlying task*; it does not put two envelopes or two partners in one group.

## Axis C is the primary case

The project goal is to preserve the author's decisions. The hardest case is when the
author has not yet decided. Underspecification is therefore not a side condition but the
primary training situation, and the target behavior is **to clarify before acting** when
a decision is consequential and undetermined.

### The degenerate policy

If the reward favors clarification directly, the cheapest winning policy is
**always-ask**: interrogate the user about everything. That is a worse product than a
model that writes, and it is reward hacking. The target is *discriminating*
clarification:

```
underspecified  ∧  consequential  ∧  not already in project files  →  ask
otherwise                                                          →  proceed or propose
```

`not already in project files` is mechanically checkable: asking a question whose answer
sits in the workspace is a failure, and it is exactly the "use the project files" goal.

### Clarification is rewarded as an outcome, not as an act

Score the final artifact against the author's **actual hidden preference** (the private
spec), not against whether the agent asked. Then guessing right is rewarded, guessing
wrong is penalized, and asking-and-converging is rewarded. Clarification earns its reward
*instrumentally*, which removes the incentive to ask without cause.

### The specified-task control

The pool must contain fully specified tasks where asking is a defect. Without that pole,
the policy drifts to always-ask and the "discriminating" behavior is never learned.

### Scorable signals

| Signal | How it is measured |
|---|---|
| Clarification targeting | did the question name a consequential, undetermined decision point? |
| No-known-info asks | count of questions answerable from the project files/KB (target 0) |
| Minimality | turns to convergence; absence of rhetorical or trivia questions |
| Non-ask on specified tasks | asks on a fully specified task are penalized |
| Outcome vs hidden preference | final artifact matches the author's real choice |
| Held-assumption transparency | when it proceeds anyway, does it state the assumption and keep it reversible? |

The last signal restates existing project canon: a proposal is not accepted canon, and an
unstated assumption is the failure that canon rules exist to prevent.

## Sampling, not crossing

The full Cartesian product of A×B×C×D×E×F×G is not tractable, and it is not what a first
run should attempt. Assign each request a **sampled tuple** and track *marginal* coverage
per axis, which is what the existing sampler and coverage `Counter` already do. Extend
that machinery rather than multiplying it.

## Staging

One 3090, a 2B base, and small batches cannot support exploring seven axes at once.
Varying everything produces a confounded result that cannot be attributed. Suggested
order:

1. **A + B** — already sampled.
2. **C** — cheap text variation, and the scientifically central axis.
3. **E** — cheap; pure robustness.
4. **G** — needs the [simulated-author](simulated-author.md) infrastructure.
5. **F** — needs a second runtime; see the boundary below.
6. **D** — depends on the base model and may not pay at 2B scale.

## Runtime boundary (F)

The repo harness in [`agent.py`](../../src/writing_agent/agent.py) exposes five
path-constrained file tools (`list_dir`, `read_file`, `search`, `write_file`,
`patch_file`) and no shell. pi exposes a wider native surface (`read`, `edit`, `write`,
`grep`, `find`, `ls`, optionally `bash`). The model **sees** the difference through the
tool schemas and through the observations the tools return; there is no hidden runtime
state. So the envelope is an *affordance* axis, not an implementation detail: adding
`bash` changes the task and moves the scoring boundary that the mechanical checks assume.

Two coherent options, to be chosen before this axis is used:

- **Shared tool contract, two executors.** pi is configured (via its SDK) with the same
  five tool names and semantics. The model sees no difference; the axis is pure
  infrastructure and should be treated as invariant.
- **Native surfaces, visible.** Each envelope keeps its own tools. Groups are naturally
  pure because the prompts differ, but the reward must stay outcome-based and the pool
  must not assume a capability the other envelope lacks.

The envelope axis is deferred until the task distribution itself works.

## Open questions

- Is thinking level (D) a conditioning axis, or a capability the policy should allocate
  itself by task difficulty? If the latter, it is not a fixed axis but a learned one.
- Do nuisance axes need explicit cross-group invariance tests, or is sampling many values
  sufficient to induce it?
- How is per-axis reward reported without turning curriculum selection into overfitting
  to the reported axis?

See [task generation](rl-task-generation.md), [multi-turn RL](multi-turn-rl.md), and the
[simulated author](simulated-author.md).
