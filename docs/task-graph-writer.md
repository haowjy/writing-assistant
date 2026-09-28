# Task-graph writer runtime

The task-graph runtime uses one verified core for writer turns, tool calls, scripted author
interactions, checks, context operations, and group members. `RolloutEnvironment` owns entry,
verified state, derivation, persistence, and publication; `RolloutDriver` sequences the
controller and typed gatherers. The ordinary evaluation CLI remains a separate path.

```text
RolloutEnvironment.enter / open_head
              │ verified RuntimeHandle
              ▼
RolloutDriver.run → RolloutEnvironment.step_input → typed port input
              │                         │ gatherer or environment step
              │ typed input record
              ▼
RolloutEnvironment.commit → one derive → one event + complete checkpoint
              │
              └──────────── TaskGraphStore.publish through LineageGate
```

A small caller can use the public result and resume contract like this:

```python
runtime = environment.enter(node_id, entry_params)
result = driver.run(runtime, max_steps=100)
runtime = result.runtime

# After a process failure, discard old handles and reload the published head.
runtime = environment.open_head(lineage_id)
```

`RunResult.directive` distinguishes `done` from `halt`; `max_steps` is only a driver guard,
not a task budget. `DriverBudgetError.runtime` is the latest committed handle. Before the
first commit publishes a head, rerun the idempotent `enter` or `open` the known entry
checkpoint. After a head exists, resume with `open_head`, not an in-memory handle.

## What a commit records

Every runtime commit contains exactly one event. Its payload is the typed input received by
the environment: a sampled `WriterTurnV1`, `ToolObservationV1`, `AuthorReplyV1`,
`EvaluatorResultV1`, context-operation input, member-start input, or controller-authored
environment step. The input's derive produces the event, the complete next state, and every
referenced artifact. Producer and gate call the same derive; replay folds the recorded inputs
from the admitted entry and requires the stored event and state to match. A store may accept
batches, but the runtime gate refuses a multi-event commit.

There is no second runtime log or generic patch reducer. The event ancestry is the recorded
input history; derived facts are not written a second time. The `OutcomeV1` referenced by
state holds the evolving check batch/results, transition edge, terminal status, reward, and
training-eligibility refs. Reward publication advances that record rather than creating a
parallel terminal/outcome ledger.

Context follows the same rule. `ContextContentV1` nodes form an immutable content chain; a
`ContextRevisionV1` names its content head and source event. Ordinary message appends are part
of the source event's derived state. Only explicit carry, seed, drop, and compact operations
have their own context-operation input and event. The context operation contract is in
[task-graph compaction](task-graph-compaction.md).

## Verified inputs and privacy

`step_input(runtime)` checks the published head and returns the verified `LineageView`, its
`next_step` directive, and the corresponding typed port input in one operation scope. The
driver gives only that allowlisted input—not a `LineageView` or store—to the matching
`SamplingRunner`, `ToolRunner`, `ScriptedAuthorSource`, or `CheckRunner`. When there is no
port input, it encodes `EnvironmentStepV1.of(directive)`. The driver commits exactly one
input before asking the controller again.

`port_input(view, directive)` is also available when a caller builds a port outside the
standard driver loop; it requires the gate's published view and its matching `next_step`.

A writer's tool input contains the current file map and one queued tool call. The rollout
`RuntimeHandle` owns no workspace and the environment does not materialize one. Callers that
explicitly need a directory can use the lower-level `TaskGraphStore.materialize` or verified
`restore(checkpoint_id, fresh_root)` helper; the driver never calls them. The text-tool
boundary is not an
OS sandbox; do not expose shell or arbitrary code execution to a writer.

## Publication and failure boundary

The store requires a `CommitVerifier` for both `publish` and `restore`; `LineageGate.view`
verifies restored checkpoints. Runtime lineages must pin `task-graph-derive-v1`. A store
without a verifier refuses publication/restore at `store.verifier`, and an unpinned runtime
lineage is refused at `state.versions_ref.transition_semantics`.

`commit` derives and persists input artifacts, publishes through the store (which invokes
the verifier), then returns the verified published handle. Only after publication does the
gate cache the view through `record_published(store, commit_id)`. Failures before head
publication leave the previous head authoritative; immutable orphan artifacts are safe.
Fresh adapter-contract violations raise `AdapterContractError`; forged or invalid recorded
inputs raise `ProjectionError`; corrupt stored bytes raise `CorruptRecordError`; a stale
head requires reopening through `open_head`.

For the scripted author contract, request/reply and outcome details see
[the scripted-author lifecycle](task-graph-scripted.md). For group member starts and
collection, see [deterministic groups](task-graph-groups.md).
