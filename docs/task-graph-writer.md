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

`WriterTurnV1` contains `action_id`, `context_revision_ref`, `raw_output_ref`, `usage`,
`adapter_trace`, and `message`. Optional context claims in `adapter_trace` are bound when
present; the context revision is the request identity, so there is no separate writer-request
record or writer-turn request reference. `EnvironmentStateV1.history` stores nonnegative
`action_count` and `tool_result_count`; tool-queue entries always include `rejection`.

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

A writer's tool input contains the current file map and one queued tool call. The rollout
`RuntimeHandle` owns no workspace and the environment does not materialize one. Callers that
explicitly need a directory must provide it through the execution port. The text-tool
boundary is not an OS sandbox; do not expose shell or arbitrary code execution to a writer.

## Publication and failure boundary

`TaskGraphStore` requires a `CommitVerifier` at construction and uses it for publication.
`RolloutEnvironment.open` and `open_head` verify saved checkpoints through `LineageGate.view`.
Runtime lineages must pin `task-graph-derive-v1`; an unpinned lineage is refused at
`state.versions_ref.transition_semantics`.

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

## Wire golden regeneration

To regenerate the explicit wire fixtures, run `uv run python -m tests.task_graph_golden_fixtures`.
Tests only compare the checked-in files; they never regenerate them. A wire change requires
an intentional `SEMANTICS_V1` bump and a matching digest update in the pairing test.
