# Gate and rollout: how the new core runs a lineage

This file covers the gate and store-verifier contract, publication handoff, view cache, rule
owners and `RolloutEnvironment`.
The invariants (I1–I10), the layer order, wire records and derives are in
[transition-seam.md](transition-seam.md). How a group starts, collects and credits its
members is in [group-coordination.md](group-coordination.md).

Read this before changing verification, publication, cache or environment behavior. The
driver, gatherer, resume and runtime-test contract is in
[rollout-execution.md](rollout-execution.md).

## Error classes

Every new-core failure has one class, and each class tells the caller one thing to do.
Catch the exact class. `CorruptRecordError` and `ConcurrentUpdateError` are both
`StoreError`s, but they call for opposite responses, so never branch on `StoreError`.

| Class | Meaning | Raised for | Caller rule |
|---|---|---|---|
| `ProjectionError` | The candidate or recorded input is invalid: a forgery | A derive or fold mismatch; a closure failure on the candidate's own event, checkpoint or commit; a hash-correct payload that its codec rejects (closure errors report the codec's own field path, for example `artifact.forged_field`); a missing reference named by the input; a directive that differs from `next_step`; a failed derive rule that is not an adapter contract, even on a fresh commit (for example, a context operation from `alternatives` with a foreign `input.policy_ref`); a persisted adapter violation (`AdapterContractProjectionError` is a subclass); a lineage whose versions do not pin `task-graph-derive-v1`; `record_published` for a commit that is not the head | Stop. Do not retry the same input. |
| `CorruptRecordError` | Bytes already on disk fail their checks | A byte changed without its hash being recomputed, found on read on both the producer path (`commit`) and the gate path | Stop. The store is damaged; re-deriving will not help. |
| `ConcurrentUpdateError` | The handle or view is not the published head | A stale handle or view, a sibling, a never-persisted candidate, a stale retry with a different input, `enter` on a lineage that already has a head, `open_head` when no head has been published | Re-open with `open_head(lineage_id)` and continue from its `next_step`. Before the first head, re-run `enter` or `open` the entry checkpoint instead. |
| `WriterRuntimeError` | The caller has a bug | A handle whose state or context differs from its checkpoint; a checkpoint or view admitted under another graph or admission policy; a port view that differs from the gate's view | Fix the caller. |
| `AdapterContractError` | A fresh adapter response breaks its contract | A gatherer check: the sample message and tool-call-array envelope, the assistant role and content, the usage and trace shape, and logprob alignment; the tool response shape; the check contract, evaluator family and packet before `evaluate` runs, and the evidence type after. A session seal failure: the executing manifest differs from the sealed one (for example, it changed after bind) or from the adapter that `view.group` pins. Or `commit`, when a derive raises `AdapterContractProjectionError`: sampling evidence or usage, group pins, a tool effect, or evaluator evidence or status that fails verification | Fix the adapter. `commit` wrote nothing and the head did not move. The gatherer may have left orphan evidence artifacts, which are harmless. |
| `ExecutionInfrastructureError` | The tool provider's infrastructure failed; no contract was broken | `ToolRunner`, when the provider reports a status other than `ok` | Nothing was committed. Resume from `open_head`. A group records a member that cannot finish with `collect_invalid`, never as a reward. |
| `DriverBudgetError` | Operational: `max_steps` ran out before `done` or `halt` | `RolloutDriver.run` | Not a task outcome. `err.runtime` is the last committed handle; inspect it, or call `run` on it again. |

A fresh adapter violation and the same bytes replayed from disk get different classes on
purpose. Replay has no live adapter, so a persisted violation is a forged input.

**The class follows the rule that failed, not the author of the input.** `commit` cannot
tell a gatherer's input from one that the caller's `alternatives` callback returned. A
derive rule that checks an adapter contract raises `AdapterContractProjectionError`, and
`commit` reports it as `AdapterContractError`. Every other derive rule raises plain
`ProjectionError`, on a fresh commit as well as on replay. The context policy is a caller
rule, because `ContextOperationInputV1` comes only from `alternatives`. A grouped lineage's
foreign `input.policy_ref` is therefore a `ProjectionError` at that path, whether fresh or
replayed. A `WriterTurnV1` returned by `alternatives` still meets the writer derive's
adapter-contract rules, so its pin drift is an `AdapterContractError` on commit.

## Gate and store verifier port

[`task_graph_gate.py`](../writing_agent/task_graph_gate.py) is the one semantic verifier.
`LineageGate` implements the store's `CommitVerifier` port (`verify_commit`, `view`) and
adds `record_published`. `RolloutEnvironment.open` and `open_head` use the gate's verified
view for saved checkpoints.

**One fold.** `view(store, checkpoint_id)` loads the checkpoint chain back to its root,
then walks forward from the newest cached view (or from the root). Each step derives the
recorded input through `derive_input`. `_step` checks `event.previous`, `event.seq`, and the
derived event and state identity against the stored ones (I1, I2). Runtime checkpoints may
not carry supplemental `artifact_refs`. A mismatch raises `ProjectionError` naming the first
differing event or state path. The pre-publication check (`verify_commit`) and cold open
(`view`) run the same `_step`. `verify_commit` rejects a commit of more than one event (I10).

**Root fixed point and re-admission.** With no cached view, the gate re-admits the root's
graph under the admission policy that the root's `ExecutionVersionsV1` pins, derives the
entry from `params_of(root.state)`, and requires equal state identity (I3). A root with
`artifact_refs` is rejected. Admitted graphs are cached by
`(store root, instance_ref, admission_policy_ref)`. Admission is a pure function of
content-addressed inputs, so that key is sufficient. The store root is in the key because
admission also checks that artifacts exist in that store. A lineage can be verified only
under the policy its root pins, because the policy is part of the root's state identity.

**The store has one runtime path.** Construction requires a semantic verifier, and runtime
publication invokes it. Runtime lineages must pin exactly `task-graph-derive-v1`. An absent
or unsupported pin is refused with
`ProjectionError("state.versions_ref.transition_semantics: runtime lineages require
task-graph-derive-v1")`. `save_checkpoint` remains a structural persistence operation;
`RolloutEnvironment.open` verifies a saved checkpoint through the gate. Runtime checkpoints
may not add `artifact_refs`.

A first commit (`expected_head is None`) must start from a parentless checkpoint. A first
commit from a mid-lineage checkpoint would leave earlier events out of the commit chain.
Group members start from the shared entry checkpoint through `MemberStartV1`; there is no
mid-lineage branch operation.

## Publication handoff

The gate verifies a candidate inside `store.publish`, but it may cache a view only after
that candidate is the published head. This section is the contract between the two steps.

- **`verify_commit` stashes its candidate.** The stash is a locked, ordered map keyed by
  checkpoint ID and bounded to the view cache's capacity (64). Two different views for one
  checkpoint ID raise `ProjectionError`. The stash is shared by all threads.
- **`record_published(store, commit_id)` takes no view.** It loads the commit and its
  checkpoint and requires `commit_id` to be the lineage's published head (otherwise
  `ProjectionError`). It then pops the stashed candidate. If there is none (evicted, or
  verified by another gate or process), it runs a verified `view` fold instead, which costs
  one fold step from the cached parent. It inserts the result into the cache and returns
  it. `CorruptRecordError` passes through; any other `StoreError` while loading becomes a
  `ProjectionError`.
- **Honest retries and identical-commit races return the published head.** They never
  raise:
  - *Retry with a stale handle.* `commit` accepts a base that is no longer the head only
    when the current head is exactly the transition it just derived: the same checkpoint
    identity, the base as its sole parent, and the same single event. It then returns that
    head through `record_published`. Any other stale commit is `ConcurrentUpdateError`.
  - *Race, or a retry after a crash past head publication.* `store.publish` finds the
    identical commit already at the head and returns it through its idempotent branch,
    which skips `verify_commit`. `record_published` then has no candidate and falls back
    to the fold. `start_member` takes this path, because its base is the shared entry of
    another lineage.

**Rejected alternatives.**
- *A thread-local candidate (S3 fix step).* The error class depended on thread history. A
  retry on the thread that failed found its leftover candidate and succeeded. The same
  retry in a fresh process raised `ProjectionError` after the head had already moved to
  exactly the requested commit. An error class must not depend on which thread or process
  happens to retry.
- *A trusting `ViewCache.insert`.* The S3.1 review inserted a view whose decoded budget
  disagreed with its checkpoint, and a view for a checkpoint that was never persisted. A
  later `gate.view` returned the forged budget (`writer_turns: 1000000000`). Closure
  re-hashes stored bytes but never compares them with a cached decoded body. Insertion is
  therefore private (`ViewCache._insert`), and only the gate's own derivation is inserted.
- *`record_published` re-checking a caller's view.* Checking the decoded fields amounts to
  re-deriving. It becomes more brittle as the view grows (indexes, samples). The verified
  fold already does the same job honestly.
- *A `publish_verified` that returns the verifier's view.* This would add a second publish
  entry point. The idempotent branch still has no view to return, so the fallback fold
  would be needed anyway.

## View cache

`ViewCache` is a locked LRU of 64 immutable views keyed by
`(root_checkpoint_id, checkpoint_id)`.
- *Keyed by checkpoint, not by event.* A checkpoint ID hashes state, event head and
  parents, so a hit is a checkpoint the gate already derived. An event hashes only its
  payload, so two sibling checkpoints can share a head event with different states. An
  event-keyed cache accepted a forged sibling (same head event, lowered budget counter or
  forged `outcome_ref`) within one process.
- *Published commits only, and only through `record_published`.* A candidate cached before
  its publish fails would open the same hole as the event key.
- *Not a validation memo.* A `LineageView` holds decoded bodies (messages, budget,
  outcome). That is sound only because every operation's closure pass re-reads and
  re-hashes the target's ancestry from disk before the gate reads the cache. The cache may
  skip re-deriving history. It may never skip reading and hashing bytes.

The human rejected a per-session validation memo, because it hides tampering that happens
during a session: `store.operation()` reuses validated objects only within one thread-local,
re-entrant scope. Revisit only if measured Phase 9 rollouts are too slow.

## Rule owners

**A derive owns every rule the gate enforces.** Producer-side code (`step_input`, the
gatherers, the driver) calls the exported owner and never re-implements the rule. A second
copy can drift from the derive, or classify the same failure differently.

| Rule | Owner | Producer-side use |
|---|---|---|
| Writer action ID | `task_graph_derive_writer.writer_action_id` | `step_input` fills `SamplerInput.action_id` |
| Group member lookup | `task_graph_derive_writer.group_member` | `step_input` fills the sealed `writer_seed` |
| Group sampling claims (seed, model, `policy_ref` and sealed refs) | `bind_group_sampling_claims`, called by `derive_writer_turn`; V2 context claims are bound at top level | `step_input` puts the pins in `SamplerInput`, and `SamplingRunner` passes them to the backend; nothing producer-side checks them |
| Writer-turn context claims (`context_revision_ref`, `context_content_hash`, rendering) | `decode_and_bind_sampling` in `task_graph_sampling`, for every lineage and both turn versions | `step_input` sets the one `context_content_hash` from `view.context.content_ref`; the derive binds each present claim |
| V2 ledger, sampling pins, chained token prefix and derived termination | `decode_and_bind_sampling` in `task_graph_sampling`; `task_graph_token_ledger` owns token bytes | `step_input` exposes only `NativeSamplingBudget` allocations for native groups |
| V2 manifest policy and rendering pins | `require_native_manifest_binding` in `task_graph_native_contracts` | Both `RuntimeSession` and V2 derive binding call the same check |
| Whether a message part carries sampled content (MEDIUM-1) | `_build_assistant_message` in `derive_writer` writes the `no-sampled-content` sentinel | Group segment credit skips exactly that sentinel ([group-coordination.md](group-coordination.md)) |
| Pre-dispatch tool error | `task_graph_derive_writer.tool_dispatch_error` | `step_input` fills `ToolInput.dispatch_permitted` |
| Usage evidence under a token limit | `sampling_usage_requirements` in `task_graph_sampling`, called by both writer derives | None. `SamplerInput` carries no usage requirement |
| Scripted author reply (proposals and mandatory feedback) | `task_graph_scripted.scripted_author_reply` | `ScriptedAuthorSource.reply` returns it; `derive_author_reply` compares against it |
| Directive to environment step | `EnvironmentStepV1.of(directive)` | The driver encodes environment steps with it; derives compare `step == EnvironmentStepV1.of(next_step(view))` |

**Gatherers keep only what a derive cannot check.** These are the assistant role, the
`SampleResult` type, non-text assistant content, binary-logprob/token alignment before the
bytes are written (`persist_logprob_trace`), evaluator family before dispatch, and the tool
`dispatch_permitted` branch. Tool effects and
evaluation evidence are checked by the producer-path derives, which map adapter violations
to `AdapterContractError`; the gate sees them as `ProjectionError` on replay.

**Rejected: gatherer-side copies.** S4's `SamplingRunner._check_pins` checked 6 of the
derive's 11 group claim keys. Drift in `policy_ref`, `model` or `context_revision_ref` passed
the gatherer and then failed the unwrapped derive call with plain `ProjectionError`. A live
adapter that drifted therefore looked like a forgery. The same gatherer also wrote an
unread `sampling_pins` key into a copied request artifact, changing each group member's
sampling identity. The gatherer-side check and copied artifact are gone. Group pin drift is `AdapterContractError` from `commit`
when fresh, and `ProjectionError` under the gate when persisted.

**The non-text-content asymmetry is intended.** Non-text assistant content from a sampler
is `AdapterContractError` in the gatherer. The same content in a `WriterTurnV1` returned by
the `alternatives` callback, or replayed from disk, never passes through the gatherer, so
`derive_writer_turn` rejects it as `ProjectionError`.

## Rollout environment

[`task_graph_environment.py`](../writing_agent/task_graph_environment.py) is the producer,
persistence and port-input boundary. Its public methods are `enter`, `open`, `open_head`,
`verify`, `step_input`, `commit` and `start_member`.

**One store scope per public method.** Each public method is `@operation_scoped`: one outer
`store.operation()` scope per step, closed when the method returns. Private helpers
(`_open`, `_verify`, `_commit`) never open their own scope, so `start_member` (open, then
commit) is still one scope. A caller must not hold a scope across several steps, because one
validator session across several publishes hides tampering between them. It must not yield
from inside one either: a generator would keep the thread-local scope open across unrelated
work.

**Construction pins the operator's policy.** The constructor takes the store, the admitted
graph, the session seals, the gate and an `AdmissionPolicy`. It requires
`store.verifier is gate` and `graph.policy == admission_policy`. The gate alone proves only
that a lineage was admitted under the policy it pins. Pinning the policy on the environment
is what makes it the operator's policy.

**Head checks.** Every path that takes a checkpoint goes through `_inspect_checkpoint`.
It loads the checkpoint, requires the environment's graph instance and pinned policy
(`WriterRuntimeError` on a mismatch), and reports whether the checkpoint is the published
head. A parentless checkpoint counts as the head while its lineage has no head yet.
`_published_checkpoint` adds the requirement to be the head (`ConcurrentUpdateError`). Only
`commit` inspects a non-head base, and only to recognize an exact retry.

**The methods.**
- `enter(node_id, params)` refuses entry parameters whose `versions_ref` pins another
  policy, and a lineage that already has a head. It runs `derive_entry`, persists the
  instance and entry artifacts, and saves the parentless root checkpoint. Nothing is
  published until the first `commit`.
- `open(checkpoint_id)` is a cold open: head and policy checks, a full gate fold, then a
  handle built from the verified state and context.
- `open_head(lineage_id)` is the resume path; see "Resume and crashes" below.
- `verify(runtime)` checks the head, then requires the handle's state to equal its
  checkpoint and its context to equal the verified view's context. It checks the session
  seals and returns `gate.view`, which costs no derive at a cached head.
- `step_input(runtime)` verifies once and returns `(view, directive, port)` in one operation
  scope. The driver uses it, so a step makes two environment calls (`step_input` and
  `commit`) and three `gate.view` lookups, including commit validation. Each lookup is a
  cache hit at a warm head. The S7.3 scaling check measures this path; do not add a
  verification call to it.
- `commit(runtime, input)` derives through `derive_input`, persists, calls `store.publish`
  (which re-derives under the lineage lock), then `gate.record_published`. The returned
  handle and directive come from the published view. A new commit costs two derives
  (producer and gate) with a warm cache. Retries follow the publication handoff above.
  `commit_observer`, if set, sees each `StepResult`.
- `start_member(entry_checkpoint_id, start)` opens the shared entry checkpoint and commits
  a `MemberStartV1`. This starts a new lineage whose first commit's parent is that entry.

Real-mode groups and all `training_mode="native"` groups require a bound session at member
start, open/resume and commit, regardless of runner mode. A native fixture cannot skip the
session/manifest re-check merely because it is not using the real runner.

`RuntimeHandle` holds `checkpoint_id`, `state` and the verified `context`; it has no
workspace.

**Persistence order** (design §7): the input payload artifact, then every other derived
artifact except context revisions, then the event, then the context revision (it names the
event), then `store.publish`, which writes the checkpoint, commit and head. There are no
`artifact_refs`: closure follows typed edges, and a missing artifact fails the publish.
`TaskGraphStore.persist_artifact` is the one place that dispatches on a derived artifact's
kind and value. A failure at any stage leaves the old head or the new one, never a partial
commit. Orphan immutables are harmless, and a retry of the same input yields the same
commit ID.

**Port inputs are the privacy boundary (I6).** A gatherer receives one of four frozen
types, never a `LineageView`:

| Directive | Port input | Contents |
|---|---|---|
| `sample_writer` | `SamplerInput` | visible messages, tools, rendering, context content hash and revision ref, action ID; for a group member, its sealed writer seed and the group's model, behavior-policy, tokenizer, template and decoding refs |
| `execute_tool` | `ToolInput` | files, the one queued call at the directive's index, the pinned tool spec, `dispatch_permitted` |
| `await_author_reply` | `AuthorInput` | `request_ref`, the private request body, the node's script, the decision and disclosure ledgers |
| `await_check_result` | `CheckInput` | `request_ref`, the private request body, the candidate files, the evaluator packet |

Other directives have no external port. To widen a port input, add the field to its type
and update the test that enumerates each type's field set. That test is the allowlist.

**Only published views reach ports.** The gate's candidate view inside `verify_commit` is
never returned to a caller before publication. A pipelined driver therefore cannot build a
request from a commit the store might still reject.
