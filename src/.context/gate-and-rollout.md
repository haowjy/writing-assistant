# Gate and rollout: how the new core runs a lineage

This file covers the code that verifies, steps and resumes a new-core lineage: `LineageGate`
and the store's verifier port, `RolloutEnvironment`, `RolloutDriver` and the gatherers.
The invariants (I1–I10), the layer order, wire records and derives are in
[transition-seam.md](transition-seam.md). How a group starts, collects and credits its
members is in [group-coordination.md](group-coordination.md).

Read this before writing code that calls the gate, the environment or the driver, and
before writing a rollout or acceptance test.

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
adds `record_published`. The store calls `view` during restore.

**One fold.** `view(store, checkpoint_id)` loads the checkpoint chain back to its root,
then walks forward from the newest cached view (or from the root). Each step derives the
recorded input through `derive_input`. `_step` checks `event.previous`, `event.seq`, and the
derived event and state identity against the stored ones (I1, I2). Runtime checkpoints may
not carry supplemental `artifact_refs`. A mismatch raises `ProjectionError` naming the first
differing event or state path. The pre-publication check (`verify_commit`) and cold restore
(`view`) run the same `_step`. `verify_commit` rejects a commit of more than one event (I10).

**Root fixed point and re-admission.** With no cached view, the gate re-admits the root's
graph under the admission policy that the root's `ExecutionVersionsV1` pins, derives the
entry from `params_of(root.state)`, and requires equal state identity (I3). A root with
`artifact_refs` is rejected. Admitted graphs are cached by
`(store root, instance_ref, admission_policy_ref)`. Admission is a pure function of
content-addressed inputs, so that key is sufficient. The store root is in the key because
admission also checks that artifacts exist in that store. A lineage can be verified only
under the policy its root pins, because the policy is part of the root's state identity.

**The store has one runtime path.** `publish` and `restore` require a configured semantic
verifier; a verifierless store refuses both. Runtime lineages must pin exactly
`task-graph-derive-v1`. An absent or unsupported pin is refused with
`ProjectionError("state.versions_ref.transition_semantics: runtime lineages require
task-graph-derive-v1")`. `save_checkpoint` remains a structural persistence operation;
the verifier runs when the checkpoint is published or restored. Runtime checkpoints may
not add `artifact_refs`.

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

**A derive owns every rule the gate enforces.** Producer-side code (`port_input`, the
gatherers, the driver) calls the exported owner and never re-implements the rule. A second
copy can drift from the derive, or classify the same failure differently.

| Rule | Owner | Producer-side use |
|---|---|---|
| Writer action ID | `task_graph_derive_writer.writer_action_id` | `port_input` fills `SamplerInput.action_id` |
| Group member lookup | `task_graph_derive_writer.group_member` | `port_input` fills the sealed `writer_seed` |
| Group sampling pins (seed, model, `policy_ref` and the sealed policy refs) | `bind_group_sampling_claims`, called by `derive_writer_turn` and wrapped as `AdapterContractProjectionError` | `port_input` puts the pins in `SamplerInput`, and `SamplingRunner` passes them to the backend; nothing producer-side checks them |
| Writer-turn context claims (`context_revision_ref`, `context_content_hash`, rendering) | `_decode_writer_turn_sampling` in `task_graph_sampling`, for every lineage | `port_input` sets the one `context_content_hash` from `view.context.content_ref`; nothing producer-side checks the claims |
| Whether a message part carries sampled content (MEDIUM-1) | `_build_assistant_message` in `derive_writer` writes the `no-sampled-content` sentinel | Group segment credit skips exactly that sentinel ([group-coordination.md](group-coordination.md)) |
| Pre-dispatch tool error | `task_graph_derive_writer.tool_dispatch_error` | `port_input` fills `ToolInput.dispatch_permitted` |
| Usage evidence under a token limit | `sampling_usage_requirements`, inside `derive_writer_turn` | None. `SamplerInput` carries no usage requirement |
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
unread `sampling_pins` key into the request artifact, which changed every group member's
`request_ref`. Both are gone. Group pin drift is `AdapterContractError` from `commit`
when fresh, and `ProjectionError` under the gate when persisted.

**The non-text-content asymmetry is intended.** Non-text assistant content from a sampler
is `AdapterContractError` in the gatherer. The same content in a `WriterTurnV1` returned by
the `alternatives` callback, or replayed from disk, never passes through the gatherer, so
`derive_writer_turn` rejects it as `ProjectionError`.

## Rollout environment

[`task_graph_environment.py`](../writing_agent/task_graph_environment.py) is the producer,
persistence and port-input boundary. Its public methods are `enter`, `open`, `open_head`,
`verify`, `step_input`, `port_input`, `commit` and `start_member`.

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
- `open(checkpoint_id)` is a cold restore: head and policy checks, a full gate fold, then a
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
- `port_input(view, directive)` is for callers that build a port outside the driver. It
  requires `view` to be the gate's view of the published head and `directive` to equal
  `next_step(view)`. It returns the typed port input, or `None` when the directive has no
  external port.
- `commit(runtime, input)` derives through `derive_input`, persists, calls `store.publish`
  (which re-derives under the lineage lock), then `gate.record_published`. The returned
  handle and directive come from the published view. A new commit costs two derives
  (producer and gate) with a warm cache. Retries follow the publication handoff above.
  `commit_observer`, if set, sees each `StepResult`.
- `start_member(entry_checkpoint_id, start)` opens the shared entry checkpoint and commits
  a `MemberStartV1`. This starts a new lineage whose first commit's parent is that entry.

`RuntimeHandle` holds `checkpoint_id`, `state` and the verified `context`. It has no
workspace (see "Design deviations" below).

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

## Driver and gatherers

[`task_graph_rollout.py`](../writing_agent/task_graph_rollout.py) holds `RolloutDriver`.
`run(runtime, max_steps=...)` loops:
1. call `step_input` for the verified view, directive and typed port input;
2. return on `done` or `halt`;
3. enforce `max_steps`, before any port runs, so `DriverBudgetError` writes nothing;
4. consult `alternatives` only when the directive offers alternatives;
5. dispatch on the port input's type to a gatherer, or encode `EnvironmentStepV1.of` when
   the port input is `None`;
6. `commit` exactly one event.

It returns `RunResult(runtime, directive)`. `directive.kind` is `done`, or `halt`, which
means the node is not runnable end to end (design §8.1). `max_steps` is an operational
guard; task budgets live in state. Adapter and caller callbacks run outside store scopes.

[`task_graph_gatherers.py`](../writing_agent/task_graph_gatherers.py) holds
`SamplingRunner`, `ToolRunner`, `ScriptedAuthorSource` and `CheckRunner`. Each takes only
its port-input type. `SamplingRunner` and `CheckRunner` get an `ArtifactSink`
(`put_artifact`, `put_bytes_artifact`), not the store. `CheckRunner` rejects a family
mismatch before it calls `evaluate`. `ToolRunner` raises `ExecutionInfrastructureError`
when the provider's infrastructure fails. Neither module may import
`task_graph_transition`, where `LineageView` lives (import test).

### Sampling requests

`SamplingRunner.turn` turns a `SamplerInput` into one typed `PreparedSamplingInput`, and
that is all `SampleBackend.sample` receives. It carries:
- the request and prepared-request refs, and the canonical messages, tools, rendering and
  request JSON;
- the context revision ref and one `context_content_hash`;
- for a group member, the sealed `writer_seed` and the model, behavior-policy, decoding,
  tokenizer and template refs. Outside a group these are `None`.

The rules:
- **There is one content hash: `view.context.content_ref`.** `port_input` sets it, and the
  derive binds a trace's `context_content_hash` claim to the same value. The gatherer never
  computes its own. It once hashed messages, tools and rendering with another formula. An
  adapter that echoed the hash it was given then failed every group sample, and no test
  noticed, because each test forged the claim from `content_ref`.
- **Decoding settings come only from `decoding_ref`.** The request JSON holds only the
  messages, and the new core has no per-call extras channel. The pinned decoding artifact is
  what the group seals and the derive binds; a second channel would let a member sample
  under settings that nobody sealed. The `request_extras` channel is absent;
  `SamplingRunner` is the only sampling path.
- **A backend reports what it was given.** `ScriptedSampleBackend` puts the supplied seed,
  the pinned refs and the context claims into its trace. When the pins did not reach the
  backend, a member was sampled with the backend's default seed, and an honest seed claim
  failed the group binding.
- **Context claims are bound for every lineage.** `decode_writer_turn_sampling` compares a
  trace's `context_revision_ref`, `context_content_hash` and rendering with the active
  context whenever each claim is present. This is O2's present-only rule applied to every
  lineage. When only the group binder checked these claims, a non-group commit with false
  context claims published. `bind_group_sampling_claims` compares only the sealed policy
  refs, `policy_ref`, seed and model. Requiring the claims to be present belongs to the
  HIGH-5 group-binding follow-up.
- **The member seal reads the verified view.** `RuntimeSession.require_member_seal` checks
  the lineage against `view.group` and the executing manifest against its `adapter_ref`. It
  never infers membership from the lineage name or reads `groups/` files.

## Resume and crashes

**Resume depends on whether the first head was published.** After a failure with a
published head, discard in-memory handles and call `open_head(lineage_id)`, from the same
process or a fresh store and gate. It follows the published head's commit to its checkpoint
and opens it with a full check. Before the first head, `open_head` raises
`ConcurrentUpdateError`; re-run the idempotent `enter(node_id, params)` or call `open` with
the known entry checkpoint ID. `DriverBudgetError.runtime` gives a resumable handle without
re-opening.

**The crash matrix** (`tests/test_task_graph_rollout.py`) crashes the driver inside
`commit` at each of the four `store.publish` fault stages (`before_immutable_writes`,
`after_immutable_writes`, `before_head_publication`, `after_head_publication`) and inside
`record_published`, on commits 1, 3, 5 and 11. Each case resumes through `open_head` on a new
`TaskGraphStore`, `LineageGate` and environment over the same store root, then runs to the
end. It proves three things:
- every case reaches the same final checkpoint and head as an uncrashed run, so nothing
  resumable lives only in memory;
- a crash after head publication, or in `record_published`, is recovered by the new gate's
  verified fold;
- commit 1 with a crash before head publication resumes by opening the known entry
  checkpoint; `open_head` is not used until a head exists.

A separate test fails after each of the 13 commits returns and resumes the same way.

**Offline replay** is tested inside `ports_disabled()`. That fixture context manager
patches the producer port classes (`ScriptedSampleBackend.sample`,
`LocalTextToolProvider.execute`, `DeterministicEvaluator.evaluate`,
`produce_evaluation_evidence`, `ScriptedAuthorSource.reply`) to raise, and fails if any was
reached. S4 first counted calls through the fixture's wrapper instances, which was vacuous:
`open` and `gate.view` never receive those objects. A mutation that made `gate.view` call a
live evaluator 65 times still passed. Prove "no port calls" by disabling the classes, not by
counting wrappers.

## Rollout tests

Build rollout tests on
[`tests/task_graph_rollout_fixtures.py`](../../tests/task_graph_rollout_fixtures.py). Its
module docstring is the API: `build_rollout_fixture` modes, `run_slice(until=)` for
stopping at a directive, `make_gatherers` for swapping one port, `commit_observer`,
`ports_disabled()`, `session=` and the canaries. Extend it through those hooks, never by
overwriting `env.commit`. Changes to the fixture must be additive and generic. A lane's own
fixtures (groups, forgeries) go in its own test module.

## Acceptance suite

The `tests/test_task_graph_accept_*` modules port the old runtime's forgery and boundary
tests to the new core (design §12). Their index is
[`tests/task_graph_acceptance_map.md`](../../tests/task_graph_acceptance_map.md): one row
per old test or scenario, naming the new test and the mechanism that rejects it.

**The mechanism rule.** An attack goes through a public seam: `RolloutEnvironment.commit`,
`TaskGraphStore.publish`, or an on-disk rewrite followed by `open_head` or `verify`. A test
may call `derive_input` to build an honest candidate before it forges it. It never calls a
derive or validator to simulate the rejection itself. Each row asserts the exact error
class and an unchanged head. A gate rejection also asserts the first differing path.

**Accepted limits are asserted, not marked as expected failures.** X3 is one: a trusted
tool adapter may attest a fabricated read observation while leaving the files unchanged,
because the tool effect contract does not recompute observations. Its test,
`test_X3_fabricated_read_observation_is_an_accepted_trusted_adapter_limit`, states that
boundary. The suite has no `expectedFailure`. A decorator treats "half fixed" and "not fixed"
alike and hides a regression of the fixed half; S5.1-B-CLASS-1 was masked this way.

**To add a row:**
1. Put the test in the `accept_*` module for its category.
2. Attack through one of the seams above, and assert the class, the path and the head.
3. Add the row to the map's section, with its design reference.
4. When a finding is a decided limit rather than a bug, name it in the map's "Findings /
   known boundaries" section and assert the limit.
5. Run a mutation of the guard the row claims. A row whose guard can be deleted with the
   suite still green is not coverage.

## Design deviations

The code differs from the transition-seam design package in these places. The package
has not been amended, so trust the code:
- **No workspace in the new core** (§7, §9). `enter` and `open` take no `workspace_root`
  and do not materialize, and `RuntimeHandle` has no `workspace`. Nothing in the driver,
  gatherers or fixture read it: gatherers use `ToolInput.files`. The S3 close review found
  the field was neither verified (a forged path committed and carried forward) nor
  refreshed after a file-changing commit. `TaskGraphStore.restore` verifies only; explicit
  materialization remains a separate store operation.
- **`CommitVerifier` exposes `view`**; there is no `verify_checkpoint` (§6.1).
- **The gate caches through `record_published`** (§6.1 step 4 has the environment insert
  the candidate); see "Publication handoff" above.
- **`run` returns `RunResult`**, not a handle, and `DriverBudgetError` carries the last
  handle (§8.2).
- **`BudgetContractV1.max_generated_tokens`** is new; see O1 in
  [transition-seam.md](transition-seam.md).
- **`WriterTurnV1.adapter_trace` accepts optional context claims** (wire-v1). The
  `context_revision_ref` claim is a typed `context_revision` edge. See "Sampling requests"
  for how they are bound.
- **The driver calls `step_input`**, not `verify` and then `port_input` (§8.2).
- **An `invalid_tool_call` part without sampled content carries a sentinel** as its `raw`,
  not a call value (§10.1).
- **Group start receipts are derived from verified ancestry, and collect does not re-check
  policy** (§12 row h); see [group-coordination.md](group-coordination.md).
- **Runtime session seals use the verified group view.** `RuntimeSession.require_member_seal`
  compares `view.group` and its adapter manifest pin; it does not infer group membership from
  the lineage name or read `groups/` files.
