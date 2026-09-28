# Transition seam: new-core contracts

The task-graph runtime is being rebuilt as a new core beside the old one. The new core
stores each step's typed **input** record as the event payload. One pure **derive** per input
kind computes the event, the next state and the new artifacts, and the producer and replay
share it. The old runtime (writer, scripted runtime classes, checks, terminal,
author validation, projection, the old replay and environment batch) stays importable and
tested as the behavior oracle until the S7 parity check, which deletes it. Do not extend it.

This checkout contains the wire records, calls, transition types, `derive_entry`, the
controller, four derive modules, the `LineageGate` with its `DERIVE` registry and view cache,
the store's verifier port, and `RolloutEnvironment`. For a lineage pinned to
`task-graph-derive-v1`, `store.publish` and `store.restore` run the gate. The driver and
gatherers are not built yet (S4), and callers stay on the old runtime until S7.2.

The design numbers these invariants, and code, tests and reviews cite them:

| # | Invariant |
|---|---|
| I1 | A committed event equals the event `derive` produces from the view before it. |
| I2 | A checkpoint's state identity equals the state `derive` produces at its event head. |
| I3 | A lineage's root checkpoint is a fixed point of `derive_entry` for its admitted node. |
| I4 | An environment-authored payload equals `next_step(view)`, and an input's kind is one the directive accepts. |
| I5 | Every artifact a checkpoint reaches is reachable through typed edges, and runtime checkpoints have empty `artifact_refs`. |
| I6 | Nothing reaches a port except a typed port input built from a published, verified view. |
| I7 | A derive reads only the view, the input and hash-addressed artifacts. |
| I9 | An edge's kind fixes the visibility of its target, so a private artifact stored public fails closure. |
| I10 | A runtime commit holds exactly one event. |

## Layer order

Imports go downward only. `tests/test_task_graph_imports.py` enforces this:

| Rank | Modules |
|---|---|
| 0 | `task_graph_errors` (no imports at all), `task_graph_wire`, `task_graph_payloads`, `task_graph_record_contracts`, `task_graph_records`, `task_graph_operation`; all sit on `task_graph` (canonical JSON, core records), which the test does not rank |
| 1 | `task_graph_calls`, `task_graph_controller`; they also use the pure policy modules `accounting`, `contracts`, `admission` |
| 3 | `task_graph_transition` (shared types only), `task_graph_derive_common`, `task_graph_derive_entry`, `task_graph_derive_writer`, `_author`, `_outcome`, `_context` |
| 4 | `task_graph_gate` (derives, store reader, admission); `task_graph_group_contract` (imports store, projection and admission) |
| 5 | `task_graph_rollout_env` (producer, store, gate and port boundary) |

- **`TYPE_CHECKING` imports count for layer order, not for cycles.** The layer test walks
  type-only imports; the SCC test ignores them. A type-only import is not a way around the
  layer order. Import the type from its lower home at runtime (for example, `GroupSpecV1`
  from `task_graph_record_contracts`, not from `group_contract`).
- No lazy imports to dodge a cycle. No `from` import of an underscore name across task-graph
  modules.
- A new seam module goes into both `NEW_SEAM_MODULES` (not in any runtime SCC) and
  `LAYER_RANKS`. No module in `NEW_SEAM_MODULES` may import `task_graph_checks`, which S7.3
  deletes. A module-name-pattern rule forbids any `task_graph_gatherers` module from
  importing `task_graph_transition`, where `LineageView` lives (I6).
- `derive_entry` may not import the store, ports, local, writer, environment, replay or any
  filesystem module. Derives in general read only the view, the input and the
  `ArtifactReader` (I7).

The derive modules still import pure helpers from two old-runtime modules:
- `derive_writer` uses `decode_and_bind_sampling`, `WriterTurnSamplingBindingV1` and
  `bind_group_sampling_claims` from `task_graph_sampling`, and `validate_ask_semantics` from
  `task_graph_scripted`;
- `derive_author` uses `resolve_script_reply`, `ScriptCoverageError` and
  `validate_ask_semantics` from `task_graph_scripted`;
- `derive_outcome` uses `CURRENT_ELIGIBILITY` from `task_graph_sampling`.

These are policy and codec functions that S7.3 keeps when it strips the runtime classes
from those modules. Do not add imports of runtime classes, and do not add helpers there.

## Adding a wire record

Choose the module by concern:

| Module | Holds |
|---|---|
| `task_graph_records` | New-core input and state records (`WriterTurnV1`, `ToolObservationV1`, `AuthorReplyV1`, `EvaluatorResultV1`, `ContextOperationInputV1`, `EnvironmentStepV1`, `MemberStartV1`, `ExternalInputsV1`, `AdmissionPolicyV1`, `OutcomeV1`), the chained context records, the registries and context materialization |
| `task_graph_record_contracts` | Sealed contracts with binding rules: `GroupSpecV1`, `GroupMemberSpecV1`, `ContextPolicyV1`, `ExecutionVersionsV1`, `SEMANTICS_V1`, `GroupError`, `CompactionError` |
| `task_graph_payloads` | `PayloadCodec`s for shared payload shapes that both runtimes write and that have no Python class: the three ledgers, `AuthorRequestV1`, `CheckRequestV1`, `RewardV1`, `TrainingEligibilityV1`, `GroupMemberSeedsV1` |

The steps:

1. **Declare each field once.** Subclass `WireRecord` as a frozen dataclass. Every
   instance field is `Annotated[type, spec]` with exactly one spec from `task_graph_wire`
   (`Hash`, `Str`, `Int`, `Bool`, `Enum`, `ListOf`, `DictOf`, `Obj`/`obj`/`obj_opt`,
   `UnionOf`, `KindUnion`, `JsonValue`, `CanonicalIntake`, `MessageValue`, `RecordOf`).
   An unannotated field fails at import. Put field-local rules (ranges, enums, non-empty,
   optionality) in the spec.
2. **Hashes are edges or explicit non-edges.** A stored reference is `Hash("<edge kind>")`.
   The edge kinds are `artifact`, `private`, `bytes`, `artifact|bytes`, `checkpoint`,
   `event`, `context_revision` and `context_node`, and the kind fixes the visibility the
   closure checks (I9). An identity that is not a stored reference is `Hash(None)`. `REFS`
   and `FIELD_SPEC` are derived from the annotations; never write them by hand.
3. **Cross-field rules go in `check()`.** Raise the domain error (`GroupError`,
   `CompactionError`, `ValueError`). Every `check()` branch ships with a negative test
   that changes one field of a valid example and expects the error. Rules that moved
   without such tests were dropped silently in S1.
4. **Registration is automatic.** `__init_subclass__` registers the class under
   `RECORD_TYPE` (tagged payloads), `EDGE_TYPE` (untagged closure kinds, such as the
   context records) or the class name. `RECORD_TYPES` and `RECORD_EDGES` in
   `task_graph_records` are built from that registry and the payload codecs. A record in
   a new module must be imported where the registry is populated.
5. **Naming during coexistence.** A shape the old runtime also writes stays byte-identical,
   including its `schema` field. A changed shape gets a new name. No `record_type` names
   two shapes.
6. **The legacy set is not an escape hatch.** `LEGACY_PAYLOAD_RECORD_TYPES` lists names only
   the old runtime writes. The closure passes those payloads without decoding them or
   following their edges. A test keeps the set disjoint from the registered names. Anything
   the new core writes must be registered.
7. **Tests** in `tests/test_task_graph_records.py`:
   - add an example to `record_examples()` (or `shared_payload_examples()`), which the
     round-trip, exact-key and coercion tests require;
   - add its edges to `EXPECTED_REFS` and its identity hashes to
     `EXPECTED_NON_EDGE_HASHES`;
   - the example scan fails on any 64-hex value outside a declared `Hash` path;
   - pin literal bytes for any shape that must stay compatible, as
     `tests/fixtures/pre_s1_group_spec.json` does.

The store follows registered edges for every payload with a `record_type`. It fails closed
on an unregistered `record_type` and on a `private` edge that resolves to a public
artifact. `put_artifact` refuses record domains, and it decodes every registered,
non-legacy `record_type` payload in canonical wire form before writing: a malformed record
raises `ValueError` at write time and never reaches disk. The chained context uses
`context_node` in `context_content/` and `context_revision` in `context_revisions/`. It is
chosen only when the state's `ExecutionVersionsV1` pins `transition_semantics`. An absent
value means legacy `contexts/`, and an unknown value raises.

## Writing a derive

A derive is `derive_x(view: LineageView, input: InputRecord, reader: ArtifactReader) ->
Transition`. It is pure: it reads the view, the input, and hash-addressed artifacts
through the reader, never a port, clock, path or mutable object. The producer and replay
call the same function on the same recorded bytes. Anything nondeterministic is recorded
verbatim in the input, and everything else is derived and never recorded.

- **Directive gate (I4).** First compute `controller.next_step(view)` once and reject an
  input the directive does not allow. An `EnvironmentStepV1` must equal the directive's
  fields exactly (kind, and `edge_id`, `task_status` or `stop_reason` where the kind has
  one). Routing never reads model or author text.
- **Do not re-check what the directive decided.** Phase, applicable checks, check
  prerequisites and author/writer budgets are the controller's rules. A derive that
  re-implements one creates a second rule site that can drift from the controller's.
  `applicable_checks` lives in `task_graph_controller` and is exported for derives that
  need the check set itself. If a forgery passes `next_step`, fix the controller rule; do
  not add a re-check in the derive.
- **Build through `task_graph_derive_common`.** Do not hand-build these pieces:
  - `new_event` computes the header, `previous`, `seq` and `payload_ref`;
  - `next_state` applies the changes, advances history and computes the tree hash;
  - `append_context` builds content nodes and a revision, and charges storage;
  - `payload_artifact` makes identity-checked artifacts;
  - `advance` builds the checkpoint and the successor view;
  - `build_transition` runs the whole record-driven path.
- **Read history indexes from the view.** `view.raw_call_ids`, `call_sources`, `samples`
  and `ancestry` exist so a derive never re-reads earlier turn artifacts. Rebuilding them
  costs O(n) reads per step.
- **I5: runtime checkpoints carry no `artifact_refs`.** `advance` builds
  `CheckpointV1(parents, state, event_head)`, so the checkpoint ID matches what the store
  publishes. Every artifact must be reachable through typed edges from the state, the
  event payload or record `REFS`. A derive that computes its own checkpoint ID with refs
  diverges from the published ID.
- **Return every new artifact.** `Transition.artifacts` lists each artifact the new
  state or the input references, including the input payload itself, as a
  `DerivedArtifact` with
  - `kind`: `artifact`, `private`, `context_revision` or `context_node`;
  - a required `value_kind`: `record`, `canonical_json` or `bytes`.

  `__post_init__` checks that `ref` is the identity of the value. `value_kind` exists
  because canonical-JSON payload bytes and raw `payload:bytes` artifacts have the same
  Python type, and persistence cannot choose between `put_artifact` and
  `put_bytes_artifact` by guessing.
- **Errors.** A derive raises `ProjectionError` for any input it rejects: a wrong directive,
  bad binding or invalid evidence. A violation of an adapter contract found in the
  recorded input raises `AdapterContractProjectionError`, a `ProjectionError` subclass;
  `derive_writer_turn` and `derive_tool_result` convert `AdapterContractError` from
  `task_graph_calls` into it. The subclass lets the producer report the adapter's fault
  (see the rollout environment's error boundary below) while replay still sees
  `ProjectionError`.
  `derive_input` wraps any other exception a derive raises in `ProjectionError` and keeps
  the cause. Store errors (`StoreError` and its subclasses) come from persistence, never
  from a derive.
- **Export `DERIVES`.** Each module exports a `DERIVES` mapping. The key is the payload
  `record_type`, except for `EnvironmentStepV1`, whose key is
  `("EnvironmentStepV1", directive.kind)`. `request_author` belongs to `derive_author`.
  `request_checks`, `commit_transition`, `seal_outcome`, `stop_exhausted` and
  `publish_reward` belong to `derive_outcome`. The gate assembles them into `DERIVE` and
  raises at import on a duplicate key. `task_graph_gate.derive_input` is the only code that
  computes a key and dispatches; the producer and the fold both call it.
  `tests/test_task_graph_derive_registry.py` pins the full, collision-free key set. Update
  it when a key is added.
- **Prove the fixed point.** `tests/test_task_graph_derive_fold.py` round-trips each input
  through its codec and checks that `derive(view, decode(encode(input)))` equals
  `derive(view, input)` across a folded lifecycle. A new input kind joins that fold.

`derive_entry(graph, node_id, params, reader) -> EntryV1` is the exception. It derives a
node's root state, its artifacts and its root `LineageView` from the admitted contract and
pinned parameters. The gate and `RolloutEnvironment.enter` both take the root view from
it; do not build a root view by hand. `params_of(state, reader)` inverts it for the root
fixed-point check (I3). A mid-run state is never a fixed point.
`derive_entry.SYSTEM_PROMPT` is a copy of `agent.SYSTEM_PROMPT`, pinned by a test, so
`derive_entry` does not import `agent`.

The controller (`next_step`, `select_edge`, `applicable_checks`, `Directive`,
`evaluate_guard`) reads only structured view fields. `select_edge` raises on any tie in
precedence. `next_step` halts as `no_admitted_evaluation` in two cases:
- `checking` with nothing to check and no feedback left;
- `awaiting_checks` with no required `each_turn`/`node_exit_candidate` check.

An empty required set never counts as a pass.

## Gate and store verifier port

[`task_graph_gate.py`](../writing_agent/task_graph_gate.py) is the one semantic verifier.
`LineageGate` implements the store's `CommitVerifier` port (`verify_commit`,
`verify_checkpoint`) and adds `view` and `record_published`.

**One fold.** `view(store, checkpoint_id)` loads the checkpoint chain back to its root,
then walks forward from the newest cached view (or from the root). Each step checks
`event.previous` and `event.seq` against the parent, derives the recorded input through
`derive_input`, and requires the derived event and state identity to equal the stored ones
(I1, I2). A mismatch raises `ProjectionError` naming the first differing path. The
pre-publication check (`verify_commit`) and cold restore (`verify_checkpoint`) run the same
`_step`; `verify_commit` rejects a commit of more than one event (I10).

**Root fixed point and re-admission.** With no cached view, the gate re-admits the root's
graph under the admission policy that the root's `ExecutionVersionsV1` pins, derives the
entry from `params_of(root.state)`, and requires equal state identity (I3). A root with
`artifact_refs` is rejected. Admitted graphs are cached by
`(store root, instance_ref, admission_policy_ref)`. Admission is a pure function of
content-addressed inputs, so that key is sufficient. The store root is in the key because
admission also checks that artifacts exist in that store. A lineage can only be verified
under the policy its root pins; the policy is part of the root's state identity.

**The store chooses the path from pinned semantics, never from its configuration.**
`TaskGraphStore._verifier_for_checkpoint` reads the lineage's pinned
`transition_semantics`. `save_checkpoint`, `publish`, `restore` and the closure's commit
validation all use it:

| Pinned semantics | Path |
|---|---|
| `task-graph-derive-v1` | The verifier is required. A store without one raises `ProjectionError` on `save_checkpoint`, `publish` and `restore`. The patch reducer never runs. Runtime checkpoints may not add `artifact_refs`. |
| Absent (legacy, until S7) | The patch reducer, post-state check and legacy writer hook run whether or not a verifier is configured. The verifier is never called. |
| Any other value | Refused. |

Two more publication rules hold on both paths:
- A first commit (`expected_head is None`) must start from a parentless checkpoint. A first
  commit from a mid-lineage checkpoint would leave earlier events out of the commit chain.
- `store.branch` raises `ProjectionError` when a verifier is configured. A branched state
  keeps its parent's `lineage_id`, so the gate could never verify it. Group members start
  through `MemberStartV1` instead.

**Error classes.** Callers tell a rejected input from a damaged store by class:
- `ProjectionError`: the candidate or recorded input is invalid. This covers derive and
  fold mismatches, a closure failure on the candidate's own event, checkpoint or commit
  (the validator's `gated_candidates`), and a wrong-visibility reference in a candidate.
- `CorruptRecordError`: bytes already on disk fail their checks. Because `put_artifact`
  validates typed payloads at write time, this class now means the stored bytes changed.

**View cache.** `ViewCache` is a locked LRU of 64 immutable views keyed by
`(root_checkpoint_id, checkpoint_id)`:
- *Keyed by checkpoint, not by event.* A checkpoint ID hashes state, event head and parents,
  so a hit is a checkpoint the gate already derived. An event hashes only its payload, so
  two sibling checkpoints can share a head event with different states.
- *Published commits only, and only through the gate.* Insertion is private
  (`ViewCache._insert`). `verify_commit` stores its candidate view in a thread-local slot.
  `LineageGate.record_published(store, commit_id, view)` inserts that candidate only if the
  commit is the lineage's published head, its checkpoint is the candidate's, and the
  persisted state identity equals the candidate's. A caller cannot insert a view it built,
  so `verify_commit` and `record_published` for one commit must run on the same thread.
- *Not a validation memo.* The cache lets a fold skip re-deriving history. It never skips
  reading and hashing bytes: every operation's closure pass re-validates the target's
  ancestry from disk before the gate reads the cache.

## Rollout environment

[`task_graph_rollout_env.py`](../writing_agent/task_graph_rollout_env.py) is the producer,
persistence and port-input boundary. Its public methods are `enter`, `open`, `verify`,
`port_input`, `commit` and `start_member`.

**One store scope per public method.** Each public method is `@operation_scoped`: one
outer `store.operation()` scope per step, closed when the method returns. Private helpers
(`_open`, `_verify`, `_commit`) never open their own scope, so `start_member` (open, then
commit) is still one scope. A caller must not hold a scope across several steps (one
validator session across several publishes hides tampering between them), and must not
yield from inside one (a generator would keep the thread-local scope open across
unrelated work).

**Construction pins the operator's policy.** The constructor takes the store, the admitted
graph, the session seals, the gate and an `AdmissionPolicy`. It requires
`store.verifier is gate` and `graph.policy == admission_policy`. `enter` rejects entry
parameters whose `versions_ref` pins another policy, and `open` and `verify` reject
lineages pinned to another policy. The gate alone proves a lineage was admitted under the
policy it pins. Pinning the policy on the environment is what makes it the operator's
policy.

**The methods.**
- `enter` refuses a lineage that already has a head. It then runs `derive_entry`,
  persists the instance and entry artifacts, saves the parentless root checkpoint and
  materializes it from `entry.view`. Nothing is published until the first `commit`.
- `open` is a cold restore: stale-head and policy checks, a full gate fold, then
  materialization.
- `verify` checks the session seals, requires the handle to be the lineage head (or a
  parentless root with no head), checks policy, state, context and graph identity, then
  returns `gate.view`. A root with no head passes only through the gate's I3 check.
- `port_input(view, directive)` requires `view` to be the verified, published head (or the
  verified root) and `directive` to equal `next_step(view)`. It builds the one typed input
  for that directive.
- `commit(runtime, input)` runs `verify`, rejects a stale head with
  `ConcurrentUpdateError`, derives once through `derive_input`, persists, calls
  `store.publish` (which re-derives under the lineage lock), then
  `gate.record_published`. The returned handle and `StepResult.directive` come from the
  published view, not from a restore-after-publish pass. With a warm cache a commit costs
  two derives (producer and gate) and `verify` costs none.
- `start_member` opens the shared entry checkpoint and commits a `MemberStartV1`, which
  starts a new lineage whose first commit's parent is that entry.

**Persistence order** (design §7): the input payload artifact, then every other derived
artifact except context revisions, then the event, then the context revision (it names
the event), then `store.publish`, which writes the checkpoint, commit and head. There are
no `artifact_refs`; closure follows typed edges, and a missing artifact fails the publish.
A failure at any stage leaves the old head or the new one, never a partial commit. Orphan
immutables are harmless, and a retry of the same input yields the same commit ID. The
crash-matrix test covers 10 of the 11 event kinds; `budget_charged` is unreachable because
no admitted entry has token limits (see the O1 rationale below).

**Port inputs are the privacy boundary (I6).** A gatherer receives one of four frozen
types, never a `LineageView`:

| Directive | Port input | Contents |
|---|---|---|
| `sample_writer` | `SamplerInput` | visible messages, tools, rendering, context revision ref, action ID; for a group member, its sealed writer seed and the group's model, behavior-policy, tokenizer, template and decoding refs |
| `execute_tool` | `ToolInput` | files, the one queued call at the directive's index, the pinned tool spec |
| `await_author_reply` | `AuthorInput` | the private author request, the node's script, the decision and disclosure ledgers |
| `await_check_result` | `CheckInput` | the private check request, the candidate files, the evaluator packet |

Other directives have no external port. To widen a port input, add the field to its type
and update the test that enumerates each type's field set; the test is the allowlist.

**Only published views reach ports.** The gate's candidate view inside `verify_commit` is
never returned to a caller before publication. A pipelined driver therefore cannot build a
request from a commit the store might still reject.

**Error boundary.** On the producer path, `commit` maps `AdapterContractProjectionError` to
`AdapterContractError`: the live adapter broke its contract, and nothing is written. The
same bytes read back under the gate stay `ProjectionError`, because by then they are a
recorded input to reject, not an adapter to blame.

## Rationale and rejected alternatives

- **No validation memo across operations.** Every store operation re-reads and re-hashes
  the bytes it reaches. `store.operation()` reuses validated objects only within one
  thread-local, re-entrant scope. The human rejected a per-session memo because it hides
  tampering that happens during a session. Revisit only if measured Phase 9 rollouts are
  too slow.
- **Views are not a memo.** A `LineageView` holds decoded bodies (messages, budget,
  outcome). That is sound only because each operation runs the closure pass over the
  target's ancestry before it uses a view. A cache of views may skip re-deriving history.
  It may never skip reading and hashing bytes.
- **The view cache is keyed by checkpoint, not by event.** An event-keyed cache accepted
  a forged sibling checkpoint (same head event, lowered budget counter or forged
  `outcome_ref`) within one process. A candidate cached before its publish fails opens the
  same hole, so only published commits are inserted.
- **Rejected: a trusting `ViewCache.insert`.** The S3.1 review inserted a view whose
  decoded budget disagreed with its checkpoint, and a view for a checkpoint that was never
  persisted. A later `gate.view` returned the forged budget (`writer_turns: 1000000000`),
  because closure re-hashes stored bytes but never compares them with a cached decoded
  body. Only the gate's own candidate, matched to the published head, is inserted.
  `publish` keeps its return type; the candidate passes from `verify_commit` to
  `record_published` through the thread-local slot instead.
- **Rejected: choosing the path by verifier presence.** S3.1 first ran the gate whenever a
  verifier was configured and the legacy reducer otherwise. The review broke it both ways:
  a default `TaskGraphStore(root)` published a forged v1 lineage (terminal root, forged
  outcome, 10⁶ budget limits), and a verifier store accepted a legacy commit whose empty
  effect changed `outcome_ref`. Deferring the fix to S7 would have left the default
  constructor able to publish unverified v1 heads while S3.2 and S4 built on it. The
  lineage's pinned semantics decides instead, and S7.3 then deletes the legacy row as one
  branch.
- **Rejected: derive re-checks of controller rules.** `derive_author_request` once matched
  the directive and then re-checked the feedback phase (through `task_graph_checks`),
  prerequisites and budgets. That gave three sites for the applicable-checks rule. Before
  the re-checks were deleted, forgery tests showed the directive alone rejects each case.
  The controller's rules match the deleted ones: the `tool_error` author budget,
  `_feedback_budgets_allow`, `ask_semantics` exactly when the node is `scripted_author`,
  and the contract's author-call limit, pinned in the entry budget. I4 is the single
  authority.
- **O1: a writer turn without sampling evidence is charged 0 tokens.** No admitted entry
  has token limits: `BudgetContractV1` declares none, so neither runtime seeds
  `generated_tokens` or `total_tokens`. `derive_writer_turn` requires usage evidence only
  under a token limit and binds usage to token IDs only when IDs are present, the same rule
  as the old runtime. This is not a seam regression. Token-budgeted admission is a named
  follow-up before Phase 8 native training; it makes `budget_charged` reachable, and the
  crash matrix must then cover it.
- **One validator for the sampled message, not a sentinel.** The records codec and
  `task_graph_calls` once accepted different values for the same message, so the store
  could persist a `WriterTurnV1` that `parse_calls` rejects. That is a producer/replay split
  on the same bytes. The strict canonical-value decoder lives in `task_graph_wire`.
  `intake_message` returns a `SampledMessageV1`, `parse_calls` only decodes that type, and
  the type carries the guarantee a `$sampled_message_v1` marker used to carry. Do not add
  a second validator for a shape that already has a codec.
- **Group and context-policy records moved down into the records layer.** `GroupSpecV1`,
  `GroupMemberSpecV1` and `ContextPolicyV1` moved out of `group_contract` and
  `compaction`, and `ExecutionVersionsV1` gained a codec, all in
  `task_graph_record_contracts`. There are two reasons:
  - the closure must decode them and follow their hash fields (I5, I9);
  - `task_graph_transition` must import `GroupSpecV1` at runtime, and `group_contract` sits
    above the store.

  Their binding rules (group-ID hash, seed derivations, member ordinals, seed-policy
  checkpoint) moved with them into `check()`. The first move dropped `schema` from the
  group specs, which changed their identity and broke `resume` for pre-S1 groups. It also
  dropped the seed-policy checkpoint rule. Both were restored and pinned. Treat any change
  to an unchanged shape's bytes as a decision, not a side effect.
- **Limits apply to total `task_graph*` source, never to one file.** A per-file cap on the
  S2 derive consolidation was met by moving about 315 lines of derive logic into
  `task_graph_scripted` and `task_graph_sampling`, the old-runtime modules S7 strips. The
  lines were relocated, not removed. Before that, records declared every field twice (a
  class field plus a side table), which is why the spec now lives in the annotation.
