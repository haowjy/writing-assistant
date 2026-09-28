# Transition seam: new-core contracts

The task-graph runtime is being rebuilt as a new core beside the old one. The new core
stores each step's typed **input** record as the event payload. One pure **derive** per input
kind computes the event, the next state and the new artifacts, and the producer and replay
share it. The old runtime (writer, scripted runtime classes, checks, terminal,
author validation, projection, the old replay and environment batch) remains importable for
the behavior-oracle tests. Current callers use the new core; S7.3 removes the old runtime.
Do not extend it.

This checkout contains the wire records, calls, transition types, `derive_entry`, the
controller, four derive modules, `LineageGate`, `RolloutEnvironment`, the gatherers and
`RolloutDriver`. For a lineage pinned to `task-graph-derive-v1`, `store.publish` and
`store.restore` run the gate. The old role classes remain only for their behavior-oracle tests
until S7.3 removes the legacy runtime.

This file covers records, derives and the layer order. How the gate, the environment, the
driver and the gatherers run a lineage (error classes, publication, rule owners, resume)
is in [gate-and-rollout.md](gate-and-rollout.md).

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
| 6 | `task_graph_gatherers`, `task_graph_rollout` (typed port inputs only) |

- **`TYPE_CHECKING` imports count for layer order, not for cycles.** The layer test walks
  type-only imports; the SCC test ignores them. A type-only import is not a way around the
  layer order. Import the type from its lower home at runtime (for example, `GroupSpecV1`
  from `task_graph_record_contracts`, not from `group_contract`).
- No lazy imports to dodge a cycle. No `from` import of an underscore name across task-graph
  modules.
- A new seam module goes into both `NEW_SEAM_MODULES` (not in any runtime SCC) and
  `LAYER_RANKS`. No module in `NEW_SEAM_MODULES` may import `task_graph_checks`, which S7.3
  deletes. A module-name-pattern rule forbids `task_graph_gatherers` and
  `task_graph_rollout` (and any submodule of either) from importing
  `task_graph_transition`, where `LineageView` lives (I6).
- `derive_entry` may not import the store, ports, local, writer, environment, replay or any
  filesystem module. Derives in general read only the view, the input and the
  `ArtifactReader` (I7).

New-core modules still import pure helpers from two old-runtime modules:
- `derive_writer` uses `decode_and_bind_sampling`, `WriterTurnSamplingBindingV1` and
  `bind_group_sampling_claims` from `task_graph_sampling`, and `validate_ask_semantics` from
  `task_graph_scripted`;
- `derive_author` uses `resolve_script_reply`, `scripted_author_reply`,
  `ScriptCoverageError` and `validate_ask_semantics` from `task_graph_scripted`;
- `derive_outcome` uses `CURRENT_ELIGIBILITY` from `task_graph_sampling`;
- the gatherers use `ArtifactSink` and `persist_logprob_trace` from `task_graph_sampling`,
  and `scripted_author_reply` from `task_graph_scripted`.

These are policy and codec functions that S7.3 keeps when it strips the runtime classes
from those modules. Do not add imports of runtime classes. Add a helper there only when
design §11 names the pure half of that module as its S7 home. `scripted_author_reply` and
`persist_logprob_trace` were added on that basis.

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
  input the directive does not allow. An `EnvironmentStepV1` must equal
  `EnvironmentStepV1.of(directive)`, the one directive-to-wire mapping (kind, and `source`,
  `edge_id`, `task_status` or `stop_reason` where the kind has one). Do not hand-build the
  expected shape. Routing never reads model or author text.
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
  `task_graph_calls` into it, and so does the group-pin binding. The subclass lets
  `commit` report a fresh adapter fault as `AdapterContractError`, while replay still sees
  `ProjectionError` ([error classes](gate-and-rollout.md)).
  `derive_input` wraps any other exception a derive raises in `ProjectionError` and keeps
  the cause, except `CorruptRecordError`, which passes through: damaged bytes on disk are
  not a rejected input. A `MissingReferenceError` for a ref the input names becomes a
  `ProjectionError`.
- **Own the rule; export it for the producer.** A rule the gate enforces lives in its
  derive. When producer-side code needs the same rule (a port input field, a gatherer's
  reply), export the function from the derive layer and call it there. Never copy it into
  a gatherer. The current owners are listed in [gate-and-rollout.md](gate-and-rollout.md).
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

## Rationale and rejected alternatives

- **Rejected: derive re-checks of controller rules.** `derive_author_request` once matched
  the directive and then re-checked the feedback phase (through `task_graph_checks`),
  prerequisites and budgets. That gave three sites for the applicable-checks rule. Before
  the re-checks were deleted, forgery tests showed the directive alone rejects each case.
  The controller's rules match the deleted ones: the `tool_error` author budget,
  `_feedback_budgets_allow`, `ask_semantics` exactly when the node is `scripted_author`,
  and the contract's author-call limit, pinned in the entry budget. I4 is the single
  authority.

  That deletion depends on four named invariants, each owned and pinned at its rule site:
  - **Admission prerequisite scope:** feedback prerequisites name only `each_turn` checks or
    checks scoped to that feedback (`tests/test_task_graph_scripted.py`).
  - **Check-batch status reset:** `derive_check_request` resets the requested batch to
    unknown before results arrive (`tests/test_task_graph_derive_outcome.py`).
  - **Scripted-author budget pin:** `derive_entry` seeds `author_calls` only for
    `scripted_author` nodes (`tests/test_task_graph_derive_entry.py`).
  - **Check-phase entry:** only `request_checks` moves a checking view into
    `awaiting_checks`; check-result transitions stay within that active batch
    (`tests/test_task_graph_derive_outcome.py`).
- **O1: a writer turn without sampling evidence is charged 0 tokens.** Entries without
  generated- or total-token limits seed no corresponding counters. `derive_writer_turn`
  requires usage evidence under a token limit (missing usage is
  `AdapterContractProjectionError`) and binds usage to token IDs only when IDs are present,
  matching the old runtime.
  - **The contract can declare a token limit.** `BudgetContractV1.max_generated_tokens` is
    optional. It is listed in `_Contract.OMIT_NONE_FIELDS`, so it is left out of the
    canonical form when `None`, and every existing contract keeps its identity. This field
    departs from the design package. It exists so that the `token_limited` rollout fixture
    carries a real limit, which makes missing usage and `budget_charged` reachable.
  - **Still open:** admission does not yet require a usage-reporting adapter for a
    token-limited node. That rule is a named follow-up before Phase 8 native training, and
    the phase-end review judges the omit-when-`None` codec mechanism.
- **One validator for the sampled message, not a sentinel.** The records codec and
  `task_graph_calls` once accepted different values for the same message, so the store
  could persist a `WriterTurnV1` that `parse_calls` rejects. That is a producer/replay split
  on the same bytes. The strict canonical-value decoder lives in `task_graph_wire`.
  `intake_message` returns a `SampledMessageV1`, `parse_calls` only decodes that type, and
  the type carries the guarantee a `$sampled_message_v1` marker used to carry. Do not add
  a second validator for a shape that already has a codec. Intake tags a noncanonical value
  inside a sampled call and derives it as an invalid call with the uncreditable
  `$noncanonical` placeholder; a malformed sampling message or non-array `tool_calls` field
  is instead an adapter contract failure before a turn input or event is recorded.
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
