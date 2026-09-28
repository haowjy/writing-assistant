# Transition seam: new-core contracts

The task-graph runtime is being rebuilt as a new core beside the old one. The new core
stores each step's typed **input** record as the event payload. One pure **derive** per input
kind computes the event, the next state and the new artifacts, and the producer and replay
share it. The old runtime (writer, scripted runtime classes, checks, terminal,
author validation, projection, the old replay and environment batch) stays importable and
tested as the behavior oracle until the S7 parity check, which deletes it. Do not extend it.

This checkout contains the wire records, calls, transition types, `derive_entry`, the
controller and four derive modules. There is no gate, `DERIVE` registry, environment,
driver or gatherer yet. Nothing assembles or dispatches the derives outside tests, and
`store.publish` does not run them.

Code and tests cite these invariants by number:

| # | Invariant |
|---|---|
| I3 | A lineage's root checkpoint is a fixed point of `derive_entry` for its admitted node. |
| I4 | An environment-authored payload equals `next_step(view)`, and an input's kind is one the directive accepts. |
| I5 | Every artifact a checkpoint reaches is reachable through typed edges, and runtime checkpoints have empty `artifact_refs`. |
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
| 4 | `task_graph_group_contract` (imports store, projection and admission) |

- **`TYPE_CHECKING` imports count for layer order, not for cycles.** The layer test walks
  type-only imports; the SCC test ignores them. A type-only import is not a way around the
  layer order. Import the type from its lower home at runtime (for example, `GroupSpecV1`
  from `task_graph_record_contracts`, not from `group_contract`).
- No lazy imports to dodge a cycle. No `from` import of an underscore name across task-graph
  modules.
- A new seam module goes into both `NEW_SEAM_MODULES` (not in any runtime SCC) and
  `LAYER_RANKS`. `task_graph_derive_common` is currently in the first and missing from the
  second, so its rank is unchecked.
- `derive_entry` may not import the store, ports, local, writer, environment, replay or any
  filesystem module. Derives in general read only the view, the input and the
  `ArtifactReader` (I7).

The derive modules still import helpers from two old-runtime modules:
- `derive_writer` uses sampling decode/bind, call parsing and usage checks from
  `task_graph_sampling`, and `validate_ask_semantics` from `task_graph_scripted`;
- `derive_author` uses the script and ledger helpers in `task_graph_scripted`;
- `derive_outcome` uses `CURRENT_ELIGIBILITY` from `task_graph_sampling`.

These helpers must move into the new core before S7 strips those modules. Do not add
more.

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
artifact. `put_artifact` refuses record domains. The chained context uses `context_node`
in `context_content/` and `context_revision` in `context_revisions/`. It is chosen only when
the state's `ExecutionVersionsV1` pins `transition_semantics`. An absent value means legacy
`contexts/`, and an unknown value raises.

## Writing a derive

A derive is `derive_x(view: LineageView, input: InputRecord, reader: ArtifactReader) ->
Transition`. It is pure: it reads the view, the input, and hash-addressed artifacts
through the reader, never a port, clock, path or mutable object. The producer and replay
call the same function on the same recorded bytes. Anything nondeterministic is recorded
verbatim in the input, and everything else is derived and never recorded.

- **Directive gate (I4).** First compute `controller.next_step(view)` and reject an input the
  directive does not allow. An `EnvironmentStepV1` must equal the directive's fields
  exactly (kind, and `edge_id`, `task_status` or `stop_reason` where the kind has one).
  Routing never reads model or author text.
- **Build through `task_graph_derive_common`.** Do not hand-build these pieces:
  - `new_event` computes the header, `previous`, `seq` and `payload_ref`;
  - `next_state` applies the changes, advances history and computes the tree hash;
  - `append_context` builds content nodes and a revision, and charges storage;
  - `payload_artifact` makes identity-checked artifacts;
  - `advance` builds the checkpoint and the successor view;
  - `build_transition` runs the whole record-driven path.
- **I5: runtime checkpoints carry no `artifact_refs`.** `advance` builds
  `CheckpointV1(parents, state, event_head)`, so the checkpoint ID matches what the store
  publishes. Every artifact must be reachable through typed edges from the state, the
  event payload or record `REFS`. A derive that computes its own checkpoint ID with refs
  diverges from the published ID.
- **Return every new artifact.** `Transition.artifacts` lists each artifact the new
  state or the input references, as a `DerivedArtifact` with
  - `kind`: `artifact`, `private`, `context_revision` or `context_node`;
  - a required `value_kind`: `record`, `canonical_json` or `bytes`.

  `__post_init__` checks that `ref` is the identity of the value. `value_kind` exists
  because canonical-JSON payload bytes and raw `payload:bytes` artifacts have the same
  Python type, and persistence cannot choose between `put_artifact` and
  `put_bytes_artifact` by guessing.
- **Errors.** A derive raises `ProjectionError` for any input it rejects: a wrong directive,
  bad binding, invalid evidence, or an adapter-contract violation found in the recorded
  input. `derive_writer_turn` and `derive_tool_result` convert `AdapterContractError`
  from `task_graph_calls` into `ProjectionError`. `AdapterContractError` (infrastructure)
  is for the producer boundary, before anything is recorded. Store errors
  (`StoreError` and its subclasses) come from persistence, never from a derive.
- **Export `DERIVES`.** Each module exports a `DERIVES` mapping. The key is the payload
  `record_type`, except for `EnvironmentStepV1`, whose key is
  `("EnvironmentStepV1", directive.kind)`. `request_author` belongs to `derive_author`.
  `request_checks`, `commit_transition`, `seal_outcome`, `stop_exhausted` and
  `publish_reward` belong to `derive_outcome`. `tests/test_task_graph_derive_registry.py`
  pins the full, collision-free key set. Update it when a key is added.
- **Prove the fixed point.** `tests/test_task_graph_derive_fold.py` round-trips each input
  through its codec and checks that `derive(view, decode(encode(input)))` equals
  `derive(view, input)` across a folded lifecycle. A new input kind joins that fold.

`derive_entry(graph, node_id, params, reader) -> EntryV1` is the exception. It derives a
node's root state and artifacts from its admitted contract and pinned parameters.
`params_of(state, reader)` inverts it for the root fixed-point check (I3). A mid-run state
is never a fixed point. `derive_entry.SYSTEM_PROMPT` is a copy of `agent.SYSTEM_PROMPT`,
pinned by a test, so `derive_entry` does not import `agent`.

The controller (`next_step`, `select_edge`, `Directive`, `evaluate_guard`) reads only
structured view fields. `select_edge` raises on any tie in precedence. `next_step` halts as
`no_admitted_evaluation` in two cases:
- `checking` with nothing to check and no feedback left;
- `awaiting_checks` with no required `each_turn`/`node_exit_candidate` check.

An empty required set never counts as a pass.

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
- **Any view cache is keyed by checkpoint, not by event.** This constrains S3.1, which is
  not built here. An event hashes its payload, not the state after it. So two sibling
  checkpoints can share a head event and differ in state, for example a lowered budget
  counter or a forged `outcome_ref`. An event-keyed cache accepted exactly that sibling
  within one process. Insert a view only after `store.publish` returns, because a
  candidate cached before a failed publish is the same hole.
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
