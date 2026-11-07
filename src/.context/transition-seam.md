# Task-graph runtime: records and derives

Each task-graph commit contains one typed input event. One pure derive per input kind
computes the event, next state and new artifacts; the producer and verifier use the same
derive. Context content is stored as a hash chain with typed context revisions, and terminal
status, reward and eligibility are represented by the `OutcomeV1` record.

`TaskGraphStore` requires its semantic verifier at construction and uses it for publication.
Runtime lineages pin `task-graph-derive-v1`; missing or unsupported pins are refused. Runtime
stepping is owned by `RolloutEnvironment`, and `RolloutDriver` obtains each verified directive
and typed port input through `step_input`. The runtime does not materialize workspaces.

This file covers records, derives and layer order. Gate and environment rules are in
[gate-and-rollout.md](gate-and-rollout.md); the driver, gatherer, resume and acceptance-test
contracts are in [rollout-execution.md](rollout-execution.md). Groups are in
[group-coordination.md](group-coordination.md).

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
| 0 | `task_graph`, `task_graph_errors`, `task_graph_wire`, `task_graph_payloads`, `task_graph_record_contracts`, `task_graph_records`, `task_graph_group_records`, `task_graph_training_records`, `task_graph_native_contracts`, `task_graph_operation`, `task_graph_token_ledger` |
| 1 | `task_graph_accounting`, `task_graph_sampling`, `task_graph_scripted`, `task_graph_calls`, `task_graph_tool_outcomes`, `task_graph_compaction`, `task_graph_contracts`, `task_graph_admission`, `task_graph_evaluation`, `task_graph_controller`, `task_graph_artifacts`, `task_graph_context_roots`, `task_graph_group_index` |
| 2 | `task_graph_store`, `task_graph_closure` |
| 3 | `task_graph_transition`, `task_graph_derive_common`, `task_graph_derive_entry`, `task_graph_derive_writer`, `task_graph_derive_author`, `task_graph_derive_outcome`, `task_graph_derive_context`, `task_graph_eligibility` |
| 4 | `task_graph_gate`, `task_graph_group_contract` |
| 5 | `task_graph_environment`, `task_graph_training_layout` |
| 6 | `task_graph_group`, `task_graph_training_export`, `task_graph_ports`, `task_graph_local`, `task_graph_composition`, `task_graph_gatherers`, `task_graph_rollout`, `task_graph_probe_tasks` |

- **Every `task_graph*.py` file needs a rank.** The test globs the package, so a new core
  module without an entry in `LAYER_RANKS` fails. It also walks every import in the AST,
  including imports inside functions, and refuses any `task_graph*` import of the model stack
  (`torch`, `transformers`, `peft`, `trl`, …) or of `grpo*`, `native_*`, `inference` or
  `training_stages`. Tokenizer and trainer code stays on the adapter side.
- **`TYPE_CHECKING` imports count for layer order, not for cycles.** The layer test walks
  type-only imports; the SCC test ignores them. A type-only import is not a way around the
  layer order. Import the type from its lower home at runtime (for example, `GroupSpecV1`
  from `task_graph_record_contracts`, not from `group_contract`).
- No lazy imports to dodge a cycle. No `from` import of an underscore name across task-graph
  modules.
- A new seam module also goes into `NEW_SEAM_MODULES`. A module-name-pattern rule forbids
  `task_graph_gatherers` and `task_graph_rollout` (and any submodule of either) from
  importing `task_graph_transition`, where `LineageView` lives (I6).
- `derive_entry` may not import the store, ports, local, environment or any filesystem
  module. Derives in general read only the view, the input and the
  `ArtifactReader` (I7).

The core keeps pure helpers in the same concern modules:
- `derive_writer` uses `decode_and_bind_sampling` and `bind_group_sampling_claims` from
  `task_graph_sampling`, and `validate_ask_semantics` from `task_graph_scripted`;
- `derive_author` uses `resolve_script_reply`, `scripted_author_reply`,
  `ScriptCoverageError` and `validate_ask_semantics` from `task_graph_scripted`;
- `derive_outcome` uses `decide_eligibility` from `task_graph_eligibility`, which reads
  committed writer-turn and manifest evidence to persist the ordered structural reason;
- the gatherers use `ArtifactSink` and `persist_logprob_trace` from `task_graph_sampling`,
  and `scripted_author_reply` from `task_graph_scripted`.

These are the pure helpers that remain after obsolete runtime classes and unused sampling
codecs were removed. Do not add imports of runtime classes.

## Adding a wire record

Choose the module by concern:

| Module | Holds |
|---|---|
| `task_graph_records` | New-core input, outcome and context `WireRecord`s, including `ContextContentV1`, `ContextRevisionV1`, runtime manifest V1/V2 and descriptor records, `WriterTurnV1`/`WriterTurnV2`, and `TrainingAdmissionV1` (see [Native training records](#native-training-records)); registries and reference closure |
| `task_graph_group_records` | Pure group result and credit `WireRecord` classes (`GroupDecisionV1`, `GroupAdvantageV1`, `GroupSegmentCreditV1`, and related records) |
| `task_graph_training_records` | Codec-registered `TrainingBatchV1` record for a finalized group export |
| `task_graph` | Core environment records and context materialization |
| `task_graph_record_contracts` | Sealed contracts with binding rules: `GroupSpecV1`, `GroupMemberSpecV1`, `ContextPolicyV1`, `ExecutionVersionsV1`, `SEMANTICS_V1`, `GroupError`, `CompactionError` |
| `task_graph_payloads` | `PayloadCodec`s for shared payload shapes without a Python record class: ledgers, author/check requests, check evidence, reward/eligibility, and group seeds |

`WriterTurnV1` contains `action_id`, `context_revision_ref`, `raw_output_ref`,
`usage`, `adapter_trace`, and `message`; optional trace context claims are bound when present.
There is no separate writer-request record. `EnvironmentStateV1.history` stores nonnegative
`action_count` and `tool_result_count`; tool-queue entries require a `rejection` field.
`SampledMessageV1` keeps optional reasoning/thinking side channels for eligibility while
omitting them when absent, preserving prior wire identities. The native training records
are described in the next section.

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
5. **Preserve wire identity.** A retained shape stays byte-identical, including its
   `schema` field. A changed shape gets a new name. No `record_type` names two shapes.
6. **The registry is closed.** Every persisted typed payload is registered and decoded;
   anything the runtime writes must have a declared codec and reference edges.
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
artifact. `put_artifact` refuses record domains, and it decodes every registered
`record_type` payload in canonical wire form before writing: a malformed record raises
`ValueError` at write time and never reaches disk. Chained context uses `context_node` and
`context_revision` reference kinds in the shared closure; no version-dependent context
storage path remains.

## Native training records

Phase 8 adds record types rather than changing old ones. V1 records and their identities
are unchanged, and `transition_semantics` stays `task-graph-derive-v1`.

- **`RuntimeManifestV2`** declares the capabilities `usage_reporting`,
  `native_token_ledger` and `sampled_logprobs`, all required. It also carries
  `RendererDescriptorV1` (template, tokenizer and tool-schema refs, stop token IDs,
  `enable_thinking`, suffix-rules version), `TokenizerDescriptorV1` (model, revision, file
  hashes) and `DecodingDescriptorV1` (sampling settings, `max_tokens_per_decision`, seed rule,
  logprob convention). `task_graph_native_contracts` owns the one binding of manifest,
  sealed policy pins and context-root rendering pins. Seal and derive both call it.
- **`WriterTurnV2`** has the `WriterTurnV1` fields plus byte refs to the full input token IDs
  and the generated IDs (u32-le, codec in `task_graph_token_ledger`), their counts, f32
  logprobs, a `termination` claim, `sampling_pins`, and an optional `native_parse_failed`.
  That field's codec accepts only `True` or absent, never `False`. Masks are never stored:
  the export derives them.
- **One decoder.** `decode_and_bind_sampling` dispatches on the record type and refuses a
  V2 turn under a V1 manifest, and the reverse. For V2 it checks counts against usage, the
  logprob shape, the sealed pins, and exact chaining. On the same context root,
  `input_k` must begin with `input_{k-1} ++ generated_{k-1}`. Any intervening
  `context_changed` event, `carry` included, starts a new root (`task_graph_context_roots`).
- **Termination is derived, not trusted.** `_allowed_tokens` recomputes the cap from
  committed budgets and the pinned decoding descriptor. The cap is the minimum of
  per-decision, remaining generated and remaining context tokens, with ties going to
  decision, then generated budget, then context. `_validate_termination` then requires the
  adapter's claim to be the class the counts imply. An adapter that stops early and claims
  `token_limit` is rejected at the first differing path, so it cannot manufacture an
  `incomplete` outcome. A zero-generation `context_limit` turn may carry no content, calls
  or raw-output ref.
- **Writer failures end the lineage `incomplete`; they do not halt the group.**
  `termination_stop_reason` maps each validated turn to a stop reason, in this order:
  1. `token_limit` → `decision_token_limit`, `generated_tokens_budget` or
     `context_tokens_budget`, from its `limit`;
  2. `context_limit` → `context_tokens_budget`;
  3. `native_parse_failed` → `unparsed_tool_call`;
  4. calls without a `<|tool_response>` stop → `unterminated_tool_call`;
  5. a `<|tool_response>` stop without calls → `unterminated_final_answer`.

  Rule 5 fired on real weights in P1: an empty turn after a successful `write_file` ended
  the member `incomplete` with reward 0. The member stayed structurally eligible and
  trained as a negative example, as designed.

  A mapped turn commits straight to terminal state, using the overrun path's
  candidate-checkpoint shape. A parse failure under a token limit therefore reports the
  limit.
- **Parse failures are committed, not raised.** When the shared native parser reports
  malformed call text, the turn commits the decoded text as content, with no calls and
  `native_parse_failed = True`. The adapter makes this claim, and the tokenizer-backed audit
  re-derives it with the same `parse_native_response`. *Rejected:* sending the raw call to
  intake as `invalid_arguments_json` to keep the lineage alive. That needs an exact replay
  of a call that never parsed, and ending the lineage matches the legacy
  `candidate_invalid` rule.
- **Call IDs belong to the core.** `task_graph.tool_call_id(action_id, i)` returns
  `<lineage>:call:<ordinal>:<i>`. The core keeps a lineage-wide duplicate-raw-ID rule
  (`view.raw_call_ids`) that was written for API models, which emit unique IDs. Gemma's
  parser emits `call_0`, `call_1`, … afresh in each response, so every tool-calling turn
  after the first collided. `native_protocol.parse_native_response` therefore sets each
  parsed call's raw ID to its core ID (`bind_native_tool_call_ids`), and the sampler, the
  audit and the scripted CPU backend all parse through it. Do not relax the core duplicate
  check to fit a parser. Bind IDs at the adapter instead.
- **Structural eligibility.** `task_graph_eligibility.decide_eligibility(view, reader)` is
  pure, and `derive_reward` persists its first failing reason, checked in this order:
  `native_action_trace_unavailable`, `manifest_capability_missing`,
  `group_training_mode_absent`, `multi_segment_context`, `budget_overrun`,
  `reasoning_content_present`, `no_sampled_actions`, `execution_not_valid`. If none
  applies, the status is `structurally_eligible` with reason `native_evidence_structural`.
  No derive writes the reserved `eligible` status. Every scripted or V1 lineage records
  `native_action_trace_unavailable`. `budget_overrun` mirrors the export, so an
  eligible member never fails export in `finalize`.
- **Only an all-admitted `TrainingAdmissionV1` makes a member trainable.** Structural
  eligibility is what the pure core can prove. Admission is what the tokenizer-backed audit
  (`native_audit.audit_training_batch`) proves by re-deriving every token from the committed
  context. It records the result in the store, per member, as `admitted` or `refused` with
  its `failed_check`. `native_audit.require_training_admission` raises unless every member
  is `admitted`. `TaskGraphRollouts` calls it before any rows reach TRL, and the offline
  inspector requires the stored record. Any future consumer of `TrainingBatchV1`, such as
  an SFT export or another trainer, must do the same. Loading a batch proves nothing about
  admission. A refusal is infrastructure: the run halts, the record is kept, and nothing is
  relabeled.
- **Export is a projection, not a derive.** `task_graph_training_export` reads a settled
  native group's V2 chains and returns `TrainingBatchExportV1`: a `TrainingBatchV1` plus its
  u32 token, u8 mask and f64-le advantage artifacts (`advantage_f64_ref`). Canonical JSON
  refuses floats. The layout is owned by `task_graph_training_layout`, which segment credit
  and the audit also use. A trailing zero-generation `context_limit` turn is kept only as an
  audit ref and never enters the sequence or the mask. Refusals carry
  `TrainingExportError.reason_code`.

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
- **View values are frozen all the way down, so never copy one shallowly.** `LineageView`
  and its context pass their mappings through `freeze` (in `task_graph`), which turns every
  nested mapping into a `MappingProxyType` and every list into a tuple. `dict(view.budget)`
  copies only the top level, and the nested `limits` and `consumed` stay frozen. This bug
  class has shipped twice, and both times it failed quietly:
  - `charge_context_append` tested `isinstance(limits, dict)`, got a frozen mapping, and
    skipped every context-storage charge, so a paid overrun went uncharged;
  - `json.dumps` of a frozen tool observation raised, and read-token accounting surfaced
    it as a false `AdapterContractError`.

  Pass the value as a `Mapping` when the code only reads it. Call `thaw` (also in
  `task_graph`) when it needs mutable containers or plain JSON. Type-check with `Mapping`,
  never `dict`. Exact `dict` and `list` checks remain right for canonical wire values
  decoded from bytes. This applies to gatherers, accounting and the group coordinator as
  much as to derives.
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
`derive_entry` does not import `agent`. A node's initial requirements come from
`task_graph_admission.initial_requirements`, which admission and `derive_entry` share. It
takes the `requirement_version` artifact when the entry names one and the author packet
otherwise, and it rejects a packet and version that disagree. The entry state prefers the
pinned requirement version when present, falling back to the packet only when no version is
selected.

The controller (`next_step`, `select_edge`, `applicable_checks`, `Directive`,
`evaluate_guard`) reads only structured view fields. `select_edge` raises on any tie in
precedence. `next_step` halts as `no_admitted_evaluation` in two cases:
- `checking` with nothing to check and no feedback left;
- `awaiting_checks` with no required `each_turn`/`node_exit_candidate` check.

An empty required set never counts as a pass.

## Rationale and rejected alternatives

- **Rejected: derive re-checks of controller rules.** `derive_author_request` once matched
  the directive and then re-checked the feedback phase through a separate check layer,
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
  usage-charged limits seed no corresponding counters. `BudgetContractV1.usage_charged_limits`
  is the sole owner of generated, cumulative total, and context-token limits. The writer
  derive requires their corresponding usage evidence; V2 additionally binds usage counts to
  token IDs and charges `context_tokens` as the per-turn high-water mark
  `prompt_tokens + completion_tokens`, never as a cumulative sum. Reaching the high-water cap
  does not drain a cumulative controller budget: the next native sample must still be allowed
  to return the zero-generation `context_limit` record that seals the lineage incomplete.
  - **The contract can declare a token limit.** `BudgetContractV1.max_generated_tokens` is
    optional. It is listed in `_Contract.OMIT_NONE_FIELDS`, so it is left out of the
    canonical form when `None`, and every existing contract keeps its identity. This field
    departs from the design package. It exists so that the `token_limited` rollout fixture
    carries a real limit, which makes missing usage and `budget_charged` reachable.
  - **Why optional fields and not a new contract version.** A `BudgetContractV2` would force
    version dispatch in admission and `derive_entry` for optional token limits. Each unset
    limit is omitted from canonical form, preserving existing contract identities.
  - **Token-limited seals:** admission remains adapter-independent, but a real group refuses
    a manifest without `usage_reporting` before sampling. V1 keeps that single capability;
    V2 declares `usage_reporting`, `native_token_ledger` and `sampled_logprobs`, and a native
    group requires all three plus rendering and policy-pin agreement. `total_tokens` is a
    cumulative compute budget; `max_context_tokens` is the independent high-water context cap.
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
  `task_graph_scripted` and `task_graph_sampling` by concern. The layer order keeps these
  shared helpers below the derive modules. Before records moved to typed annotations, each
  field was declared in both a class and a side table; the current codec spec lives with
  the record annotation.
