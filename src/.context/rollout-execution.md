# Rollout execution

This file covers the synchronous driver, typed gatherers, native Gemma sampling, tool
outcomes, the resume path and runtime acceptance tests for the new core. The gate and
store-verifier contract, publication handoff, view cache, rule owners and environment
methods are in [gate-and-rollout.md](gate-and-rollout.md). The wire records, derives and
layer order are in [transition-seam.md](transition-seam.md); group-specific collection and
credit are in [group-coordination.md](group-coordination.md).

Read this before changing how a rollout obtains port inputs, invokes providers or the native
sampler, classifies tool outcomes, resumes, or proves offline replay.

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
that is all `SampleBackend.sample` receives. It carries canonical messages, tools, and
rendering, but no persisted request body or request-reference pair;
- the context revision ref and one `context_content_hash`;
- for a group member, the sealed `writer_seed` and the model, behavior-policy, decoding,
  tokenizer and template refs. Outside a group these are `None`.
- for a native-training group only, `native_sampling_budget` carries the remaining generated
  allowance and `max_context_tokens`; it exposes no cumulative usage ledger. The gatherer
  forwards this allocation unchanged to `PreparedSamplingInput`.
- Native sampling also receives the sealed `adapter_ref`, the action-count `decision_ordinal`,
  the pending `action_id`, and `native_history` rebuilt from the last committed
  `WriterTurnV2` and its token artifacts. The concrete adapter is
  `native_gemma.NativeGemmaSampleBackend` ([Native Gemma sampling](#native-gemma-sampling)).

The rules:
- **There is one content hash: `view.context.content_ref`.** `step_input` supplies it, and the
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
- **Context claims are bound for every lineage.** `decode_and_bind_sampling` compares a
  trace's `context_revision_ref`, `context_content_hash` and rendering with the active
  context whenever each claim is present. This is O2's present-only rule applied to every
  lineage. When only the group binder checked these claims, a non-group commit with false
  context claims published. `bind_group_sampling_claims` compares only the sealed policy
  refs, `policy_ref`, seed and model. Requiring the claims to be present belongs to the
  HIGH-5 group-binding follow-up.
- **The member seal reads the verified view.** `RuntimeSession.require_member_seal` checks
  the lineage against `view.group` and the executing manifest against its `adapter_ref`. It
  never infers membership from the lineage name or reads `groups/` files.

## Native Gemma sampling

`native_gemma.py` and `native_protocol.py` are the model side of the seam. Neither is
imported by the core. `torch` and `transformers` load lazily inside `sample()`.

- **The backend samples the trainer's live model.** `NativeGemmaSampleBackend` receives the
  model and tokenizer; it never loads weights. Every `sample()` calls
  `assert_active_adapter`, because an adapter hash covers tensors but not whether they are
  applied. The trainer checks again before the audit.
- **The ledger is append-only and re-prefilled.** The first decision renders the initial
  context (`NativeGemmaRenderer.render_initial`). Every later decision's input is the
  committed input, plus the committed generated IDs, plus
  `external_suffix(...)`, rendered from the committed messages. Sampled assistant turns are
  never re-rendered: Gemma's template reorders tool arguments and drops earlier thinking.
  `generate` uses a cache only within one call. No KV cache crosses a sample call, so a
  weight update always meets an empty cache. Each turn records
  `prefill_tokens = len(input)` and `cached_input_tokens = 0`.
- **The context limit is a writer outcome.** The backend computes `allowed` exactly as rule 3
  does and passes `max_new_tokens = allowed`. When `allowed ≤ 0` because of the context term,
  it returns a zero-generation turn with `termination.kind = "context_limit"`, and the
  derive seals the lineage `incomplete` (`context_tokens_budget`). An exhausted generated
  budget at that point is an `AdapterContractError`. The seed for each decision is
  `derive_group_seed(writer_seed, "decision", decision_ordinal)`. The stop set is `<eos>`,
  `<turn|>` and `<|tool_response>` (IDs 1, 106, 50). A sampled `<eos>` is kept, and a
  follow-up user turn is appended as masked external tokens without inventing `<turn|>`.
- **Logprobs are observational evidence, not loss inputs.** `_ObservationalLogitsProcessor`
  returns the scores unchanged. It keeps one pending fp32 `log_softmax` row and resolves it
  with the token the next step appends (`input_ids[:, -1]`); `finish()` resolves the last
  row. Only the chosen token's logprob is kept, never a full-vocabulary row. It does not
  replay sampling under a saved RNG. TRL recomputes logprobs (`num_iterations=1`), and the
  probe reports `|sampled − recomputed|` as drift without gating on it.
- **Suffix deltas.** `native_suffix` supports a tool delta of N results in exact call-ID
  correspondence, followed by one user reply only when the calls include `ask_author`. It
  also supports a final answer followed by at most one user turn. Any other delta, or a
  suffix that is not prefix-stable, raises `ProtocolError`, which is infrastructure.

### Parse classification

`native_protocol.parse_native_response` is the one parse, shared by the sampler, the audit
and the scripted CPU backend. It accepts only `native_stop` or `token_limit` turns and calls
`inference.parse_response`. Then:

- a `ValueError` whose message starts with one of the recognized malformed-output prefixes
  (`native_parse_errors.is_native_output_parse_error`) is the model's failure. The turn commits the
  decoded text with no calls and `native_parse_failed = True`
  ([transition-seam.md](transition-seam.md));
- any other exception raises `ProtocolError("Native response parser failed unexpectedly")`.
  A wrong tokenizer or a bug in our code halts the run and is never scored as model
  behavior.

The prefixes and `parse_response`'s two own message constants live in the leaf
[`native_parse_errors.py`](../writing_agent/native_parse_errors.py), which imports no
`writing_agent` module. This module matches the experiment identity's `native_*.py` source
glob. The prefixes come from the pinned Transformers response parser (including
`"json parser could not parse region as JSON"`, which a limit inside a call header
produces). The exact match fails closed: if a message is reworded, the next malformed
output halts instead of scoring. Both `inference` and `native_protocol` depend on this leaf,
so their import graph remains acyclic. A sweep test covers every prefix of three call shapes
under both terminations. Parsing then binds each call's raw ID to the core's
`tool_call_id`, so raw IDs, committed results and the audit's re-parse agree by exact ID.

### Tool outcomes

`task_graph_tool_outcomes.read_member_tool_outcomes(start_view, final_view)` is the one
reader of what a member's tool calls did. The trace check and P1 criterion 1 both use it.
It pairs every new committed call with its result by exact ID and returns:

- per-call results, ordered by action ordinal and then call index;
- `counts_by_code` and `protocol_shaped_rejection_count`;
- `files_changed` and `changed_paths`.

Each result is `"ok"` or `{"code", "text"}`:

- **Intake rejections** map to their code through `task_graph_calls.REJECTION_MESSAGES`,
  by exact message equality. An unknown text raises `ToolOutcomeError`. Persisted messages
  are unchanged.
- **Failures of an admitted call** (a missing path, unmatched patch text, and so on) are
  `tool_execution_failure`, which is model behavior. A host fault never becomes a committed
  observation.
- **Calls parsed in the turn that ends an incomplete lineage** are
  `not_executed_incomplete`. Only the final action's calls qualify, and only when the stop
  reason is a termination reason. Any other unpaired call or result raises.

`PROTOCOL_SHAPED_REJECTION_CODES` is `invalid_envelope`, `invalid_function_envelope`,
`missing_id`, `duplicate_id`, `tool_calls_not_array`, `arguments_not_object` and
`invalid_arguments_json`. The native parser and binder make each one impossible, so any
occurrence means our code refused the model's call. A real-model run with one is a halt,
however many members are admitted. Argument-shape errors (`ask_author_*`), `unsafe_path`
and budget refusals are the model's own behavior and are only recorded.

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

The `tests/test_task_graph_accept_*` modules exercise the forgery and boundary
scenarios through the new core (design §12). Their index is
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
