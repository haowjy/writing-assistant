# Rollout execution

This file covers the synchronous driver, typed gatherers, resume path and runtime
acceptance tests for the new core. The gate and store-verifier contract, publication
handoff, view cache, rule owners and environment methods are in
[gate-and-rollout.md](gate-and-rollout.md). The wire records, derives and layer order are in
[transition-seam.md](transition-seam.md); group-specific collection and credit are in
[group-coordination.md](group-coordination.md).

Read this before changing how a rollout obtains port inputs, invokes providers, resumes, or
proves offline replay.

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
  and `native_history` rebuilt from the last committed `WriterTurnV2` and its token artifacts.
  That ledger is re-prefilled per decision; no model KV cache crosses a sample call. The
  concrete adapter is `native_gemma.NativeGemmaSampleBackend`.

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
