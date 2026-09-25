# Opt-in task-graph writer stepping (Phase 4)

`TransactionalWriterV1` is a deterministic adapter for a **previously admitted** writer node and a trusted `RuntimeHandle` restored from its unsampled `ready_writer` entry checkpoint. It does not call a model. The legacy `run_agent`/`run_selected` paths remain the default. The opt-in [scripted-author lifecycle](task-graph-scripted.md) adds the author, check, transition and reward stages.
The [safe context-operation contract](task-graph-compaction.md) describes
`change_context` and its separate context budgets.

```python
from writing_agent.task_graph_writer import TransactionalWriterV1
from writing_agent.task_graph_projection import project_writer_context

writer = TransactionalWriterV1(store, admitted_graph, rollout_id, entry_checkpoint_id)
request_ref = writer.prepare_request(handle, adapter_request)  # before sampling
step = writer.submit_action(handle, parsed_assistant_message,
                            prepared_request_ref=request_ref, raw_output=adapter_raw_bytes,
                            trace=adapter_trace,
                            usage=adapter_usage)
handle = step.runtime
while (handle.state.position["phase"] == "ready_writer"
       and handle.state.continuation["next_call"] < len(handle.state.continuation["tool_queue"])):
    handle = writer.step_tool(handle).runtime  # scripted asks pause at awaiting_author
projected = project_writer_context(store, entry_checkpoint_id, handle.checkpoint_id)
```

The caller supplies the backend's parsed assistant message (`role`, text `content`, optional `tool_calls`) and any exact request/raw-output/trace evidence it has. `prepare_request` **before sampling** durably stores arbitrary adapter-owned request input and rendering/context pins; it does not inspect the payload's messages. `prepare_verified_messages` additionally checks a typed `messages` sequence against the current projected context, and rechecks it on publication and recovery. Neither path verifies arbitrary backend bytes or proves what a remote model consumed. `submit_action` rejects a stale prepared reference. Preparation is not a paid-call reservation or response journal. The direct `exact_request=` option persists supplied evidence before action commit for fake/evaluation adapters that did not prepare a request. Absent raw output is marked missing rather than reconstructed from the parsed message. Missing token IDs or logprobs are likewise explicit. Opaque logprobs, if supplied, must be a pre-persisted binary `payload` artifact named by `per_token_logprobs_ref`; numeric arrays are not silently converted through JSON. Even supplied tokens/logprobs do not confer native on-policy eligibility: Phase 4 has no token alignment or native loss mask. Only writer assistant text, call syntax, and endings carry loss-eligible metadata; seed, system, user, author, environment, and tool material do not. Provider usage detail fields are retained but not double-counted into top-level token charges.

The entry state's public `budgets_ref` must contain a canonical JSON artifact:

```json
{"schema":1,"limits":{"writer_turns":5,"tool_calls":10,"read_tokens":100,"storage_bytes":4096},"consumed":{"storage_bytes":17},"read_tokenizer":"whitespace-v1"}
```

The four required limits must equal the admitted node contract. Optional `generated_tokens` and `total_tokens` limits require backend usage evidence. Zero remaining capacity rejects `prepare_request` before sampling; `stop_exhausted` then seals a terminal, specifically classified stop. A sampled response that exceeds either limit becomes a durable `budget_charged` terminal event: it retains the supplied request, raw output, trace, parsed action and usage, charges the model call and measured tokens, and executes no tool. Writer-produced `budget_charged` and `termination_recorded` events use the explicit `writer_runtime` actor and must append their exact record to `WriterRuntimeLogV1`, even when they are the first event. Generic Phase 2 events retain their ordinary actor and are not inferred to be writer stops merely from their kind; they remain valid only on lineages without an admitted writer-entry capability. The admitted writer dispatcher rejects unsupported event kinds. A sampled stop's record and trace action IDs must both equal the independently derived next action ordinal. Its consumed counter may exceed the limit only in that terminal state. `context_tokens` is rejected at runtime initialization: this adapter cannot truthfully preflight exact rendered input plus reserved output before preparing a request. Only `whitespace-v1` read counting is currently semantically verifiable; custom counters/tokenizers are rejected. `storage_bytes` is the current UTF-8 byte gauge, not a cumulative write count. `attempted_tool_calls` counts every queued result, including validation and limit errors. Returned read observations alone incur read-token charges; rejected reads do not expose content.

Each action publishes one `writer_action` and a following `context_changed` event in one atomic Phase 2 commit. Its full ordered call queue starts at cursor zero. Each `step_tool` publishes exactly one `tool_result` plus a following `context_changed` event, also atomically, and advances the cursor once. The second event lets the context revision cite the already-hashed source event without a hash cycle; the two events are never separately visible. Action/message/trace records and the per-event metadata index are immutable artifacts reachable from the checkpoint. Phase 2 recorded effects remain the sole replay reducer. A restored handle resumes its next queued call; the adapter never regenerates or reruns committed calls. Stale handles fail the lineage-head check.

After the first safe context operation, `context_operations`, `context_bytes`,
and `context_storage_bytes` are also metered. They are byte/count budgets,
not an approximation of backend token capacity; a paid observation that
exceeds one remains committed and can be sealed with a typed context-budget
stop. See [compaction](task-graph-compaction.md) for exact definitions.

Backend call IDs are preserved in action metadata, while unique rollout-scoped logical IDs bind projected call syntax, queue entries, and results. Each element of a declared call array, including null or non-object elements, becomes exactly one queued call and one charged result; an absent array means no calls, and an explicit non-array `tool_calls` value rejects an otherwise admissible action because no call count is declared. A sampled token overrun still journals its usage/output evidence and stops, regardless of a malformed batch envelope. Invalid envelopes/discriminants, duplicate argument JSON keys, invalid UTF-8/lone surrogates, unsafe paths, missing/duplicate IDs, and unavailable calls use a safe `invalid_call` queue placeholder; representable bounded backend syntax remains escaped in immutable action evidence. Raw envelopes are bounded to 128 KiB, 64 nesting levels and 4096 nodes before evidence serialization; argument JSON is bounded to 64 KiB before decoding, and decoder-depth errors normalize to the same placeholder. They cannot mutate files. A mixed `ask_author`/file-tool batch rejects **all** calls before execution; an isolated `ask_author` is available only on an admitted scripted-author node. Ordinary file-tool batches are sequential: successful earlier effects remain committed if a later valid call fails. Legacy/default nodes expose only `list_dir`, `read_file`, `search`, `write_file`, and `patch_file`. Each call runs in a disposable private staging workspace with the existing text-tool semantics; publication restores a fresh workspace from the committed checkpoint. Expected path, including parent-is-file conflicts, patch, and policy failures become observations; disk/permission/Unicode corruption and other unexpected I/O failures interrupt before cursor advance or charge. Legacy `Workspace.dispatch` retains its older catch-all behavior. No shell, network, or arbitrary code is available.

`project_writer_context` takes an admitted entry checkpoint as its trust anchor, traverses only that checkpoint's causal suffix, and checks actor, audience, logical IDs, call/result order and originating action, queue/cursor, exact file delta, token/read/storage charges, pre/post execution-value hashes, trace context pins, and the persisted context against the authorized projection. The producer stages immutable candidate events/artifacts and runs **that same complete history walk** against the candidate state before `store.publish` can move the head. A rejected candidate may leave unreachable immutable files, but authority remains at its previous restorable checkpoint. Phase 4 events have exact state-field/history-key ownership: actions/results can change only continuation, position phase, budget, and runtime-log index (plus their respective history ID; results alone can change files); stops can change only position phase, outcome, runtime-log index, and sampled-stop budget. Every other authoritative field and nested position/continuation value must remain unchanged. Actions/results are legal only from `ready_writer`; a call-bearing action/result remains there, a call-free action enters `checking`, and budget stops enter `terminal` with incomplete/valid/pending outcome statuses. Trace type, independent action ordinal, request/prepared/raw-output references, rendering, usage, model and seed claims must agree with the owning action and prepared request. An adapter's `per_token_logprobs_ref` must exactly match the outer trace/record `logprob_ref`, including presence or absence. Restore and replay invoke the same history walk; the generic Phase 2 effect reducer is unchanged. Operational events, private packets, checks, rewards, sibling histories, and the project file map are not rendered. Project files can enter the conversation only through successful tool observations. Context content has a provenance-free hash; each persisted revision adds its source-event provenance. A tool-only exhausted turn can be sealed with `stop_exhausted`; a final reply instead enters `checking`, where later check/interaction work decides completion. The resulting terminal or checking checkpoint is replayable through `TaskGraphStore.replay`.

## Runtime adapters

The default offline composition root (`local_runtime_dependencies`) supplies the
caller-owned sampling evidence codec, atomic local checkpoint environment, staged
text tools, and deterministic file evaluator. `TransactionalWriterV1(...,
dependencies=bundle)` accepts an alternate `RuntimeDependenciesV1`; scripted
author, check, and terminal runtimes inherit that bundle from the writer unless
given one explicitly. A port implementation cannot bypass admitted check schemas,
semantic replay, or the current native-ineligible decision. No default adapter
invokes a model, shell, or network.

Each port has an immutable `PortDescriptorV1`. `bundle.manifest()` returns a
content-addressed `RuntimeManifestV1`; its artifact hash can occupy a sealed
group `adapter_ref` without changing the existing group wire format. This seals
the declared composition, not a proof that a remote backend consumed arbitrary
request bytes. See [group coordination](task-graph-groups.md) for policy sealing.
