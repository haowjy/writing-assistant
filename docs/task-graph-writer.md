# Opt-in task-graph writer stepping (Phase 4)

`TransactionalWriterV1` is a deterministic adapter for a **previously admitted** writer node and a trusted `RuntimeHandle` restored from its unsampled `ready_writer` entry checkpoint. It does not call a model, author simulator, evaluator, or graph transition. The legacy `run_agent`/`run_selected` paths remain the default.

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
while handle.state.continuation["next_call"] < len(handle.state.continuation["tool_queue"]):
    handle = writer.step_tool(handle).runtime
projected = project_writer_context(store, entry_checkpoint_id, handle.checkpoint_id)
```

The caller supplies the backend's parsed assistant message (`role`, text `content`, optional `tool_calls`) and any exact request/raw-output/trace evidence it has. A capable adapter calls `prepare_request` **before sampling**; this durably stores the exact request and rendering/context pins, and `submit_action` rejects a stale prepared reference. Preparation is not a paid-call reservation or response journal. The direct `exact_request=` option persists supplied evidence before action commit for fake/evaluation adapters that did not prepare a request. Absent raw output is marked missing rather than reconstructed from the parsed message. Missing token IDs or logprobs are likewise explicit. Opaque logprobs, if supplied, must be a pre-persisted binary `payload` artifact named by `per_token_logprobs_ref`; numeric arrays are not silently converted through JSON. Even supplied tokens/logprobs do not confer native on-policy eligibility: Phase 4 has no token alignment or native loss mask. Only writer assistant text, call syntax, and endings carry loss-eligible metadata; seed, system, user, author, environment, and tool material do not. Provider usage detail fields are retained but not double-counted into top-level token charges.

The entry state's public `budgets_ref` must contain a canonical JSON artifact:

```json
{"schema":1,"limits":{"writer_turns":5,"tool_calls":10,"read_tokens":100,"storage_bytes":4096},"consumed":{"storage_bytes":17},"read_tokenizer":"whitespace-v1"}
```

The four required limits must equal the admitted node contract. Optional `generated_tokens`, `total_tokens`, and `context_tokens` limits require corresponding adapter evidence before accepting an action. `storage_bytes` is the current UTF-8 byte gauge, not a cumulative write count. `attempted_tool_calls` counts every queued result, including validation and limit errors. Returned read observations alone incur read-token charges using the versioned `read_tokenizer` supplied to the adapter (default `whitespace-v1`); rejected reads do not expose content.

Each action publishes one `writer_action` and a following `context_changed` event in one atomic Phase 2 commit. Its full ordered call queue starts at cursor zero. Each `step_tool` publishes exactly one `tool_result` plus a following `context_changed` event, also atomically, and advances the cursor once. The second event lets the context revision cite the already-hashed source event without a hash cycle; the two events are never separately visible. Action/message/trace records and the per-event metadata index are immutable artifacts reachable from the checkpoint. Phase 2 recorded effects remain the sole replay reducer. A restored handle resumes its next queued call; the adapter never regenerates or reruns committed calls. Stale handles fail the lineage-head check.

Backend call IDs are preserved in action metadata, while unique rollout-scoped logical IDs bind projected call syntax, queue entries, and results. Missing, duplicated, malformed, or unavailable calls become paired error observations and consume attempted-call budget without file effects. A mixed `ask_author`/file-tool batch rejects **all** calls before execution; an isolated `ask_author` is unavailable in Phase 4. Ordinary file-tool batches are sequential: successful earlier effects remain committed if a later valid call fails. Only `list_dir`, `read_file`, `search`, `write_file`, and `patch_file` are exposed. Each call runs in a disposable private staging workspace with the existing text-tool semantics; publication restores a fresh workspace from the committed checkpoint. No shell, network, or arbitrary code is available.

`project_writer_context` takes an admitted entry checkpoint as its trust anchor, traverses only that checkpoint's causal suffix, and checks actor, audience, logical IDs, call/result order, queue, trace context hash, and the persisted context against the authorized projection. Operational events, private packets, checks, rewards, sibling histories, and the project file map are not rendered. Project files can enter the conversation only through successful tool observations. Context content has a provenance-free hash; each persisted revision adds its source-event provenance. A tool-only last allowed turn can be sealed with `stop_exhausted`; a final reply instead enters `checking`, where later check/interaction work decides completion. The resulting terminal or checking checkpoint is replayable through `TaskGraphStore.replay`.
