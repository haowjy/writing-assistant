# Safe task-graph context operations

The opt-in task-graph writer can replace **future request context** without changing
past actions, prepared requests, tool observations, files, decisions, or outcomes.
This is a deterministic environment operation, not a writer action or a model
summary. It is available on both text-tool and scripted-author lineages; the
legacy runner remains unchanged.

```python
from writing_agent.task_graph_compaction import ContextPolicyV1

policy = ContextPolicyV1(
    "compact",
    retained_exchanges=1,
    summarizer_version="visible-text-v1",
    max_summary_chars=2048,
    max_operations=8,
    max_context_bytes=100_000,
    max_context_storage_bytes=1_000_000,
)
step = writer.change_context(handle, policy)
handle = step.runtime
```

The immutable `ContextPolicyV1` supports only:

| Operation | Future context |
|---|---|
| `carry` | The exact current messages, with a new explicit revision |
| `seed` | System/request plus a named ancestor checkpoint's visible context; seeded assistant messages are zero-mask |
| `drop` | System/request only; authoritative files and ledgers do not change |
| `compact` | System/request, one zero-mask environment summary, then a specified number of complete recent exchanges |

A seed names an immutable same-lineage ancestor checkpoint via `seed_name` and
`seed_checkpoint_ref`; arbitrary files, sibling histories, and unverified
prefixes are not accepted. `visible-text-v1` renders the role and typed visible
parts of selected messages, then takes the first `max_summary_chars` Unicode
characters. Zero is valid and records an exact empty UTF-8 summary artifact.
There is no model, semantic paraphrase, caller-supplied summary text, or learned
compaction policy.

Every operation requires `ready_writer`, a drained call queue, no pending author
request, check batch, external request, in-flight effect, or unmatched tool/author
exchange. A final answer in `checking` cannot be compacted before its checks.
An author request cannot be split between acknowledgement and reply. The
retained tail counts completed exchanges, not individual messages or prior
environment summaries.

Each published context-operation `context_changed` event has a typed immutable
`ContextOperationV1` record. It binds the old/new revision and content hashes;
indexed message and source-event identities; exact source-event sequence range;
removed messages and retained tail; seed name/checkpoint; fixed algorithm and
configuration; exact summary text and binary artifact; and every context budget
charge. The revision's provenance names the preceding event, avoiding a hash
cycle with its own `context_changed` event. The shared semantic projector checks
the record, field/actor/audience ownership, selection, source identities,
summary proof, and charges both before the lineage head moves and on restore or
offline replay. Replay uses the recorded summary and selection; it never invokes
the summary generator or a model.

The immutable admitted writer entry selects that semantic walk, including for a
first operation, a missing or retyped runtime log, and every event in a batch.
Generic Phase 2 lineages without a typed writer entry retain their generic
reducer. A context-operation event must name the causal pre-state's visit,
version, and provenance references exactly.

The first operation freezes `context_operations`, `context_bytes`, and
`context_storage_bytes` limits in the rollout budget artifact. Later policies
must use the same limits. `context_bytes` measures canonical UTF-8 bytes of
the active messages, tools, and rendering inputs. `context_storage_bytes`
accumulates serialized context revisions and summary bytes from activation
onward; it is distinct from workspace-file `storage_bytes`.
An operation that would exceed a limit rejects before publication. A paid
writer or author observation that grows context beyond a limit remains
committed and charged; further request preparation/sampling is rejected and
`stop_exhausted` seals a valid `context_budget` or
`context_storage_budget` outcome at a drained writer boundary. The existing
`context_tokens` limit remains unsupported because this adapter cannot verify
rendered input tokens plus reserved output capacity.

Only the already validated writer-visible projection is eligible for summary.
Private author packets, checks, rewards, undisclosed requirements, sibling
rollouts, and raw workspace files never enter it. Files can appear only after a
permitted tool observation. Original action traces and prepared requests retain
their original context revision, raw evidence, and loss eligibility for later
credit assignment; no historical action is relabeled against a later summary.

`prepare_verified_messages(handle, payload)` verifies that the payload's typed
`messages` sequence is exactly the current context and preserves that assertion
for publication and recovery. A stale pre-compaction sequence is rejected. The
adapter can prepare it with
`{"messages": [message.to_dict() for message in handle.context.messages]}`
plus its other request fields, then sample from the persisted payload. Only
that typed message sequence is verified. The older `prepare_request` path only
pins arbitrary adapter-owned payload bytes or JSON to a context identity; it
does **not** verify their messages. It does not claim that a backend rendered
or used the compacted context. Neither path
verifies arbitrary backend request bytes or makes native training eligible;
native token alignment and loss masks remain unimplemented.
