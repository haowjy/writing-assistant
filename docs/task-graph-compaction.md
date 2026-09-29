# Task-graph context operations

Context operations replace future writer context without changing past actions, file state,
authoritative ledgers, or outcomes. The policy is a versioned `ContextPolicyV1`; the
operation is submitted as a typed `ContextOperationInputV1` containing its policy reference.
`derive_context_operation` calls `require_quiescent`, `select_context`, and `charge_budget`
directly. Replay reruns those pure functions and never calls a model or summarizer; there is
no separate context-operation evidence record.

## Context records and event boundary

The active context is represented by two immutable record families. `ContextContentV1`
nodes form a hash chain of messages, tools, and rendering pins. A root holds the tools and
rendering; each child appends content to its parent. `ContextRevisionV1` points to the
content head and records the source event. A checkpoint references the revision.

Ordinary writer/author/tool message appends are part of the source event's derived state;
they do not create additional context events. An explicit carry, seed, drop, or compact
operation has a `ContextOperationInputV1` input and one event. Its derived context, selected
source identities, and budget charge are reproduced from the input and verified view. The
revision's source-event reference avoids a hash cycle. Every commit still carries exactly one
runtime event, and there is no second runtime log.

## Available policies

| Operation | Future context |
|---|---|
| `carry` | The exact current messages, with a new explicit revision |
| `seed` | System/request plus a named same-lineage ancestor's visible context; seeded assistant messages have zero mask |
| `drop` | System/request only; authoritative files and ledgers stay unchanged |
| `compact` | System/request, one zero-mask environment summary, then a configured number of complete recent exchanges |

`visible-text-v1` renders role and typed visible parts of selected messages and takes the first
`max_summary_chars` Unicode characters. A zero limit is a valid exact empty summary. No model,
semantic paraphrase, caller-supplied summary, or learned policy is involved.

Every operation requires a quiescent writer boundary: ready for a writer turn, a drained tool
queue, no pending author request or check batch, and no unmatched tool/author exchange. A
checking final answer cannot be compacted before its checks; an author exchange cannot be
split between acknowledgement and reply. A seed names an immutable same-lineage ancestor;
sibling histories and arbitrary files are not accepted.

## Replay, accounting, and privacy

The derive checks the verified old context and ancestry, computes the selected messages and
summary, and charges context budgets. Replay recomputes selection and summary from the same
policy and view. Private author packets,
checks, rewards, undisclosed requirements, sibling rollouts, and raw files never enter a
summary. Only already validated writer-visible material is eligible.

The first operation pins `context_operations`, `context_bytes`, and
`context_storage_bytes` limits. `context_bytes` measures canonical UTF-8 bytes of the active
messages, tools, and rendering; `context_storage_bytes` counts persisted context-content nodes
and revisions. The unsupported `context_tokens` limit cannot be truthfully preflighted by this
adapter. An operation that would exceed a limit is rejected before publication. A paid writer
or author observation that grows context beyond a limit remains committed and charged; later
sampling is blocked until the environment seals a context-budget stop at a drained boundary.

Compaction preserves original action inputs and context revisions for later segment credit.
It never relabels an old action against a new summary. The writer entry and group policy seal
the required context/rendering pins; see [the writer runtime](task-graph-writer.md) and
[deterministic groups](task-graph-groups.md).
