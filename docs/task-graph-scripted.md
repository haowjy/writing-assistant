# Scripted-author task-graph lifecycle

The scripted-author path runs on the same typed, verified task-graph core as writer-only
rollouts. It supports `scripted_author` interaction nodes, role-typed private author and
evaluator packets, declared public decision IDs and labels, a deterministic answer/feedback
script, file checks, and integer reward weights. It does not add model-backed author replies,
semantic judges, native optimizer traces, or training updates.

Build and admit the immutable graph and its artifacts, then create the unsampled root through
`RolloutEnvironment.enter`. `RolloutDriver.run` obtains a verified view and next directive,
passes only the matching typed port input to a gatherer, and commits one input/event at a
time. See [the writer runtime](task-graph-writer.md) for the entry, resume, and publication
contracts.

## Author exchange

The `ask_author` tool schema exposes public decision IDs and labels, never private answer
values. It must be the only call in the assistant message. Invalid IDs, duplicate IDs,
malformed or oversized fields, and unknown proposal references become paired writer tool
observations; a mixed batch rejects before any file operation. A valid call queues a typed
author request. The script resolves a reply only after the driver reaches its author-reply
directive.

`ScriptedAuthorSource` returns an `AuthorReplyV1`; `derive_author_reply` applies the exact
scripted answer and validates it. The single `author_turn` event's derivation commits the
control-tool acknowledgement, authorized disclosure and requirement changes, user utterance,
budget, context update, and cleared request together. These facts are derived from one input,
not appended as a sequence of independently published events. The acknowledgement and user
message enter the immutable context content/revision records; the runtime keeps no separate
log. On resume, a persisted request is answered from the frozen script; no live author is
called. Author text such as “done” never establishes task completion.

A script selector miss is a simulator-coverage failure with unavailable reward, not bad
writing. Tool-call exhaustion takes precedence over malformed-call and author-call exhaustion;
otherwise malformed calls take precedence over author exhaustion. Each drained error counts as
an attempted call without permitting a successful author request beyond the limit.

## Checks and outcomes

`RolloutDriver` dispatches check requests through the typed evaluator port. The deterministic
evaluator supports file-target `nonempty`, `contains`, `excludes`, `excludes_all`,
`word_range`, and `exact` checks in this mode. Results name the frozen candidate and admitted
check/evaluator packet; the evaluator cannot edit files. Replay re-derives recorded results
offline and does not call the evaluator.

The state points to one evolving `OutcomeV1`: it holds the check batch and results, selected
transition edge, terminal status, reward ref, and training-eligibility ref as those stages
complete. Reward arithmetic is exact integer numerator/normalization arithmetic. Evaluation
cannot rewrite the candidate state. Training eligibility remains separate from reward and is
not native on-policy eligibility.

The writer sees only its visible messages, file-tool observations, and explicit author
utterance. The private packet, evaluator material, undisclosed requirements, and reward stay
outside writer context. Requirements and decisions remain authoritative in their private
ledgers; any authorized public disclosure is added by the same `author_turn` derive.

For context carry/seed/drop/compaction, see [the context-operation contract](task-graph-compaction.md).
For sealed member policies and group results, see [deterministic groups](task-graph-groups.md).
