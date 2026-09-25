# Opt-in scripted-author task-graph slice

This is a deterministic research runtime beside the unchanged legacy runner. It
accepts only `scripted_author` interaction nodes with role-typed private author
and evaluator packets, public decision IDs/labels, an exact private answer and
feedback script, strict deterministic file checks, and integer reward weights.
`simulated_author`, semantic judges, compaction, native optimizer traces, and
model-backed author replies are not supported here.

## Entry and authority

Construct the immutable graph and all referenced artifacts, call `admit_graph`,
then save an unsampled `ready_writer` entry checkpoint. The state must name the
admitted private `AuthorPacketV1` and initialize `DecisionLedgerV1`,
`DisclosureLedgerV1`, and `RequirementLedgerV1` artifacts. The entry context
must use `writer_tool_schemas(entry.tool_allowlist, admitted_node.interaction_policy)`:
the `ask_author` schema exposes only public IDs and labels, never answer values.
Decision IDs use bounded ASCII identifier syntax; labels are bounded printable
text. Admission screens every rendered public ID and label against undisclosed
private author text. This is
an exact-byte disclosure guard, not a semantic confidentiality proof.
The tests in `tests/test_task_graph_scripted.py` show a complete construction.

The writer may call `ask_author` with exact `question`, `decision_ids`,
`proposals`, and `option_refs` fields. It must be the only call in its assistant
message. Invalid IDs, duplicate IDs, malformed or oversized fields, and unknown
proposal references become paired writer tool observations; a mixed batch rejects
all calls before any file operation. A valid action commits first. Its author
request commits separately with a unique request ID and `awaiting_author` phase.
`ScriptedAuthorRuntimeV1.reply` resolves the frozen script, commits a tool
acknowledgement, disclosure, and exactly one explicit user reply in one commit,
then resumes the same node. A partial, reordered, duplicated, or ackless reply
cannot publish or restore; the control call must be drained before the request
clears or reply is visible. A restored
request does not call a live author. Script selector misses terminalize as
`simulator_error` with unavailable reward, not as bad writing. Author text such as
“DONE” never establishes completion.
Tool-call exhaustion takes precedence over malformed-call and author-call
exhaustion; otherwise malformed calls take precedence over author exhaustion.
Each drained error counts one attempted call, but no exhausted tool counter is
incremented or capped to permit a successful author request.

```python
writer = TransactionalWriterV1(store, admitted_graph, rollout_id, entry_checkpoint)
author = ScriptedAuthorRuntimeV1(writer)
checks = DeterministicChecksV1(writer)
terminal = ScriptedTerminalV1(writer)

action = writer.submit_action(handle, assistant_message)
request = writer.step_tool(action.runtime)  # phase: awaiting_author
reply = author.reply(request.runtime)       # phase: ready_writer
# After an eventual final writer action has entered checking:
batch = checks.request_checks(final_action.runtime)
result = checks.check_next(batch.runtime)   # repeat for each outstanding check
edge = terminal.transition(result.runtime)
outcome = terminal.terminal_outcome(edge.runtime)
reward = terminal.reward(outcome.runtime)
```

If mandatory feedback is declared, after the relevant progress checks call
`author.request_feedback(handle)` and `author.reply(request.runtime)`. It delivers
at most one ordered feedback item per completed writer turn, then requires writer
continuation. A feedback rule may name a private `RequirementUpdateV1`; the
environment commits its typed supersession separately from the author utterance.
When its prerequisite fails or the author/writer budget cannot deliver the next
feedback item, `terminal.stop_incomplete(handle)` seals a valid incomplete outcome;
checks not run are explicitly `not_run` reward components rather than fabricated
passes or evaluator outages.
Reward components must have terminal (`each_turn` or `node_exit_candidate`)
check evidence; progress-only checks cannot be reward components in this slice.
Drained writer-turn exhaustion, including after a reply, and measured generated-
or total-token overruns seal a typed incomplete `TerminalOutcomeV1` bound to a
frozen checkpoint. `terminal.reward` then publishes the declared incomplete
score and availability; sampled overruns retain their usage and output evidence.
The writer sees only the explicit user reply and its own file-tool observations;
the private packet, check/evaluator material, requirement bytes, and reward do not
enter writer context.

## Frozen checks and reward

`request_checks` accepts only a quiescent final-turn checkpoint. That immutable
checkpoint is the candidate target. Every check request names that target, its
admitted check and evaluator packet, and the active requirement version. The
deterministic evaluator supports only file-target `nonempty`, `contains`,
`excludes`, `excludes_all`, `word_range`, and `exact` in this mode. A result records
recomputed evidence and cannot edit files. The environment alone applies a
permitted terminal edge or seals an incomplete outcome. Terminal outcome,
component reward, reward availability, and training eligibility are separate
immutable records. Reward uses exact integer numerator/normalization arithmetic;
training is ineligible until native action-token alignment exists.

Every producer stages its event, invokes `project_writer_context` as the shared
semantic validator, and then publishes through the compare-and-swap store.
The store also runs that validator for Phase 5 direct publication before moving
the head. Restore and offline replay invoke the same validator
without a model, network call, author provider, or check worker. A failed producer
may leave unreachable immutable artifacts; it cannot move the lineage head.
The immutable admitted entry contract anchors Phase 5 authority even if a child
tries to null its author packet or drop the runtime log. Unknown generic event
kinds cannot mutate this lineage; generic Phase 2 fixtures remain supported
outside it.
