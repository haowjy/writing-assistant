# Group coordination on the new core

This file covers `GroupCoordinatorV1` in
[`task_graph_group.py`](../writing_agent/task_graph_group.py) and its records in
[`task_graph_group_contract.py`](../writing_agent/task_graph_group_contract.py). The
sealed `GroupSpecV1` and its binding rules are in `task_graph_record_contracts`. How the
environment starts, resumes and verifies a lineage is in
[gate-and-rollout.md](gate-and-rollout.md). Read this before changing how a group starts,
collects or credits members.

**Mental model.** A group is a sealed `GroupSpecV1`: the entry checkpoint, the sampling
policy, an optional native-training mode and one seed slot per member. `training_mode` is
omitted when absent, so existing group identities stay fixed. A native-training group
requires `RuntimeManifestV2`, all three native sampling capabilities, and descriptor pins
that match both the sealed policy and entry rendering; V1 remains the evaluation contract.
Each member is its own new-core lineage that starts
from the shared entry checkpoint with a `MemberStartV1`. The gate verifies everything about
a member's history, including `view.group`, and binds every policy pin as it derives each
step. The coordinator adds only group bookkeeping: which result fills which slot, rewards,
advantages and segment credit. It reads verified views and the typed records they
reference, never raw event payloads.

## Start

`start(spec, ordinal, policy=...)` checks the sealed start contract, then takes one of two
branches:
- the member has no published head: call `environment.start_member(entry,
  MemberStartV1(spec.identity(), ordinal))`;
- otherwise, call `environment.open_head(member_id)`. A retry after any crash lands here.

It then verifies the handle, checks `_assert_member_view` (`view.state.position
["lineage_id"]` is the member and `view.group == spec`), and writes the start receipt.

**The receipt is derived from verified ancestry.** `_start_receipt` walks `view.ancestry` to
the node whose parent is the sealed entry, and that node is the start checkpoint. The file
`groups/<group_id>/start-<ordinal>.json` (`member_id`, `parent_checkpoint_id`,
`start_checkpoint_id`) is an immutable cache of that answer. Every call recomputes it from a
verified view, and `_receipt` refuses to replace the file with different bytes.

*Rejected: a stored receipt as its own source of truth.* The receipt used to be written
after `start_member` returned. When a crash fell between the two and a worker then advanced
the member through `open_head`, the retry recorded the advanced head as the start. The
receipt is immutable, so every later `start` and `collect` failed. Deriving it from ancestry
makes that crash window harmless, and it reduced `start` to the two branches above.

## Collect and finalize

`collect` (and `collect_scripted` and `collect_invalid`, which build a result and call it)
and `finalize` admit every result through `_admit_result`. That is the one admission
boundary for live collection and for finalizing a reopened group.
- **A real result** is checked against `_verified_member_view`: `open_head`, then `verify`,
  then `_assert_member_view`. The outcome, reward and eligibility come from `view.outcome`
  and the `OutcomeV1`, `RewardV1` and `TrainingEligibilityV1` it references. Segment credit
  comes from `view.samples` and the contexts in the verified ancestry.
- **A fixture result** carries a `GroupScriptedTerminalV1`. A fixture group is never
  eligible for the native optimizer, and fixture and real results cannot share a group.
- **An infrastructure failure** carries a `GroupExecutionFailureV1` and never a reward.
- `reward_of(result)` is the one reward read. `finalize` and the reward status share it.

The result kinds are typed records, so their codecs do the schema checks that used to be
hand-written dictionary checks. Member-result, decision and advantage wires are unchanged.
Segment credits add optional completion offsets for eligible native turns; absent offsets
are omitted, preserving existing non-native credit identities.

**There is no collect-time policy check, and none should be added.** The gate binds the
group pins in `derive_writer_turn` and the context policy in `derive_context_operation` as
it derives each step, so every view that `open_head` returns has already passed them. And
`view.group == spec` carries the design §10.4 obligation. The check that S6 added was deleted for two reasons:
- It was dead. Removing its call left every test green, because the gate's fold had already
  rejected every drift it was meant to catch.
- It walked the wrong range. Its `context_changed` walk followed `event.previous` to
  genesis, through the entry lineage's pre-group history, where `view.group` is `None`. It
  could only add false rejections.

The only way to make a collect-time check live is to collect from a store opened without the
gate, which is the design the seam replaced.

## Segment credit (MEDIUM-1)

Design §10.1 gives credit only to message parts that carry sampled content. **The writer
derive makes that decision; the group reads it.** In `_build_assistant_message`, an
`invalid_tool_call` part whose call has no bounded, canonical sampled value gets the exact
`raw` value `{"$noncanonical": "no-sampled-content"}`. That covers:
- a call that intake replaced with a placeholder (for example, a call over the size or depth
  bound);
- a value that contains any `$noncanonical` tag;
- an index past the parsed calls;
- a non-list `tool_calls` value that contains any such tag.

`_segment_credits` maps `text` to `assistant_text`, and `tool_call` or `invalid_tool_call`
to `tool_syntax`. It skips a part only when `raw` equals the sentinel. Never re-infer
"sampled" in the group by inspecting nested tags; the derive's decision is the one rule.
Before the sentinel existed, an unbounded call earned `tool_syntax` credit, which was hashed
on the environment's placeholder.

`_segment_credits` accepts both `WriterTurnV1` and `WriterTurnV2` evidence. V2 byte ledgers,
sampling pins and termination remain owned by the writer derive; group collection does not
re-validate sampling evidence or infer eligibility from the token trace.

For structurally eligible native groups, segment credits also carry the turn's completion
token span, derived from the same V2 chain used by `task_graph_training_export`. The export
builds one `TrainingBatchV1` per settled native group; generated IDs are loss-masked in and
external suffix IDs out. A trailing zero-generation context-limit turn stays on the batch as
audit evidence but never enters its sequence or loss mask.

## Runtime boundary

`GroupCoordinatorV1` requires a `RolloutEnvironment`. It has no bare-store, branch or workspace-restore start path; group members begin and resume only through the verified environment.

Real collection binds the result to the verified published member head. `collect_invalid`
also refuses to relabel a valid terminal outcome as infrastructure-invalid. Group session
seals are checked before member publication, so a manifest mismatch cannot leave a started
head behind.
