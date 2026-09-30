# Group coordination on the new core

This file covers `GroupCoordinatorV1` in
[`task_graph_group.py`](../writing_agent/task_graph_group.py). Its result, decision,
advantage and credit records are in `task_graph_group_records`; the sealed `GroupSpecV1` and
its binding rules are in `task_graph_record_contracts`; `task_graph_group_contract` holds the
seed derivation, policy validation and sealed-environment resolution. How the environment
starts, resumes and verifies a lineage is in [gate-and-rollout.md](gate-and-rollout.md).
Read this before changing how a group starts, collects, credits or trains members.

**Mental model.** A group is a sealed `GroupSpecV1`: the entry checkpoint, the sampling
policy, an optional native-training mode and one seed slot per member. `training_mode` is
omitted when absent, so existing group identities stay fixed. A native-training group
requires `RuntimeManifestV2`, all three native sampling capabilities, and descriptor pins
that match both the sealed policy and entry rendering; V1 remains the evaluation contract
([Native training groups](#native-training-groups)).
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
["lineage_id"]` is the member and `view.group == spec`), and calls `start_receipt`.

**The receipt is derived from verified ancestry.** `start_receipt` walks `view.ancestry` to
the node whose parent is the sealed entry, and that node is the start checkpoint. The file
`groups/<group_id>/start-<ordinal>.json` (`member_id`, `parent_checkpoint_id`,
`start_checkpoint_id`) is an immutable cache of that answer. Every call recomputes it from a
verified view, and `_receipt` refuses to replace the file with different bytes.

*Rejected: a stored receipt as its own source of truth.* The receipt used to be written
after `start_member` returned. When a crash fell between the two and a worker then advanced
the member through `open_head`, the retry recorded the advanced head as the start. The
receipt is immutable, so every later `start` and `collect` failed. Deriving it from ancestry
makes that crash window harmless, and it reduced `start` to the two branches above.

`collect_completed(spec, ordinal, runtime)` accepts a completed runtime, rebuilds its
terminal and available-reward references from the verified view, and owns the result
receipt through `collect`. Adapters do not assemble terminal references themselves; the
trainer and the S11 trace check both collect through it.

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
token span, taken from the same layout (`task_graph_training_layout`) the export uses.

## Native training groups

Native-group sealing, export, audit, and trainer-consumption receipts are described below.
The canonical requirement that every member be admitted before trainer consumption is in
[transition-seam.md](transition-seam.md).

- **Sealing.** `seal` requires a `RuntimeManifestV2` with all three native capabilities,
  descriptor refs equal to the sealed `POLICY_FIELDS`, and a renderer equal to the entry's
  rendering pins; every refusal is `AdapterContractError` (`_require_group_seal_contract`).
  It also refuses an entry whose budget declares `max_total_tokens`. Every limit that can end
  a native lineage must be one rule 3 derives, and a cumulative total is not: a trailing
  `context_limit` turn is charged its whole input, so the total could bind first and push an
  honest member through the overrun path. Compute stays bounded by turn counts,
  `max_context_tokens` and the stage supervisor. Real members need a bound session in every
  runner mode (`_require_bound_group_session`). The environment keeps the two meanings in
  separate flags: `require_session` (a session must be bound) and `caller_is_group_session`
  (the caller is the coordinator acting for the group).
- **Export.** `finalize` settles the group. `export_training_batch` then projects a `ready`
  or `tie` group into a `TrainingBatchV1`: per member, `prompt_ids` (the first turn's input),
  `completion_ids` (generated and external suffix IDs up to the last generated token), an
  `env_mask` that is 1 on generated IDs only, and the exact advantage plus its f64 value. It
  refuses `pending` and `invalid` groups, members that are not `structurally_eligible`,
  more than one context root, an empty mask and an over-cap length. A trailing
  zero-generation `context_limit` turn stays on the batch as an audit-only ref. Eligibility
  already refuses what export would, so an eligible member failing layout in `finalize` is a
  `GroupInvariantError`, not a training refusal.
- **Audit and admission.** The tokenizer-backed `native_audit.audit_training_batch` re-derives
  every member's tokens and writes a `TrainingAdmissionV1` (see
  [transition-seam.md](transition-seam.md)). `record_training_admission` checks that the
  admission, batch and decision all belong to this sealed group, then writes the immutable
  receipts `groups/<group_id>/training-batch.json` and `training-admission.json` under the
  group lock. `record_training_consumed` writes `trainer-consumed.json` only when it names
  those same refs. That receipt carries `admission_status`, so a refused batch leaves a
  durable `refused` record before the run halts.
- **One group per step.** The trainer claims `groups/step-NNNNNN` under a lock before
  sealing, and every later state is written back into that reservation: `sealing`,
  `sealed`, `consumed`, `audit-refused`, `adapter-drift` or `halted`. `groups_by_sequence`
  (in `task_graph_group_index`) is the one fail-closed parser of sealed specs and
  reservations. An unrecognized file or state refuses. Both of the trainer's refusals use
  it:
  - it refuses to seal when any group or reservation exists for the step;
  - resume preflight refuses, before the model loads, when anything exists at or after the
    checkpoint's `global_step`.

  A crash mid-step is therefore terminal. Only a clean `--stop-after-step` boundary resumes.
  A whole-group rerun needs a separate, explicit experiment under a new ID.
- **Outcomes in training.** A `tie` group trains with exact zero advantages, and the
  optimizer still steps. A `pending` or `invalid` group, a collection failure or an admission
  refusal halts the run without resampling.

Adapter callers never address `groups/<group_id>` receipt paths directly; the coordinator
owns collection, the step index and the training receipts. The trainer side
(`grpo_task_graph.TaskGraphRollouts`) is described in
[CONTEXT.md](CONTEXT.md), and the probe that exercises it in
[task-graph training](../../docs/task-graph-training.md).

## Runtime boundary

`GroupCoordinatorV1` requires a `RolloutEnvironment`. It has no bare-store, branch or workspace-restore start path; group members begin and resume only through the verified environment.

Real collection binds the result to the verified published member head. `collect_invalid`
also refuses to relabel a valid terminal outcome as infrastructure-invalid. Group session
seals are checked before member publication, so a manifest mismatch cannot leave a started
head behind.
