# Deterministic task-graph groups

`GroupCoordinatorV1` prepares offline comparison and segment-credit artifacts; it is not a
GRPO trainer. It seals a group around an admitted, unsampled entry checkpoint and a complete
environment/policy contract, starts isolated member lineages through the verified rollout
environment, and collects terminal evidence from verified views. It does not call a model or
update weights. The separate legacy evaluation and SFT APIs are unchanged.

## Seal and start

Construct the coordinator with the same `RolloutEnvironment` used by the members:

```python
coordinator = GroupCoordinatorV1(environment, workers_root)
spec = coordinator.seal(
    entry_checkpoint_id,
    policy=policy,
    group_seed=group_seed,
    group_sequence=group_sequence,
    member_count=member_count,
)
runtime = coordinator.start(spec, ordinal, policy=policy)
```

Seal before starting a member. Every policy field is required and content-addressed:
`model_ref`, `behavior_policy_ref`, `tokenizer_ref`, `template_ref`, `adapter_ref`,
`decoding_ref`, `simulator_ref`, `context_policy_ref`, and `controller_ref`, plus
`rng_derivation_version="sha256-domain-v1"`. Tokenizer and template pins must match the
entry context's rendering. Policy artifacts must already exist. The group identity includes
the entry checkpoint, policy, seed, runner mode, member count, and `group_sequence`; do not
pool separate groups. Use `runner_mode="fixture"` only for scripted fixture results.

Each member has its own lineage, created by `RolloutEnvironment.start_member` from the
shared entry and its sealed `MemberStartV1`. Its initial environment seed is shared; its
writer seed is derived by member ordinal. The coordinator has no bare-store start, branch, or
workspace-materialization path. The `RolloutDriver` gathers typed port inputs and commits one
event per step; see [the writer runtime](task-graph-writer.md). A runner that needs a
materialized execution environment must provide that separately through its execution port.

A start receipt is a cache of verified ancestry: every `start` checks the member view and
re-derives the start checkpoint as the member's first checkpoint below the sealed entry. A
retry after a crash resumes through `environment.open_head(member_id)` and produces the same
receipt, even if the member has since progressed. The receipt is not its own authority.

## Collect and finalize

`collect` and `finalize` validate real members by opening and checking their published head
through the environment and reading outcome, reward, eligibility, samples, and contexts from
the verified view and its ancestry. The gate derives every event against the sealed group and
context-policy pins, so collection does not repeat a policy walk. A fixture result carries a
typed scripted terminal record and is never optimizer-eligible. An infrastructure-invalid
member carries typed failure evidence and no reward. Missing or unavailable reward holds the
group pending; an infrastructure-invalid member invalidates the group; tied rewards yield
explicit zero advantages.

For rewards `rᵢ`, the coordinator stores exact reduced fractions for each reward, mean,
population variance, and centered reward. The declared normalized advantage is
`centered / sqrt(population_variance)`; its expression is kept symbolic because the result may
not be rational. All-tie groups have an exact zero advantage and `status="tie"`; artifacts
contain no floating-point values.

One artifact is emitted for each eligible historical writer text part, tool-call syntax part,
and assistant ending. The writer derive marks whether a sampled part is creditable. An
`invalid_tool_call` part without bounded sampled content carries the exact sentinel
`{"$noncanonical": "no-sampled-content"}` and receives no segment credit. Tool observations,
system/user/author/seed/environment/summary messages and other ineligible parts receive no
credit. Phase 7 segment artifacts set `native_optimizer_eligible=false` and have null token
mask/logprob refs: they are not token-level training targets.

A collected `final_checkpoint_id` currently needs to belong to the member lineage; binding it
to the verified current head and orphan collection remain a HIGH-5 follow-up. Do not treat
this limit as evidence that the checkpoint's lineage history was not verified.

## Runtime boundary

Group members use the same single-event runtime core as other rollouts: typed inputs derive
events and complete next state; the store requires the lineage gate for publication and
restore; the `OutcomeV1` referenced by state holds check, transition, terminal, reward, and
eligibility references. No separate runtime log is used. See
[context operations](task-graph-compaction.md) for the pinned context policy and safe
compaction boundary.
