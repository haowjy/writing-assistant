# Composed writing sessions and compaction-aware RL

Proposed design, researched 2026-09-15. Task families describe capabilities; they are
not mutually exclusive conversation types. A writing project can combine planning,
KB construction, drafting, correction, and revision in one bounded training session.
The [task generator](rl-task-generation.md) should compose these stages around a
shared project state rather than merely concatenate unrelated prompts.

## Existing strategies

[MUA-RL (2025)](https://arxiv.org/html/2508.18669v1) puts simulated users and real tool
execution inside multi-turn GRPO rollouts. It uses terminal task-completion rewards
and masks user/tool-observation tokens. Its reported RL setup caps sessions at 30
interaction turns and 32,768 tokens. It supports the approach, not a claim of realistic
creative-writing simulation; its tasks are operational tool-use scenarios.

[EigenData (2026)](https://arxiv.org/html/2601.22607v1) studies synthetic multi-turn
tool-use data and verifiable-reward RL, including simulator-related instability.
Synthetic users are a trainable/testable environment component, not free realism.

[CompactionRL (July 2026)](https://arxiv.org/html/2607.05378v1) makes summarization a
policy action and optimizes summaries and execution using the final outcome. It
reconstructs context from summaries and recent turns, and uses PPO with adjustments
for credit across segments. Its evidence concerns coding agents at much larger scale;
our Gemma/3090 integration would be a separate experiment. Compaction is not a
training feature supplied merely by shortening an evaluation transcript.

## Generate interactions without a prerecorded conversation

Start with a source-backed project state, author goal, and a plan for possible next
requests. The writer acts; real workspace tools update files. A controller then selects
a follow-up that is valid for the actual output and project state. The session itself
becomes the generated conversation data.

Use three feedback sources, labeled separately:

- **Scripted author decisions:** choose an option, request a new scene, change a style
  preference, or approve only part of a proposal. Resolve references against actual
  generated options rather than assuming an option always exists.
- **Verified error feedback:** a failed check identifies the wrong path, broken link,
  or protected edit. A semantic error requires supporting source evidence before a
  simulated user asserts it occurred. If no error occurred, do not invent one to
  satisfy a predetermined correction turn.
- **Simulated author reactions:** a permitted model receives the author's goal,
  interaction style, visible output, and relevant state. It can ask questions, express
  preferences, clarify a request, or continue the project. Its approval is not itself
  the reward or proof of correctness.

Also construct recovery tasks from deliberately flawed starting drafts, recording
that the error was inserted by the environment. These teach editing without requiring
the current writer to make the error first. Distinguish factual corrections from a
new preference or changed premise. Do not teach automatic agreement with unsupported
accusations or treat reasonable alternatives to a vague request as factual errors.

Validate simulator consistency and grounding, and freeze its version during an
experiment. It must not reveal private rubrics or reward labels in its feedback.
The writer's learning targets/actions exclude simulator messages and tool observations.

## Example composed session

1. Ask for three directions for a Moby-Dick LitRPG adaptation at a supplied cutoff.
2. Select one of the actual proposals and request a navigable project KB.
3. Ask for a scene based on that KB, saved to a manuscript file.
4. Identify a verified problem, or request an explicit change of preference, then
   ask for a local revision with unrelated text protected.
5. Accept specified developments and update the KB before drafting the next scene.

This is a possible task graph, not a fixed five-turn transcript. Clarifications,
repairs, rejection, and early completion can create different paths. Record session
ID, stage IDs, all applicable families, dependencies, requirement versions, acceptance
history, snapshots, simulator identity, and feedback provenance. Keep one primary
family only for compatibility with existing reports; add multi-family coverage when
implementing composed records. The current dataset schema has not been changed here.

Sample bounded sessions from an ongoing stream of source checkpoints and branch
variations. Begin with two or three connected stages, then expand to five when runtime
and reward behavior are understood. An unlimited supply of sessions does not require
one infinitely long rollout. Resuming a saved project later must record its origin,
policy version, and state; do not silently reuse stale action probabilities.

## Reward and credit

Keep stage scores and final project checks. A proposed mixed reward uses a normalized
combination of applicable stage outcomes and final-state consistency; weights remain
to be selected and tested against a terminal-only baseline. A final average alone can
hide a failed crucial stage, while an all-or-nothing score can be too sparse.

Check persistence: a wiki still works after a later edit, accepted facts remain
consistent with the manuscript, and unrelated material survives. When the author
changes a requirement, mark the old requirement superseded instead of scoring both
as simultaneously binding. Count an outcome once; repeated checks are diagnostics,
not repeated opportunities to collect reward.

Save initial and revised quality separately. Do not reward improvement alone: the
policy could create bad first drafts to earn repair credit. Author-simulator praise
and the number of turns/files/tool calls are not evidence of successful collaboration.
Learning must distinguish helpful recovery from avoidable damage.

For group-relative RL, share the initial state, author goal, task policy, budgets, and
simulator configuration across a group. Feedback may correctly differ after different
writer actions. Keep follow-up selection constrained and seeded where possible to
reduce simulator noise without forcing identical conversations.

## Context and compaction

Three limits currently differ:

| Limit | Current evidence |
|---|---|
| Gemma E2B architectural context | 131,072 tokens in cached model config; Google documents 128K |
| Main evaluation/pilot configuration | 8,192 total context tokens, reserving up to 2,048 generation tokens per call |
| SFT preparation | 2,048 tokens per complete prepared trajectory; overlength records fail |
| RL training context on this 3090 | Not measured; no RL integration or compaction implemented |

The [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4) is the model
limit reference. Local evidence is `SFTSettings.max_length`, the configurations in
`scripts/evaluate.py` and `scripts/pilot_e2b.py`, and the input-plus-output budget check
in `inference.py`. Other benchmark scripts can choose different inference budgets.
Thinking, tool schemas, observations, source text, and output all consume context.
Advertised inference context does not establish a practical training memory budget.

Proposed memory handling: preserve project files, accepted decisions and source evidence
outside the conversation; compact older interaction into a bounded working summary;
retain recent turns and the current request. Allow retrieval of older material from
an archive with the same access rules. Canonical author decisions must not be replaced
by an unchecked summary or by the candidate's own KB assertions.

Start with a fixed context-management policy and test continuation across a controlled
compaction boundary. Later compare policy-generated summaries trained on downstream
success. A fixed external summarizer is environment behavior; it is not being trained
just because the writer learns from the resulting context.

Log the exact context and sampled actions for every segment, plus compaction triggers,
summary provenance, state versions, and policy probabilities. Training must evaluate
action likelihoods against the context actually used for sampling. Do not replace old
contexts with a later summary, flatten compacted segments into an impossible transcript,
or treat user/tool observations as policy actions. Delayed credit must cross the
compaction boundary when earlier actions affect later outcomes.

Compaction limits the active window, not total rollout time, storage, or all training
costs. Measure training at the current 2K cap, then probe 4K and 8K as targets subject
to memory/throughput results. Do not promise 128K training or assume an 8K rollout can
be optimized intact by the present 2K SFT exporter. Segment-aware RL is new work.

The [local compute plan](local-compute-and-tracking.md) records the decision to test
the 3090 first, defer rentals, and add optional qualitative/quantitative W&B tracking.

## Next work

Specify the composed-session schema and grounded follow-up controller; validate a
short mixed-family session with stage snapshots and final-state checks. Then measure
the bounded training memory budget, implement one reproducible compaction boundary,
and compare downstream success with full-context and fixed-summary controls. A learned
summarizer and long sessions follow only after those components are reliable.
