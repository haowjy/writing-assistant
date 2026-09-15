# Training data and task generation: research to implementation

Prepare 100 source-backed training tasks before generating a larger SFT answer set.
Use GLM-5.3 through Reka for task authoring, semantic review, and simulated authors
in separate calls. The user has settled training-use permission. The approved cap is $10 total for generating and reviewing this batch, excluding
SFT answer trajectories. Credentials remain pending; no paid calls have run.

This guide joins the earlier [SFT/RL research](rl-bootstrap-research.md),
[branching-fiction design](rl-task-generation.md), [multi-turn design](multi-turn-rl.md),
and [3090 compute plan](local-compute-and-tracking.md). The recommendations below
are our design choices, not claims that any paper validated this writing harness.

The [writing-reward data review](writing-reward-data.md) identifies public preference
and rating datasets for judge calibration and separates them from unlabeled prose
references used for distribution metrics.

The [information-value design](information-value.md) covers relevance, useful coverage,
redundancy and structure for planning, discussion, KB pages and nonfiction, with a
separate treatment of narrative contribution in prose.

## Keep the training artifacts distinct

| Artifact | Contents | Used for |
|---|---|---|
| Source packet | Original text, source spans, cutoff, provenance, work group and split | Grounding generation and checking its claims |
| Training task | Request, starting workspace, branch rules, available tools, stages, private evidence and reward criteria | Sampling attempts for RL; requesting demonstrations for SFT |
| SFT demonstration | Successful assistant conversation and executed tool trace | Supervised updates on assistant actions |
| RL rollout | Current-policy actions, observations, file changes, probabilities, rewards | Policy updates from sampled outcomes |
| Judge record | Rubric, evidence, score, explanation, uncertainty and judge identity | Auditing or eventually training a reward model |
| Evaluation case | Frozen held-out source/project, request and private labels | Measuring generalization without optimizing against these examples |

Task generation therefore does not require writing an ideal scene for every request.
For F2 revision and F5 KB-based writing, it does require usable starting drafts or KBs.
Those generated inputs must be checked separately; the candidate's answer cannot
become the reference truth for its own evaluation.

## What the research supports

| Evidence | Method relevant here | Application and limit |
|---|---|---|
| [Self-Instruct](https://arxiv.org/abs/2212.10560) | Generates instructions and instances, then filters invalid and similar examples before SFT. | Separate generation from admission; retain rejection reasons. Generic instruction diversity does not establish narrative grounding. |
| [AgentInstruct](https://www.microsoft.com/en-us/research/publication/agentinstruct-toward-generative-teaching-with-agentic-flows/) | Uses raw documents and code as seeds for synthetic prompts and answers across skills including editing, writing and tools. | Closest pattern for source packet → skill-conditioned tasks. Its large post-training study does not establish the right size for our pilot. |
| [Magpie](https://github.com/magpie-align/magpie) | Synthesizes alignment instructions from instruction-tuned models using their chat-template prefixes, with subsequent filtering. | Useful evidence for instruction diversity; its prefix-generation technique is not equivalent to a normal hosted chat request and does not supply our story evidence. |
| [LIMA](https://arxiv.org/abs/2305.11206) | Demonstrates adaptation using 1,000 curated SFT examples. | Prefer useful, varied demonstrations over arbitrary volume; not evidence that our 24 drafts are sufficient. |
| [DeepSeek-R1](https://arxiv.org/html/2501.12948v1) | Compares RL without a preliminary SFT stage with a pipeline that includes cold-start data. | Keep IT → RL and IT → short SFT → RL as alternatives. Reasoning results do not settle creative-writing behavior. |
| [Writing-Zero](https://arxiv.org/html/2506.00103v1) | Uses pairwise writing judgments for writer RL; judge preparation is a separate substantial data effort. | Writer training need not require a large SFT corpus. API judging can bootstrap our reward work, but reward quality still needs testing. |
| [RLMR](https://arxiv.org/html/2508.18642v1) | Combines writing-quality rewards and constraint verification. | Keep prose quality and task compliance as separate reward components. File existence alone is not literary success. |
| [Rewarding Creativity / RLCS](https://arxiv.org/html/2601.07149v1) | Prepares a story judge using expert preferences and filtered model explanations before writer RL. | Review the judge itself; fluent explanations do not prove correct preferences. See the earlier research for reported agreement and scale. |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | Uses model critiques/revisions and AI preferences in a staged training pipeline. | Teacher calls can create review and preference data; learning a smaller judge later could amortize cost. Its target behavior differs from fiction. |
| [AgentGym](https://agentgym.github.io/) and [AgentGym-RL](https://arxiv.org/abs/2509.08755) | Separate interactive environments, agents, and training; support stateful multi-turn learning. | Reuse the environment/reset/step boundary. These systems do not provide our source interpretation, prose rewards, or Markdown project design. |
| [MUA-RL](https://arxiv.org/html/2508.18669v1) | Studies RL with simulated users, tool interaction, and user/tool tokens excluded from policy actions. | A simulator can produce follow-ups without recorded human conversations. Its task distribution and reward design need adaptation. |
| [EigenData](https://arxiv.org/html/2601.22607v1) | Investigates user simulation and its effect on training. | Simulator behavior is a measured component, not interchangeable background text. |
| [CompactionRL](https://arxiv.org/html/2607.05378v1) | Trains context compaction within long agent trajectories. | Summaries can become policy actions with delayed consequences. Our first short tasks do not require learned compaction. |

The synthesis is source-backed task generation plus independent checks, followed by
real harness execution. It is not unrestricted self-generated text fed back into
training. No cited result establishes an optimum of 100 tasks; that is our bounded
first collection for inspecting coverage and failure modes.

## The generation pipeline

1. **Freeze sources and splits.** Group works, authors where known, and derivatives
   before sampling. Preserve upstream held-out roles. The initial manifest selects
   five human source works already assigned to train: one Gutenberg opening and
   four Tell Me a Story stories. The old synthetic projects are not inputs to this
   batch. Anonymous authors and unrecorded semantic overlap remain limitations.
2. **Prepare source packets.** Retain exact text, source identity and hashes. A cited
   claim needs a supporting quote or span; a quote being present does not prove the
   interpretation. Keep explicit knowledge, character beliefs, plans, and unknowns
   separate. A short passage supports local tasks, not claims about an entire book.
3. **Assign coverage before asking the model.** Select family, transformation,
   specificity, style, delivery, and stage sequence. Treat these as generation
   assignments, then inspect what the generated tasks actually realize. Avoid
   forcing mutually incompatible combinations just to fill cells.
4. **Generate the task and evidence together.** Give the generator the source and
   the [task-authoring instructions](task-generator-instructions.md). Produce a
   visible request/workspace, declared branch changes, private source evidence,
   and applicable scoring questions. Never ask it to invent source facts to make
   its preferred plot gradeable.
5. **Check mechanically.** Validate structure, safe paths, supported tools, source
   hashes, quoted spans, required starting artifacts and duplicate tasks. Check
   links and reachability in supplied KBs. Check visible/private separation using
   the existing scenario compiler; keep unaccepted candidates outside training.
6. **Review semantically in a fresh call.** Check whether the task is possible,
   natural, grounded and useful. Check that private criteria do not demand an
   unstated plot, style or exhaustive KB. Review source interpretation, purposeful
   detail selection, and whether allowed alternatives would pass. A second call
   to the same model is procedural separation, not independent model bias.
7. **Exercise the workspace.** Verify initial files and scripted turns work with
   the actual harness. For SFT answers, execute the teacher's tool calls and retain
   real observations. Never accept invented success logs. For RL, freeze the same
   task and initial state for every member of a rollout group.
8. **Admit, revise or reject.** Save reasons, revisions, model/provider identity,
   request/response hashes, usage and costs. A schema-valid task remains pending
   until content review passes. Report requested, generated, rejected, and accepted
   counts separately. Replacements need their own generation accounting.

Successful demonstrations follow task admission where they address a measured
policy weakness. Fresh RL attempts come from the current policy rather than a
static teacher-answer corpus. This ordering supports both training methods.

## The first 100 assignments

Use 20 primary tasks per family. Of the 100 assignments, 50 have one stage, 25 have
two, and 25 have three. Stage-family counts consequently exceed 100. Instructions
mix loose and explicit requests. Follow-ups should change direction, add a request,
or ask for a revision without falsely asserting that an unseen answer made an error.
Adaptive criticism requires a later simulator that reads the actual output.

The manifest samples a [variation catalog](../../data/training/variation-catalog-v1.json)
with 60 genre options, 20 styles, 40 tropes, 32 situations, and 12 continuity challenges.
This catalog supplies combinations, not a requirement that each batch or story cover
every entry. The seeded sampler shuffles options to limit habitual repetition;
the generator must adapt incompatible suggestions and record realized coverage.
Close continuation preserves source genre/style. A genre
adaptation must change meaningful narrative constraints rather than rename a character.
Names inherited from a source normally stay fixed; demographic and naming coverage
requires additional sources or explicit adaptation, not cosmetic duplicates.

The current assignments include 33 two-genre blends, 34 single-genre transformations,
and 33 source-preserving continuations. Each assignment also suggests a trope,
situation and continuity challenge. These are generation inputs, not hidden scoring
requirements. Only choices grounded and made visible in the resulting task become
binding. The 15 book candidates are acquisition ideas, not additional admitted sources.

For example, nautical adventure plus LitRPG could make shipboard experience alter
available skills, with rank creating a conflict over command. Cozy mystery plus
workplace science fiction might make a maintenance log central to a small community's
dispute. The blend should affect choices and consequences; it need not announce itself
in the prose. Both examples are design illustrations, not generated records.

## Coherence and prose quality constrain variety

The five task families classify the assistant's work, independently of genre:

| Family | What the assistant does | Example |
|---|---|---|
| F1 | Writes prose in a reply | Continue a scene under the supplied story constraints |
| F2 | Authors or revises files | Change a confrontation while preserving the ending |
| F3 | Brainstorms and plans | Offer alternative consequences of an accepted plot change |
| F4 | Builds or maintains a KB | Extract useful knowledge, organize navigation, update canon |
| F5 | Writes using a KB | Read project knowledge and create a consistent new scene |

A project may visit several families over many tasks. For a long story, keep a stable
project identity, manuscript, timeline, character knowledge and author decisions in
files. Sample the next task against that current state. Do not roll a new genre at
every turn unless the author requests a change. This project-state progression is
part of the [multi-turn design](multi-turn-rl.md), not implemented by the current
independent request sampler.

Minimizing formulaic writing means reviewing repeated explanations, interchangeable
images, generic dialogue, unearned emotional statements and inconsistencies in context.
It does not require uniformly terse prose or avoiding familiar tropes. Keep these
literary judgments separate from adherence to KB facts and instructions, and support
critiques with excerpts. Retain repetition, lexical diversity, source overlap and
corpus-level distribution metrics as diagnostics. Increasing lexical novelty alone
does not show better prose or faithful continuity.

Example combinations include planning alternatives → writing one branch → recording
accepted decisions; constructing a KB → revising a scene → writing from the updated
KB; and direct prose → editing the result → updating project notes. Distinguish
proposed events from enacted events throughout. A vague request must allow multiple
defensible outcomes, while precise requests can specify paths or protected passages.

Five source works are enough to test this process, not broad literary coverage.
The catalog conservatively joins the four anonymous TMAS works through their shared
author label, so there are two connected lineage groups, not five independent groups.
Moby-Dick, later-book checkpoints, longer continuity, additional languages, and wider
naming/genre distributions require source acquisition and review before expansion.
The current book opening cannot stand in for those conditions.

## Training systems and runtime boundaries

Keep Python functions and editable research scripts as the user interface. The
existing scenario compiler and file harness own workspace execution; task generation
should emit their format rather than introduce another tool protocol. Preserve
native Gemma thinking/tool serialization when sampling the local writer.

The existing SFT path uses Transformers, PEFT, bitsandbytes and TRL. TRL also exposes
[GRPO custom rewards and rollout/environment hooks](https://huggingface.co/docs/trl/grpo_trainer).
Those are integration points, not proof that our native Gemma multi-turn RL loop is
implemented. Keep provider transport, task generation, environment execution and
reward calculation separate so that cloud GPU migration does not change the tasks.

Run the writer locally first. Remote Reka calls can author tasks and score semantic
outcomes without sharing GPU memory with training. Judge and simulated-author prompts
have separate visibility: the judge may see private evidence; the simulated author
gets their goals and conversation, not reward keys. Do not leak the judge's preferred
answer into author feedback. Keep Astra on held-out evaluation, separate from the
training reward role.

Begin with short tasks within measured context limits. Record thinking tokens and
tool observations even when they are not prose or supervised targets. Learned
compaction, multi-GPU infrastructure, and training a smaller reward model remain
later work. W&B is an optional mirror of local scores, critiques and artifacts.

## Current artifacts and remaining work

- [Preparation script](../../scripts/prepare_training_tasks.py) and
  [Python preparation interface](../../src/writing_agent/task_generation.py).
- [Frozen batch manifest](../../data/training/task-generation-v1.json): coverage,
  source IDs, hashes, preferred provider and honest generation status.
- Local [request review](../../data/processed/training-tasks-v1/review.md) and
  [complete source-backed requests](../../data/processed/training-tasks-v1/requests.json).
- [Generation script](../../scripts/generate_training_tasks.py) and
  [admission pipeline](../../src/writing_agent/task_authoring.py): source-bound requests,
  generated-task validation, independent semantic review, saved outcomes and compilation
  into the existing visible/private scenario format. Default execution only inspects.
- [GLM transport and output grader](../../src/writing_agent/openrouter.py): pinned
  OpenRouter/Reka route, shared persistent budget, saved reasoning and usage, hashed
  response cache, and the existing evidence-backed rubric/scorecard interface.
  Task review and assistant-output grading use separate fresh system/user contexts.
- Live API verification and the 100 generated tasks remain pending credentials.
  Scripted tests verify accounting and admission, not provider availability or task quality.
  The prepared requests are not 100 completed training tasks.

Reka's [public direct-API model list](https://docs.reka.ai/chat/models) names its own
Flash and Edge models and allows account-specific availability. GLM-5.3 access is
confirmed through the OpenRouter listing, not through that direct API. Use an
OpenRouter key for the confirmed route; a Reka key requires checking the account's
available models first. Do not infer direct access from OpenRouter provider identity.

Reproduce preparation with `.venv/bin/python -m scripts.prepare_training_tasks`.
This performs no inference, training or paid calls. The source catalog is rebuilt
through the documented [SFT seed workflow](dataset-starter.md) when absent.

Inspect with `.venv/bin/python -m scripts.generate_training_tasks`. To execute the
approved batch, call `generate(execute=True)` from that module. The key belongs in
ignored `.env` as `OPENROUTER_API_KEY`. Calls and their shared ledger live under
`data/processed/training-tasks-v1/paid-calls/`; generated outcomes, review materials
and compiled tasks live in the sibling `generated/` directory. Resume with the same
inputs and call directory. An unresolved transport reservation stops new calls until
its charge is reconciled; never delete the ledger to retry a paid batch.

The cap includes a 5.5% platform-fee allowance and uses routing price ceilings for
preflight reservation. Those ceilings are not a current-price quotation. Rejected or
malformed outputs still consume budget and remain available for inspection. The
requested count is 100 proposals; admitted count can be lower. Revisions require a
new input/output version while retaining the same paid-call ledger and cap.

`GLMGrader(client).grade(packet)` accepts the existing `grading_packet` format;
`apply_judgment` applies its validated result to a scorecard. Invalid or unavailable
judgments leave scores pending. This supplies semantic ratings, not an implemented
RL reward loop or calibrated reward model. No writer rollout, SFT answer generation,
training or existing benchmark grading is launched by the task-generation script.
