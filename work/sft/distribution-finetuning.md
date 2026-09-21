# Distribution fine-tuning: matching human writing instead of a good answer

Reviewed 2026-09-21. Source: [Fixing LLM writing with Distribution Fine Tuning](https://deftwriting.com/research/distribution-fine-tuning)
(Rosmine, May 2026). This records what the technique does, what it implies for our data
construction, and what it does not settle.

## What the technique is

Not a new loss. **DFT is a data-construction and conditioning discipline**, and all of
the author's DFT training used LoRA. The diagnosis is the point:

> "SFT focuses on training individual samples and in doing so misses out on distribution
> level information. DFT trains at this higher level, optimizing the distribution of
> outputs so that it better matches the training data."

The construction, in order:

1. Take **human documents** (a fineweb subset), and strip boilerplate line by line.
2. **Recover the conditioning**: synthesize the prompt that would have produced this
   document, plus its use case and style, plus an outline including the facts and quotes
   that appear in it.
3. Condition each sample on: target length in tokens, an **em-dash-allowed flag**, use
   case, style, the prompt, and the outline. A quarter of the samples have *no* outline.
4. Train on the human document as the target.

The extra conditioning fields are, in the author's words, "not required for the DFT
algorithm to work, they're just there to force people to think more." That is the tell:
**the conditioning vector is the constraint channel.** Explicit, complete conditioning is
how "match the human distribution" stays appropriate to the situation instead of drifting
into randomness.

## Why this dissolves the widen-versus-constrain tension

A natural objection is that widening the distribution trades against context fit. The
reported numbers say otherwise: DFT widened the output distribution **and** became more
context-sensitive.

| Dimension (14B, judge win rate) | SFT baseline | DFT |
|---|---:|---:|
| Prompt relevance | 44.0 | **75.0** |
| Depth | 35.5 | **87.5** |
| Creativity | 32.5 | 86.0 |
| Coherence | 54.5 | 70.0 |
| Clarity | 70.5 | 82.0 |

SFT is **simultaneously too narrow and too generic**. At temperature 0.7, 53.3% of its
outputs contain three or more consecutive sentences starting with the same word, against
17.4% for human text (DFT: 18.6%) — repetitive. And it scores 35.5 on depth — generic.
Both symptoms have one cause: it emits the highest-probability continuation instead of
sampling from the distribution of appropriate responses.

So "widen the distribution" and "understand the context" are not opposing constraints.
The constraint is not a narrower output distribution; it is **conditioning**. Given a
complete description of the situation, the output should be drawn from the human
distribution *for that situation* rather than from one global average mode.

Two further measurements support the distribution-level framing:

- **Common tokens carry the signal.** For a 14B SFT model at temperature 0.8, the top ten
  tokens account for **87.2%** of the squared L2 distance from human text, and they are
  ordinary ones: `" the"`, `"."`, `" is"`, `" The"`, `" a"`, `" was"`. Slop is largely
  overused function words and sentence shapes, not exotic vocabulary.
- **BLEU is the wrong metric here.** The author's higher-BLEU baseline turns out to
  overuse common 3-grams (`be used to` 621 vs 55). Metrics that reward overlap with a
  single reference reward exactly the genericness being removed.

| Scale (14B) | MMD ↓ | JMQ ↑ | Token L2 ↓ |
|---|---:|---:|---:|
| SFT superbaseline | 0.037 | 0.49 | 0.0039 |
| DFT | **0.018** | **0.80** | **0.0036** |

A 4B DFT model beats a 14B superbaseline on MMD and an 8B superbaseline on JMQ, where the
superbaseline is a best-over-all-hyperparameters bound and therefore overoptimistic.
Self-BLEU: human reference 0.061, 14B SFT at temperature 0.9 0.080, 14B DFT 0.064.

## What we already have

The metric side is largely built. `prose.py` implements two of the three primary measures
and one of the diversity guards, and the third is stubbed in the backlog.

| Their metric | Ours |
|---|---|
| N-gram token distribution L2 | **D1** — implemented |
| MMD over embeddings | **D2** — implemented |
| Self-BLEU across prompts | **D4** — implemented |
| Judge model quality (JMQ) | stubbed as "model judging" in [evaluation experiments](../research-plan/evaluation-experiments.md) |
| Within-prompt dispersion | D6 |
| Repetition and cross-output overlap | D11 |

`references.load_matched_references` already binds frozen human reference texts and
`data/references/pilot-matched-v1.json` holds a first set. The
[prose metric catalog](../custom-eval-suite/prose-metrics.md) is the owning document for
what each of these means.

## What we do not have, which is the whole technique

**Every training target we produce is model-generated.** The generator emits conditioning,
an LLM writes the output, and that output becomes the target. This is the
[self-consumption risk](multi-turn-rl.md) in its most direct form: we inherit our own
model's distribution and amplify it, including its tells.

DFT inverts the construction, and the inversion fits us closely because we are already
half way there — we supply human text as *context*. What is missing is that the *target*
is human too.

```
Today:  synthesized conditioning + human context  ->  model output    (target)
DFT:    recovered conditioning   + human context  ->  human document  (target)
```

Concretely, chapter N of a public-domain novel becomes one training sample where the
conditioning is chapters 1..N-1 as project state, plus a recovered brief, outline,
register and tells policy, and the target is the chapter a human actually wrote.
*Frankenstein* alone has 28 sections; a public-domain fiction corpus has thousands. The
samples are free, human, uncontaminated, and contain no model output anywhere in the loop.

This also supplies the missing corpus for the previous open question in
[information value](information-value.md): reference documents for KB and planning
artifacts, where prose corpora do not exist and real-world companion wikis are the
closest analogue.

## The conditioning vector is the interface

The fields that matter, and where each comes from:

| Field | Source |
|---|---|
| Project state (files, manuscript, KB) | our scenario format |
| Artifact type (prose / KB / plan / mixed) | [long-form suite](../custom-eval-suite/longform-suite.md), F1-F5 |
| Register appropriate to that artifact | the same |
| Outline, or deliberately no outline | DFT's quarter-empty convention |
| Target length | DFT |
| **Tells policy** (em-dashes and the rest) | DFT's em-dash flag crossed with the anti-slop tells catalog (`.pi/skills/anti-slop/SKILL.md`) |
| Use case and style | DFT |

Two consequences worth keeping in view:

- **The tells policy becomes a conditioning field rather than a hope.** The em-dash flag is
  the cheapest high-signal field in the set, and it generalises to the whole anti-slop
  list. Asking for a register explicitly and then scoring it is a different and much more
  reliable thing than hoping a holistic quality score suppresses a habit.
- **Artifact-conditioned scoring is how the conditioning gets checked.** A sample
  conditioned as a KB page must be scored on KB dimensions, not on prose quality. That is
  the requirement recorded for the reward in [task generation](rl-task-generation.md):
  "do not score a KB as if it were a literary scene."

## Caveats

- **DFT is single-turn and prompt-conditioned.** Our agent is multi-turn and tool-using
  over a workspace. Trajectory-level distribution is not addressed by this work at all.
- **Their demo is blogs and news, not fiction.** The author states it directly: the demo
  models are "unlikely to do as well at creative writing, since it has not been trained for
  that use case yet." We would be running the unvalidated variant.
- **JMQ is biased in the direction that flatters us.** Judge models prefer LLM output and
  especially their own; the author cites this against his own metric. Our DeepSeek judge
  inherits the same bias, and so does any judge we add.
- **DFT is not uniformly better on every tell.** Its non-English-character rate is 8.1%,
  against 1% for SFT at temperature 0.7 and 36.5% at 1.0. It is better than SFT *at the
  sampling setting where SFT matches DFT's diversity*, not at every setting.
- **Matching metrics is not the goal.** The author added seven further metrics
  specifically to check DFT was not overfitting MMD and JMQ. We would need the same
  discipline, and our external anchor is the natural partner for it: prose metrics moving
  while EQ-Bench does not means we trained the measurement.
- **Their samples are blog length.** DFT at our 8K-64K contexts is unmeasured. See the
  [compute plan](local-compute-and-tracking.md).
- **Two imitation-learning alternatives performed poorly** at their scale (TextGAIL failed
  past 64 tokens; IQLearn beat SFT on some metrics at some settings but beat no super
  baseline). Data construction is doing the work here, not a cleverer objective.

## Next step, and it is cheap

**Measure how far our current model already is from the human distribution, before
building anything.** `prose.score_prose` already records D1, D2 and D4, and
`references.load_matched_references` already binds human references. Comparing Gemma
E2B-IT prose against the Gutenberg reference distribution costs local GPU time and no API
budget, and it produces one number we can re-measure after every subsequent change.

That result decides the priority of everything above it:

- **Large gap** — DFT-style construction is the highest-leverage change available, ahead of
  the reward work, because a better reward applied to a model that cannot produce
  human-like text will not close the gap.
- **Small gap** — the distribution is not our problem, and effort belongs in the reward
  and task-distribution work.

Record the configuration with the number: tokenizer, embedding model revision, reference
selection and bandwidth, n-gram orders, and sample counts. D1 and D2 are distributional
and need enough samples to be meaningful; D4 needs at least two outputs per prompt.

## Relationship to the rest of the plan

- **[Training distribution axes](training-distribution-axes.md)** owns the task variables.
  Distribution matching constrains the *targets*, not the task matrix.
- **[Task generation](rl-task-generation.md)** owns the reward. Artifact-conditioned
  scoring is the check that conditioning was honoured, and is a prerequisite here.
- **[Multi-turn RL](multi-turn-rl.md)** owns the trajectory-level design this technique
  does not address.
- **[Prose metrics](../custom-eval-suite/prose-metrics.md)** owns the definitions of D1-D13
  and the caveat that they are diagnostics rather than objectives.
- **[External anchor](../external-benchmarks/longform-eqbench.md)** is the independent
  check that distribution improvements correspond to real quality.
