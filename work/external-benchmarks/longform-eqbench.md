# External long-form anchor: EQ-Bench Longform Writing

Selected 2026-09-21. Pinned revision `34f60a028c3f973c19cde98dc5a9e8f9875a87e3`.

## Why an external anchor exists

Our own long-form suite is authored from the same taxonomy that produces our training
tasks and our reward. It can show that a policy **moved**; it cannot show that it moved
in a better direction, because the thing being measured was designed by the same head
that designed the target. A held-out split defends against memorisation, not against
shared bias.

This benchmark is the falsifier. It was authored by EQ-Bench, it has a public
leaderboard with real model rankings, and we did not choose its criteria. If our
internal suite improves while this one does not, the internal suite is measuring our
generator's opinion.

## What upstream does

Thirteen steps per prompt, twelve prompts:

| Steps | Content |
|---|---|
| 1–5 | Brainstorm, intention and chapter plan, human-vs-LLM critique, final plan, character profiles |
| 6–13 | Eight chapters of roughly 1000 words each, composing a novella |

Each chapter and the whole piece are judged by Claude Sonnet 4.6 against fourteen
0-20 criteria, with six of them lower-is-better. Weights emphasise forced metaphor
(5×) over purple prose (1×), and the forced-metaphor term is convex
(`(v/20)^1.7 × 20`) so partial metaphor is punished harder than proportionally.

Upstream also reports chapter-length, a "GPT-ism" frequency, n-gram repetition, a
chapter-quality **degradation** figure (last chapter minus first), and a
single-sentence-paragraph penalty that scales chapter scores. Only the last of these
contributes to the score.

## Why this one

- **Genuinely external.** Third-party authored, with a public leaderboard. Our position
  on it is a direction signal our own suite cannot produce.
- **Long-form by construction.** Eight chapters plus planning is exactly the regime our
  training window (2K–8K) does not cover and our inference window (128K) does.
- **Same judge we already run.** Sonnet 4.6 at pinned pricing is already implemented in
  `anthropic_grading.py`, with an approved reservation-ledger protocol.
- **The criteria already overlap our scorecard.** Nuanced characters, compelling plot,
  coherence, purple prose and tell-don't-show are close to our Q-dimensions;
  "followed chapter plan" and "faithful to writing prompt" are our plan adherence and
  intent, and "characters consistent with profile" is our continuity. That gives an
  independent read on whether our reward measures what we think it does.
- **It measures degradation.** First-to-last chapter quality drop is the single most
  relevant external metric for a long-form agent, and we have nothing comparable.

## Adaptation, and its deltas

`longform.py` reimplements the protocol, criteria, weights and arithmetic from the
pinned source. Two modes:

| Mode | Generation | Score |
|---|---|---|
| `faithful` | Reply-only, exactly as upstream | Comparable to the leaderboard |
| `workspace` | Chapters also saved to `manuscript/chapter-N.md` | **An adaptation**, not comparable |

Scoring is identical in both modes; only the prose selector and the delivery step
differ. `workspace` exists because our research is about file-delivering agents, and a
reply-only benchmark does not test that.

Divergences that must be reported with any score:

- **Staccato curve.** Upstream interpolates with the wrong coefficient, so its factor
  falls to 0.4 at an index of 18 and then jumps back to the 0.6 cap. That is a
  discontinuity, not a policy. We implement the continuous curve the upstream comment
  describes; `legacy_curve=True` reproduces the upstream number.
- **Criterion filtering.** We filter parsed metrics to the declared criteria and report
  missing ones. Upstream accepts whatever the judge returns, so a judge that invents a
  metric can move the score. The arithmetic is identical when the judge behaves.
- **Sampling and context.** Upstream generates at `temp=0.7, min_p=0.1` over an API.
  Our local runs use greedy decoding and a bounded context. This changes the number.
- **Slop diagnostics.** Not reimplemented; they do not contribute to the score, and the
  upstream implementation pulls `wordfreq` and parallel workers for a diagnostic.

## Cost and limits

A full run is 12 prompts × 9 judgments = **108 Sonnet 4.6 calls**, roughly $5 in judge
tokens at the pinned rates — above the $2 approved for the Creative Writing v3
baseline. A reduced run (fewer prompts, or `chapters=3`) is much cheaper. **No paid run
is authorized by this document.**

Context is the binding local constraint. Eight chapters of ~1000 words plus plans and
profiles exceed a 16K window; upstream supports `NUM_CHAPTERS`, and
`chapters=3` fits comfortably. Report the chapter count with any score, because it
changes it.

This anchor cannot tell us whether our *agentic* capabilities improved — retrieval,
KB maintenance, protected edits, canonical updates. It scores prose and planning. Those
remain the internal suite's job, with the circularity caveat attached.

## Status

- Acquisition pinned in `acquisition.py` as `eqbench_longform`; fixtures land in
  `data/raw/research/eqbench-longform/`.
- `longform.py` loads the release, renders the 13-step protocol, compiles 12
  `final_eval` scenarios through the unmodified `compile_scenarios` contract,
  reimplements upstream scoring, and parses judge output.
- Compiled releases: `data/processed/longform-benchmark-v1-faithful/` and
  `-workspace/`. Both are gitignored; rebuild from the pinned acquisition.
- No generation, no judging and no paid call has run.
- Upstream declares MIT in its README but ships no LICENSE file.
