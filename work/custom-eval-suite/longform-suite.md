# Long-form final evaluation suite

Frozen 2026-09-21. Specification: `data/scenarios/longform-v1.json`.

## What this is

Six held-out cases that test whether an agent can hold a long project together: supply
several chapters of a real novel, ask for more, and check that the result respects what
came before, follows a plan, propagates a revision, and lands an arc. The development
suite tests short single-scenario capability; this tests the regime our training window
(2K–8K) does not reach and only inference (128K) can.

## Read the circularity caveat first

**This suite is not sufficient to support a claim that writing improved.** It is
authored from the same taxonomy as our training tasks, its checks are written by the
same head that wrote the reward, and its prose rubric is the one our reward is modelled
on. A policy can satisfy it while getting worse by any external standard.

It is the *direction* test's local half. The [external EQ-Bench anchor](../external-benchmarks/longform-eqbench.md)
is the half that can falsify. Run them together and report both; if this suite improves
while that one does not, this suite is measuring our own taste.

## The six cases

| ID | Axis | Work | Supplied | Deliverable | Family | Spec. |
|---|---|---|---|---|---|---|
| LF-01 | `chaptered_continuation` | Frankenstein | Letters 1–4, Ch 1 | 3 chapters, files | F2 | L3 |
| LF-02 | `chaptered_continuation` | Alice | Ch I–III | 2 chapters, files | F2 | L0 |
| LF-03 | `sustained_plan` | Frankenstein | Ch 1–5 + outline | 2 chapters, files | F2 | L3 |
| LF-04 | `long_range_revision` | Alice | Ch I–VI | revise all + 1 new, files | F2 | L3 |
| LF-05 | `kb_bootstrapped_novel` | Sherlock Holmes | *A Scandal in Bohemia* | linked KB + new case | F4 | L1 |
| LF-06 | `divergent_branch` | Frankenstein | Ch 1–4 | closing chapter, reply | F1 | L0 |

Specificity is sampled at L0, L1 and L3 rather than swept, so the suite tests the axis's
extremes and one interior point; it is not powered to estimate a slope. Delivery varies
(five file-writing, one reply-only) because the delivery channel is a nuisance axis that
should not move the score.

## How the cases are grounded

Three public-domain works — *Frankenstein*, *Alice's Adventures in Wonderland*, and
*The Adventures of Sherlock Holmes* — are split on their own heading patterns and
supplied as a manuscript the candidate can read and edit.

Every case declares `anchors`: facts the checks may cite. **The build fails if an anchor
does not occur in that case's supplied text.** This is the mechanism that keeps labels
honest — a check cannot cite a fact the candidate was never given, and an author (human
or model) cannot assert one from memory. It caught two real errors while authoring:
`Clerval` in a case that supplies only Frankenstein's letters, and `Justine`/`William`
in a case that stops before they appear.

Supplied files are deterministic slices, so a rebuild reproduces identical bytes and the
frozen hashes stay meaningful.

## Holdout proof

`holdout_audit` refuses to build if a benchmark work is already in use. Checked by
content hash, against the hashes actually claimed rather than everything present in a
catalog — a work imported but never referenced is still free to reserve.

Current result: `held_out`, against 15 training sources and the 50 sources the
development cases reference. The three works *are* present in the custom-eval catalog as
available imports; no development case uses them, which is what makes reserving them
legitimate. If a development case ever adopts one, the build stops.

## Protocol

- **Never** use these cases for checkpoint selection, prompt iteration, or reward
  debugging. Running them to see how a change looks converts them into a development
  set, and the log will not say when that happened.
- Run once at baseline and once at the end, plus any comparison arms needed for a
  reported claim. Record the freeze hash with the result.
- Report completion and semantic scores separately. A missing chapter is a delivery
  failure, not a low prose score.
- Rebuild and re-freeze if the specification changes, and report the new hash.

## What it does not do

- **Statistical power.** Six cases cannot distinguish small effects. Treat differences
  within noise as noise.
- **Breadth.** Three works, all Western public-domain novels. Genre coverage is thin and
  the style is Victorian-to-Edwardian.
- **Human-reviewed labels.** The checks are authored and mechanically anchored, but no
  human has reviewed them, and the prose rubric is inherited from the development suite.
- **Contamination beyond these three works.** Held out against the sources we know
  about, not against everything that might be in a base model's training data.
- **Agentic breadth.** It tests long-range continuity, planning and revision. Retrieval
  coverage, protected edits and canonical updates get only partial coverage.

## Rebuilding

```bash
python scripts/build_longform_suite.py
```

Writes `data/processed/longform-v1/` with the compiled release and a `freeze.json`
carrying the holdout result, per-case visible and label hashes, and a freeze hash.
The compiled release is gitignored; the specification is tracked.

## Status

- Built and frozen. Six cases compile through the unmodified `compile_scenarios`
  contract and reload.
- No candidate has run against it. No paid call has been made.
- `context_target_tokens` per case is 14K–28K, so a run needs an inference window
  substantially larger than the external checks' current 16K setting.
