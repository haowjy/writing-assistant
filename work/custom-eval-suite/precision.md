# Precision is not a detail

The 50-case E2B baseline was generated with nf4 weights. Every comparison built on it
therefore carried an unquantified confound: nothing separated capability from 4-bit weight
approximation. E2B fits in bf16 on this device (9.6 GB), so the confound was measurable
rather than permanent, and measuring it changed the answer in an inconvenient direction.

## What we did

One prompt (`F1-01`), twenty-five attempts, twice: once in nf4 and once in bf16. Same
model revision, same seeds 4200–4224, same generation settings, one field flipped. Because
the seeds match, the two arms are paired attempt by attempt. Each arm ran all 25 attempts in
one process, nf4 in 16m04s and bf16 in 11m47s.

Twenty-five attempts is the count that clears the sample floors described in
[the long-form suite](longform-suite.md#sampling-and-the-measurement-this-suite-cannot-make):
`D1`, `D4` and `D6` report at full power, `D2` at low power because its reliable count is
fifty.

## What we found

| Measure | nf4 | bf16 | Δ | Reads as |
|---|---|---|---|---|
| D1 n-gram L2 (n=1) | 0.1013 | 0.1006 | −0.0008 | flat |
| D1 n-gram L2 (n=2) | 0.0461 | 0.0514 | +0.0053 | further from human |
| D1 n-gram L2 (n=3) | 0.0334 | 0.0385 | +0.0051 | further from human |
| D2 MMD | 0.3006 | 0.3180 | +0.0174 | further from human |
| **D4 self-BLEU** | 0.6131 | **0.7068** | **+0.0937** | **more alike each other** |
| **D6 dispersion** | 0.0896 | **0.0709** | **−0.0187** | **tighter cluster** |
| D11 cross-Jaccard | 0.0662 | 0.0991 | +0.0329 | more shared trigrams |
| D11 duplicate rate | 0.0000 | 0.0000 | 0 | no exact repeats in either arm |

**bf16 is less diverse, not more.** Four independent measures agree on the direction:
self-BLEU rises, dispersion falls, cross-Jaccard rises, and both n-gram distance and MMD
move slightly further from the human reference. The only measure that is flat is unigram
n-gram distance.

A paired bootstrap over attempts (400 resamples) puts the shift beyond resampling noise:
`D4` mean Δ `+0.0371`, 95% interval `[+0.0142, +0.0611]`, `P(Δ>0) = 1.000`; `D6` mean Δ
`−0.0181`, 95% interval `[−0.0290, −0.0073]`, `P(Δ>0) = 0.000`. The bootstrap distributes
slightly differently from the realized difference (`+0.0937`) because resampling with
replacement repeats texts and self-BLEU rewards exact repeats; the realized difference is
the number to quote, and the interval is a check that it is not noise.

## Why this matters

Quantization noise perturbs sampling, and that perturbation shows up as extra variance. So
a 4-bit measurement makes the model look **more diverse than it is** — which is the
flattering direction. Any diversity or distribution claim measured on a quantized arm is
inflated by an unknown amount.

This is the same failure the rest of this session was about: a number that reads as
meaningful while the evidence underneath does not support it. We would have reported a
more diverse model, and the error would have been invisible, because nothing in the
artifact says which precision produced it.

It also sharpens the substantive picture. In bf16, self-BLEU is **0.71** and dispersion
**0.071**. Repeating one short prompt twenty-five times yields outputs that are closely
alike. That is consistent with the repetition the
[distribution-fine-tuning](../sft/distribution-finetuning.md) evidence reports
(53.3% repeated sentence openings against 17.4% for human prose), and it is the baseline a
training run has to move.

## A side effect worth knowing

**bf16 is about 1.4× faster than nf4 here** — 27.7s per attempt against 39.7s, medians 28.0
and 39.4 — while using 11.6 GB against 8.4 GB. Completion lengths are near-identical (733 and
759 tokens), so this is throughput rather than bf16 simply writing less. An earlier draft of
this section claimed roughly twice as fast, from polling attempt counts while the runs were in
flight; the realized latencies do not support that, and the per-attempt figure is the one to
quote.

For a model this small on a 24 GB card, the quantized path bought memory we were not short of
and cost both fidelity and wall time. Quantization is for fitting a model that does not
otherwise fit.

## Still open

- The 50-case rerun in bf16 is in progress to move the deterministic baseline metrics
  (`Q1`, `Q3`, `Q4`, `Q13`, `R1`) onto the faithful arm. `scripts/rerun_precision.py` flips
  only the precision field and carries the context limit over verbatim, because a rerun
  that also moves the window would attribute two changes to one cause.
- **The judged metrics are the gap.** `Q2` prose quality (2.20) and the semantic rubrics
  need a paid grading pass on the bf16 arm, and the `$2` Sonnet 4.6 approval was scoped to
  `CWv3`. Until that runs, the *quality* half of the confound is unmeasured; the
  distribution half, above, is measured and adverse.

## Reproducing

```
uv run python scripts/baseline_distribution.py --execute --quantization nf4 --repeats 25
uv run python scripts/baseline_distribution.py --execute --quantization bf16 --repeats 25
uv run python scripts/baseline_distribution.py --collect --quantization bf16
```

Results are in `runs/distribution-e2b-it-2026-09-21-{nf4,bf16}/`. They are not tracked,
because `runs/` holds generated artifacts.
