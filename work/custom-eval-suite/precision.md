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

## The 50-case rerun, and what it could not settle

The same 50 cases were regenerated in bf16 — same seeds, same generation settings, same
context limit, one field flipped. Both arms were then scored identically, using only the
deterministic metrics, so nothing further was paid for. Paired on the cases where both
arms produced the metric:

| Metric | Paired | nf4 | bf16 | Δ | nf4 scored | bf16 scored |
|---|---|---|---|---|---|---|
| Q1 instruction adherence | 9 | 0.7407 | 0.8519 | **+0.1111** | 9 | 9 |
| Q3 task completion | 13 | 0.5385 | 0.5385 | 0.0000 | 17 | 13 |
| Q4 tool correctness | 24 | 1.0000 | 1.0000 | 0.0000 | 24 | 24 |
| Q13 continuity | 0 | — | — | — | 0 | 0 |
| R1 latency (seconds) | 49 | 82.376 | 57.790 | **−24.586** | 49 | 49 |

**The one solid result is latency.** On the real 50-case workload bf16 is 30% faster,
paired across 49 cases. That corroborates the single-prompt measurement above and makes the
same point: the quantized path was slower as well as less faithful.

**The quality half is not settled, and this comparison is not able to settle it.** The
paired counts are 9, 13 and 24 against 50 cases, and the reason is structural: 115 of the
development checks are `semantic` and stay `pending` until a judge runs, so a case only
resolves a metric when all of its checks happen to be deterministic. Comparing means over
different case subsets would have produced a number for Q1 and Q3 that looks like a result
and is not one — the first version of this comparison did exactly that, and reported Q3 as
+0.127 from 17 nf4 cases against 13 bf16 cases. The numbers above are paired on the
intersection of the two subsets instead.

Q1's +0.111 across 9 cases is suggestive and nothing more. It is not comparable to the
published baseline's Q1 of 0.794, because that figure comes from a graded run where the
semantic checks had resolved, and so covers a different and much larger case set. **The
`Q2` prose-quality figure of 2.20 and the semantic rubrics remain unmeasured on the bf16
arm**, and they are where a precision difference would show up if it affects writing rather
than throughput.

## Still open

- The 50-case rerun in bf16 is done, and it settled throughput while failing to settle
  quality; see above. The regeneration itself is complete and its cards are scored.
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
