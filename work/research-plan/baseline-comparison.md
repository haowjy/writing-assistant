# Gemma baseline comparison

The next experiment compares 12B and E4B pretrained and instruction-tuned
checkpoints, then selected community derivatives, before training our own model.
The target is Gemma 4 on the local RTX 3090, operated through the terminal.

## Checkpoints

The four official repositories are
[12B base](https://huggingface.co/google/gemma-4-12B),
[12B IT](https://huggingface.co/google/gemma-4-12B-it),
[E4B base](https://huggingface.co/google/gemma-4-E4B), and
[E4B IT](https://huggingface.co/google/gemma-4-E4B-it).

## Hardware

The machine has one RTX 3090 with 24,576 MiB VRAM. A host-side check on
2026-09-11 reported 23,626 MiB free. GPU inspection worked outside the execution
sandbox; available memory will vary.

Run one checkpoint at a time with a short-context, single-request pilot.
The official [12B weights](https://huggingface.co/google/gemma-4-12B/tree/main)
occupy about 23.9 GB before runtime buffers and KV cache, so plan a quantized
12B pilot. The [E4B weights](https://huggingface.co/google/gemma-4-E4B/tree/main)
occupy about 16 GB; test BF16 at short context before assuming workload fit.
File sizes are planning estimates, not measured GPU allocations.

Pilot a common quantization method and precision for the main four-model
comparison, with any E4B BF16 run reported separately. Verify runtime support
before selecting the format; measure speed before increasing context or batching.

## Evaluation sequence

1. Define the first metric specification and author a small development set.
   Include expected facts, required/forbidden/deferred beats, edit boundaries,
   and accepted state changes. Use this set to debug prompting and scoring.
2. Add writing-only generation alongside the existing agent loop. Evaluate
   continuation with the same prose prefix across base and IT models; separately
   evaluate instruction-driven writing and full conversations. Record each
   rendered prompt, template, and any few-shot examples. Base models need an
   explicit completion protocol; a missing chat template is a harness issue.
3. Run a small integration pilot on all four checkpoints. Verify completion
   extraction, stopping, tool parsing, and score inputs by reading actual outputs.
4. Freeze an independently authored held-out suite. A starting proposal is
   50–100 tasks across writing, revision, critique, brainstorming, and stateful
   collaboration, with repeated samples on a writing subset. Estimate runtime
   from the pilot before fixing sample counts.
5. Run the official checkpoint matrix, then shortlisted derivatives under the
   same protocol. Keep prompt tuning and candidate selection on development
   data. Report paired differences and uncertainty across held-out tasks.

## First scorecard

| Area | Initial measures | Work needed |
|---|---|---|
| Agent behavior | Constraint violations, state fidelity, edit preservation, tool validity, completion rate | Extend literal checks with task labels and reviewed semantic judgments |
| Writing | Blinded pairwise preference and rubric scores for coherence, characterization, pacing, dialogue, and redundancy | Fix judge version/prompt; randomize order and calibrate against human ratings |
| Distribution | N-gram L2, embedding MMD, diversity across repeated samples | Specify reference corpus, tokenizer, normalization, embedding model, kernel, estimator, and diversity function |
| Runtime | Wall time per task, generated tokens, throughput, peak GPU memory, failure/truncation rates | Add measurements; distinguish generation from scoring and server startup |

Keep these dimensions separate. Distributional similarity is a diagnostic;
it needs comparison with human judgments before it can support quality claims.
The current five scripted fixtures only test the harness. See the
[metric candidates](evaluation-experiments.md) for the broader research scope.

Record checkpoint and tokenizer revisions, inference-library versions, GPU,
precision/quantization, context limits, sampling settings, thinking mode where
applicable, and token/tool budgets. Count reasoning tokens in the generation
budget. Use comparable precision for the main comparison and label quantized
runs separately. Compare speed at matched workload and concurrency.

## Community candidates

Search model cards and lineage for actual writing/roleplay training. Distinguish
fine-tunes, merges, weight edits, and quantization-only conversions. Capture the
parent revision, data disclosure, training method, template, and available
weights; unknown training data leaves benchmark overlap uncertain.

Initial Gemma 4 leads found on 2026-09-11:

- [Gryphe/Gemma-4-12B-StyleTune](https://huggingface.co/Gryphe/Gemma-4-12B-StyleTune):
  the author describes narrative training of the output projection. Test its
  prose changes and instruction following independently; capability preservation
  is an author claim, not established evidence for this project.
- [andyoneal/Gemma-4-E4B-Nightcap](https://huggingface.co/andyoneal/Gemma-4-E4B-Nightcap):
  a documented merge of three writing/roleplay derivatives. Its parent list is
  also a source for finding individual fine-tunes to test before adding merges.

These are discovery leads, not evaluated recommendations. Verify lineage and
runtime compatibility before selecting them for evaluation.

## Completion evidence

Deliver a terminal-generated comparison table with per-task outputs, separate
agent and writing scores, runtime measurements, and documented failure cases.
The result should support choosing a checkpoint for the first training pilot.
No model evaluation has been run for this work item yet.
