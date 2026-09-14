# External benchmark runs

Use E2B-IT with thinking enabled for the initial baseline. On 2026-09-14 the user
expanded the run to include the planned external checks, then reconsidered the
cost of LLM grading. The user selected 32 Creative Writing v3 outputs with a $2 grading cap. The queue
is custom50, the prose baseline, the coding diagnostic, then IFEval. WritingBench
is deferred.

| Benchmark | Selection | Grading | Current execution |
|---|---:|---|---|
| IFEval | All 541 prompts, 834 instructions | Official strict/loose prompt and instruction accuracy | Queued after the coding diagnostic |
| HumanEval+ | HumanEval/0–31 | EvalPlus base and extended tests, one generation per task | Queued after the prose baseline |
| Creative Writing v3 | 32 prompts, first seed variant each | Sonnet 4.6 rubric | Approved; queued after custom50 |
| WritingBench | 1,000 downloaded prompts | Request-specific LLM rubric | Deferred until a later decision |

The Creative Writing subset is estimated to cost roughly $0.60–$1.40 in judgment tokens,
with an approved $2 ceiling. These are estimates from the
[full-run cost assumptions](../creative-writing-v3/cost-plan.md), not measured
charges. The 32-output/$2 selection supersedes the earlier 96-output/$10 scope. IFEval and HumanEval+ have no judge API charge. Local generation can
take hours; these checks do not share the short custom suite's output budget.

Freeze the baseline subset and reuse it at selected training checkpoints. Keep it
out of training data. Repeated checkpoint-based selection makes this a development
comparison, so reserve a separate final evaluation set for final claims. Label a
32-output prose run and the coding diagnostic as subsets, not full leaderboards.

## Python execution and artifacts

[scripts/external_checks.py](../../scripts/external_checks.py) exposes `prepare()`,
`generate(name)`, `grade_ifeval()`, `grade_coding()`, and the sequential `run()`.
Generation uses the shared Transformers backend with greedy decoding, an
8,192-token output allowance, a 16,384-token context, and NF4 weights. Thinking is
saved separately and consumes the output allowance. Each prompt gets a fresh
conversation. Failures are retained without retries and enter grading as empty
responses, preserving the selected denominator.

Run artifacts are under `runs/external-e2b-it-2026-09-14/`: the selection manifest,
per-item prompts, exact traces, thinking, responses, grading logs, and results.
The automatic-check pipeline starts after the prose baseline, runs the 32 coding tasks and grades them,
then runs IFEval and grades it. It does not invoke an LLM judge.

IFEval uses the downloaded upstream verifier at
`26d8ccdab6fec61b5c83ad6327ea8bda9e580288`; that revision keeps its scoring functions
in `evaluation_main.py`. Dependencies are in the `external` extra. NLTK tokenizer
data lives in the private user directory `~/nltk_data`, since NLTK rejects this
workspace's directory permissions for downloads.

HumanEval+ uses dataset v0.1.10 and EvalPlus 0.3.1. The
[Dockerfile](Dockerfile) builds a CPU-only evaluator. The
[container script](evalplus_runner.py) calls EvalPlus's Python sanitizer and scorer
on the staged subset. Containers have no network, a read-only root, no Linux
capabilities, a non-root user, and CPU/memory/process limits. They receive only
benchmark inputs and an output directory. Each run records its immutable image ID.
The repository and credentials are not mounted.

WritingBench is acquired at `ae2d5176449b7b769815482641d35926f26793eb` under
`data/raw/research/writingbench/`, with source hashes and license. It has not been
integrated into paid execution.

## Verification

- The 53-test repository suite passes, including external generation resume,
  immutable selections, invalid paths, and preserved generation failures.
- Official IFEval grading of empty fixture responses returns zero for all four
  aggregate metrics, over 541 prompts and 834 instructions.
- The isolated EvalPlus fixture run passes 31 canonical solutions and rejects the
  deliberately broken HumanEval/0 solution on both base and extended tests.
- Fixture outputs are separate from model runs under
  `runs/external-evaluator-fixtures-2026-09-14/`.

Sources: [IFEval](https://github.com/google-research/google-research/tree/26d8ccdab6fec61b5c83ad6327ea8bda9e580288/instruction_following_eval),
[EvalPlus 0.3.1](https://github.com/evalplus/evalplus/tree/v0.3.1),
[WritingBench](https://github.com/X-PLUG/WritingBench/tree/ae2d5176449b7b769815482641d35926f26793eb).
