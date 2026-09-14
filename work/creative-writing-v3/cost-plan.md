# Creative Writing v3: first run and API budget

Use Creative Writing v3 rubric scoring as the primary external prose benchmark.
Keep our custom suite for tools, KB work, and prose-distribution diagnostics.
Start with Gemma E2B-IT on the local RTX 3090, thinking enabled, 32 prompts with
three seed variants each (96 outputs). Other checkpoints remain separate runs.

## Paid grading protocol

Use `claude-sonnet-4-6` through the direct Anthropic API, rubric scoring only.
No pairwise Elo calls initially. This gives rubric results, not an official
leaderboard Elo. The repository's current recommendation is Sonnet 4.6; the
website's older rubric-judge description differs, so record this exact protocol.

Anthropic's listed Sonnet 4.6 prices, checked 2026-09-14: $3 per million input
tokens and $15 per million output tokens. Standard synchronous pricing below;
no cache or Batch API savings assumed.

| Scope | Estimated judge cost |
|---|---:|
| Four-output integration check | $0.07–$0.17 |
| 32 prompts, one variant each | $0.58–$1.39 |
| 32 prompts, three variants each | $1.73–$4.18; budget $2–$5 |
| Four models, 96 outputs each | $6.91–$16.70; budget $8–$20 |

Assumptions per judgment: 2,500–4,500 input tokens, 700–2,000 output tokens.
These are estimates, not token counts from Anthropic. Inspected rubric overhead
averages 3,274 characters before candidate text; the prompts generally request
longer writing than our short pilot. Each generated item receives one rubric
judgment on the successful path. Actual length and retries change cost.

Upstream caps rubric outputs at 4,096 tokens. At 4,000 input tokens and that full
output limit, 96 successful calls cost $7.05. That is an example, not an absolute
upper bound on input length or retries. Approved spending ceiling: **$10 for the
first E2B-IT rubric run**, with per-request usage accounting and budget reservation
before sending calls. The local adapter enforces this ceiling through persistent request reservations;
it does not change the Anthropic account limit.

The upstream README estimates roughly $10/model for the benchmark, but its
pairwise process is adaptive. We have not established a reliable full-Elo cap.
Rubric outputs can be reused later for Elo; no need to regenerate Gemma writing.

Haiku 4.5 costs $1/$5 per million input/output tokens, so the same token volumes
would be about one-third as expensive. Changing the judge changes the measurement;
use Sonnet 4.6 initially for compatibility with the upstream recommendation.
Batch API pricing can reduce judgment cost by 50%, but requires a batch submission
adapter; it is not assumed in the initial run or these totals.

Local candidate generation has no model API charge, but uses local GPU time and
electricity. Runtime is not yet measured for these roughly 1,000-word tasks. The
short custom-pilot timings do not justify a precise full-run estimate.

## Execution

The user authorized the custom 50-case run followed by the E2B-IT external prose
benchmark on 2026-09-14. The custom run is in progress. Creative Writing v3 waits
for its report, then verifies one generation and one paid judgment before running
the remaining items. WritingBench, other external checks, and other checkpoints
remain separate scope.

The implementation is [scripts/creative_writing_v3.py](../../scripts/creative_writing_v3.py).
Call `prepare()`, `generate(subset=...)`, `grade(subset=...)`, and `report()` from
Python. Importing the module does not execute a benchmark. The sequential research
orchestrator is [scripts/run_benchmarks.py](../../scripts/run_benchmarks.py).

Prompts, rubric, criteria, and inspected upstream code are pinned at
`c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b` under
`data/raw/research/creative-writing-v3/`. Generation uses temperature 0.7, min_p 0.1,
top_p 1, top_k 0, a 12,000-token output allowance, and a 16,384-token context.
Thinking consumes output tokens and is saved separately; the judge receives final
writing only. Local inference uses NF4 quantization. Failed generations are kept
without upstream's automatic retries. The rubric contains 22 criteria. Following upstream instructions, omitted criteria
remain unscored and the item score averages the returned numeric criteria. The full
benchmark score requires a judgment for every output.

Credentials load from `ANTHROPIC_API_KEY` or the ignored root `.env` at grading
execution. The free token-count endpoint verified key and model access. The ledger
reserves a conservative request cost before each paid call, replaces it with actual
usage on success, and retains unresolved reservations after interruptions. It never
automatically retries an uncertain paid request. Completed raw judgments are cached.

Artifacts are under `runs/creative-writing-v3-e2b-it-2026-09-14/`: `manifest.json`,
per-item generation/thinking/judgments, `judgments/ledger.json`, `report.json`, and
`review.md`. The custom run lives in `runs/custom50-e2b-it-2026-09-14/`.

Sources: [benchmark README](https://github.com/EQ-bench/creative-writing-bench),
[pinned task implementation](https://github.com/EQ-bench/creative-writing-bench/blob/c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b/core/conversation.py),
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
