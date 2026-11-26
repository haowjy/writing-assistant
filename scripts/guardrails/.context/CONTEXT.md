# Guardrail evaluations: why they run this way

`scripts/guardrails.sh` produces the three general-ability numbers the benchmark reports
beside every creative-writing result (`design/benchmark.md`, External Guardrails: drops of at
most 2, 3 and 5 points against the base checkpoint). The numbers are only comparable when
the tool versions, data and scoring mode are identical, so all of them are pinned in
`Dockerfile`; the image tag is derived from that file's hash and stamped into
`guardrails.json`. Bumping a pin starts a new series: rerun the base checkpoint with it.

## Tools

| Score | Tool | Scoring |
|---|---|---|
| `ifeval` | `lm-eval` 0.4.13, task `ifeval` | prompt-level strict accuracy, 541 prompts, greedy |
| `humaneval_plus` | `evalplus` 0.3.1, HumanEval+ v0.1.10 | pass@1 on base + plus tests, greedy, instruct prompt |
| `bfcl` | `bfcl-eval` 2026.3.23 (BFCL v4) | unweighted mean of the four `multi_turn` categories, 800 entries, native function calling |

- `lm-eval` has no BFCL task (checked in 0.4.13), so BFCL uses Berkeley's own package. It
  only evaluates models in its built-in registry; `bfcl_run.py` adds an entry for the
  endpoint's model using BFCL's OpenAI chat-completions handler in FC mode. Prompt mode
  would measure our parsing of text, not the model's tool calling.
- The full multi-turn set is affordable: about 150k input tokens per entry, so roughly
  120M tokens per run. No subset. Locally that is hours: a `GUARDRAILS_LIMIT=3` run on
  Gemma 4 26B QAT Q4_0 (llamAmpere, 4 slots, thinking on) took 349 s for 12 BFCL entries
  (1.9M input tokens, about 5.4k prefill tokens/s), so the 800 entries take about 6.5 h;
  IFEval about 30 min and HumanEval+ generation about 10 min (evalplus sends one request at
  a time). About 7 h in all on Gemma; Inkling on one slot is several times slower.
- The image `ganler/evalplus:v0.3.1` actually contains evalplus 0.4.0.dev2, so it is not
  used; the image here installs 0.3.1.
- Dependencies are resolved as of `RESOLVED_BEFORE` (uv `--exclude-newer`), so a rebuild
  resolves the same transitive versions.
- Scores reflect the endpoint's default mode. On DeepInfra, Gemma 4 has thinking off by
  default. Published numbers measured with thinking on are not comparable.
- Local llama.cpp servers think by default for Gemma 4, Qwen3.5 and Inkling (reasoning comes
  back in `reasoning_content`). The tools' output caps (evalplus 768 tokens, IFEval 1280)
  count that thinking: in a `GUARDRAILS_LIMIT=3` run on local Gemma (2026-11-26) one of
  three HumanEval+ replies was cut off mid-thought and the thought was taken as the code.
  Until the thinking mode for guardrails is decided, a full run on a thinking-on server
  measures the caps. The hosted Gemma numbers were thinking off.
- None of the three tools can add request-body fields, so an endpoint with `params` in
  `models.toml` (Hy4's thinking switch) is refused rather than measured in a different
  mode from the one the benchmark runs it in.

## Run-to-run noise

"Greedy" through a hosted endpoint is not deterministic. Two runs of IFEval against DeepInfra's
Gemma 4 26B-A4B, a few hours apart (2026-11-25), returned identical text for only 56 of 541
prompts, and 16 prompts flipped pass/fail. Prompt-strict accuracy was 88.91 and 88.54. A
difference of a few tenths of a point between checkpoints is noise, not a regression. Serve
both checkpoints the same way (same engine and precision), and repeat a run before acting on
a drop near a threshold.

## Sandbox

Two steps run model-written code: HumanEval+ executes generated programs, and BFCL
`eval()`s call strings built from the function names the model returned. Every step runs
in a container as the invoking uid, with no capabilities, `no-new-privileges`, a read-only
root, a 4 GB tmpfs `/tmp`, memory and pid limits, and only its own output directory
writable. The steps that run model-written code have no network and no key: HumanEval+
execution sees only a copy of the samples, and BFCL is split into `bfcl_run.py generate`
(endpoint, key, network) and `bfcl_run.py evaluate` (`--network none`, `OPENAI_API_KEY=unused`:
bfcl-eval's OpenAI handler builds a client when constructed but evaluation never calls it).
Checked 2026-11-26: evaluate alone, that way, over the saved generation of the 2026-11-25
Gemma run reproduced its scores exactly. For a loopback `base_url`, the endpoint steps use
the host network.

## Failure handling

- A one-token request goes out first, so an unfunded account, a bad key or a wrong model
  name fails in seconds, before the output directory is created.
- A run removes any earlier `guardrails.json` first and writes the new one only after
  `summarize.py` has checked that each suite was fully scored (541, 164, 4 x 200).
- lm-eval re-raises after its retries.
- evalplus retries forever, so codegen is bounded by `timeout`.
- `GUARDRAILS_LIMIT=N` runs every step on the first N items (lm-eval `--limit`, evalplus
  `--id_range`, BFCL's id file and `--partial-eval`; HumanEval+ execution is skipped, as
  evalplus refuses a partial sample set and that step never calls the endpoint) and stops before `summarize.py`, so a
  partial score can never be stored as a result. It is the check for a loopback endpoint,
  which has no preflight request, and its per-step seconds are what a full-run estimate
  scales from.
- BFCL records a failed request as a wrong answer. `bfcl_run.py` regenerates entries that
  failed on transport errors (up to 3 rounds) and fails if any remain. Entries that fail
  because of the model's own output (bad JSON arguments) stay scored as BFCL scores them,
  and `model_errors` counts them.
