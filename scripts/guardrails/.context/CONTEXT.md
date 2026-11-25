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
  120M tokens per run. No subset.
- The image `ganler/evalplus:v0.3.1` actually contains evalplus 0.4.0.dev2, so it is not
  used; the image here installs 0.3.1.
- Dependencies are resolved as of `RESOLVED_BEFORE` (uv `--exclude-newer`), so a rebuild
  resolves the same transitive versions.
- Scores reflect the endpoint's default mode. On DeepInfra, Gemma 4 has thinking off by
  default. Published numbers measured with thinking on are not comparable.

## Sandbox

Two steps run model-written code: HumanEval+ executes generated programs, and BFCL
`eval()`s call strings built from the function names the model returned. Every step runs
in a container as the invoking uid, with no capabilities, `no-new-privileges`, a read-only
root, a 4 GB tmpfs `/tmp`, memory and pid limits, and only its own output directory
writable. HumanEval+ execution also gets `--network none` and sees only a copy of the
samples. BFCL needs the endpoint, so it keeps a network, but it has no host mounts beyond
its output directory. For a loopback `base_url`, the endpoint steps use the host network.

## Failure handling

- A run removes any earlier `guardrails.json` first and writes the new one only after
  `summarize.py` has checked that each suite was fully scored (541, 164, 4 x 200).
- lm-eval re-raises after its retries.
- evalplus retries forever, so codegen is bounded by `timeout`.
- BFCL records a failed request as a wrong answer. `bfcl_run.py` regenerates entries that
  failed on transport errors (up to 3 rounds) and fails if any remain. Entries that fail
  because of the model's own output (bad JSON arguments) stay scored as BFCL scores them,
  and `model_errors` counts them.
