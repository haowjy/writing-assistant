# Local inference and checkpoint evaluation

The evaluation script runs Transformers/PyTorch in the same Python process as the
agent harness. No model server is required. The existing HTTP backend remains
available for other callers.

Install the runtime using a Python distribution with development headers:

```bash
uv python install 3.12
uv sync --managed-python --python 3.12 --extra data --extra metrics --extra inference
```

The uv-managed interpreter includes `Python.h`, required by Triton when compiling
its CUDA bindings. The lockfile fixes the installed dependency versions. Checkpoint downloads are
separate from installation; the loader defaults to local files only.

## Saved checkpoints

[The research script](../scripts/evaluate.py) defines the four baseline configurations.
It selects NF4 weights, BF16 compute, SDPA attention, one GPU, and an 8,192-token
input-plus-output budget. These are experiment settings to validate on the 3090,
not measured speed or memory guarantees. Context overflow fails explicitly rather
than dropping instructions or retrieved knowledge. Models load once per selected
batch; each scenario gets a fresh conversation and workspace.

After approving a particular run, call the library from Python:

```python
from pathlib import Path
from scripts.evaluate import MODELS
from writing_agent.inference import evaluate_checkpoint
from writing_agent.suite import load_scenarios

cases = load_scenarios(Path("data/processed/custom-eval"), ["F1-01", "F2-01"])
config = dict(MODELS[3])  # Gemma E4B instruction-tuned.
planned = evaluate_checkpoint(cases, config, Path("runs/checkpoint-eval"))
# Explicit execution, after approval:
# results = evaluate_checkpoint(cases, config, Path("runs/checkpoint-eval"),
#                               execute=True, allow_download=True)
```

To evaluate a complete training save, set `id` to its directory. Save the tokenizer
alongside the model, or supply `tokenizer={"id": ..., "revision": ...}` explicitly.
For an adapter save, retain the exact base model configuration and set
`adapter={"id": "checkpoints/run-a/checkpoint-1000"}`. Add `training_run` and
`global_step` to the configuration to retain training provenance in each result. This loads the PEFT adapter
without merging it into the base. Hub sources require immutable commit hashes;
local saves are hashed by file content, so overwriting weights cannot reuse an
older result accidentally. Evaluate only completed, immutable saves. Hashing large
local checkpoints adds startup I/O.

Results carry model, adapter, tokenizer, template hash, dependency versions,
precision, generation settings, and the harness/scenario identity. Different
checkpoint identities produce separate attempts in the same output directory.
Downloads require network access and any upstream access permissions.

## Tool and prose protocol

The harness owns tool execution. The model sees public instructions, permitted
tool schemas, conversation history, and tool results. It never receives the
private grading labels. Calls operate through the existing constrained workspace.

The current protocol is `gemma-native-v1`. Schemas are passed through the pinned
Gemma chat template's `tools` argument. The template produces native declarations;
we do not insert a custom JSON instruction. Generated text retains special tokens
until the tokenizer's response parser separates tool calls, thinking, and content.
The adapter assigns call IDs, while the shared harness validates and executes tools.
Tool results are rendered as native tool responses on the associated assistant turn.
Conversational replies remain ordinary text.

Every local generation records its rendered prompt and input token IDs before
execution, then its raw output and generated token IDs before parsing. These events
are in `trace.jsonl`; the pilot review links captured prompts when available. This
makes input-format and parsing failures inspectable without reconstructing them.
The tokenizer chat-template and response-template hashes are recorded with results.

Instruction-tuned research and pilot configurations enable thinking through the
explicit `enable_thinking=True` model setting. Omitted settings retain the earlier
non-thinking behavior. Thinking shares the `max_tokens` output budget with tool
calls and final text; budget exhaustion remains an execution failure.
The parser’s `thinking` field is saved separately from answer content and mapped
to the template’s `reasoning` field for subsequent tool calls. Prose extraction
uses answer content or designated files, never the separate thinking field.
Base text-only conditions retain explicit role-labelled transcripts. Native tool
execution requires a verified chat template: base transcript conditions reject tool
use until a separate base-model formatting decision is implemented and verified.
A generation token limit remains an execution failure; history is never silently
truncated to make it fit.

The earlier `writing-json-v1` and `writing-tools-v2` outputs remain historical
artifacts. The five-case E2B pilot used v2, with schemas embedded as ordinary text.
Three file/workspace cases have since been rerun with native tools and thinking
disabled; those artifacts remain separate. New protocol identities prevent
those outputs from being reused as native results.

The native integration follows Google's
[Gemma function-calling guide](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
and the checkpoint's pinned templates. The installed Transformers parser handles
nested values and multiline strings; we do not use the guide's demonstration regex
as a general file-content parser.

## Evaluation during future training

Training is not implemented here. The evaluation interface supports these two
integration points:

1. **Saved checkpoint evaluation:** a training job finishes a checkpoint save and
   queues its path plus global step and run ID. A separate evaluation process calls
   `evaluate_checkpoint` on a fixed development subset. On the single 3090, schedule
   this after training releases the GPU, or pause/save/exit training before loading
   evaluation weights. A concurrent evaluator needs its own available GPU memory.
2. **Evaluation using resident weights:** a training loop passes its existing model
   and tokenizer to `TransformersBackend`, then calls `run_selected`. Generation
   uses inference mode, restores each module's train/eval flag, and preserves CPU
   and selected-device PyTorch RNG state even after failure. No second copy of the
   weights is loaded. This path is serial and intended for a single-device model;
   distributed/sharded training requires a separate integration.

```python
from writing_agent.inference import TransformersBackend
from writing_agent.suite import run_selected

# model/tokenizer belong to the trainer. record identifies the exact training save
# and includes the generation/protocol settings used by TransformersBackend.
# Call at a pause between optimizer steps, with no active forward/backward pass.
# results = run_selected(cases, record,
#     lambda: TransformersBackend(model, tokenizer, record),
#     Path("runs/training-development"), execute=True)
```

Before training begins, approve a fixed development subset and evaluation cadence.
Run inexpensive mechanical/prose measurements at those checkpoints; schedule Astra
judgments less frequently with an explicit call budget. Reuse saved artifacts for
rescoring. Keep scenario set, seed, prompt protocol, context budget, and precision
fixed across the curve. If resident training weights use a different precision
from standalone NF4 evaluation, label those as separate conditions.

Repeated checkpoint selection uses development data. Keep final evaluation out of
training feedback and run it only after checkpoint selection. The full 200 attempts,
repeat/probe experiments, and external benchmarks remain approval-gated; no trainer
callback automatically launches them.

## Upstream interfaces

The loader follows [Transformers Gemma 4 documentation](https://huggingface.co/docs/transformers/model_doc/gemma4),
the [12B Unified model card](https://huggingface.co/google/gemma-4-12B),
[bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes),
and [PEFT adapter loading](https://huggingface.co/docs/peft/quicktour).
