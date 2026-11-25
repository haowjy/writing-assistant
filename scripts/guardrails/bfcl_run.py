"""Run BFCL multi-turn against one OpenAI-compatible endpoint and print its scores as JSON.

Runs inside the guardrails image (`/opt/bfcl/bin/python`), because BFCL `eval()`s call
strings whose function names the model chose. bfcl-eval only evaluates models in its own
registry, and that registry has no entry meaning "this model behind this URL", so this adds
one: its OpenAI chat-completions handler in native function-calling (FC) mode, which reads
OPENAI_BASE_URL and OPENAI_API_KEY.

    bfcl_run.py <model> <project-root> <threads> [<category> ...]

Writes the scores to <project-root>/scores.json (BFCL prints its own trace to stdout); exits non-zero if any entry still failed on a
transport error (HTTP 5xx, timeout, connection) after the retry rounds, because BFCL would
otherwise score it as a wrong answer.
"""

import json
import os
import sys
from pathlib import Path

model, root, threads, *categories = sys.argv[1:]
categories = categories or ["multi_turn"]
os.environ["BFCL_PROJECT_ROOT"] = root  # read at import time

from bfcl_eval.__main__ import cli  # noqa: E402
from bfcl_eval.constants.category_mapping import VERSION_PREFIX  # noqa: E402
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig  # noqa: E402
from bfcl_eval.model_handler.api_inference.openai_completion import (  # noqa: E402
    OpenAICompletionsHandler,
)

MODEL_CONFIG_MAPPING[model] = ModelConfig(
    model_name=model,
    display_name=f"{model} (FC)",
    url="",
    org="",
    license="",
    model_handler=OpenAICompletionsHandler,
    is_fc_model=True,
    underscore_to_dot=True,  # OpenAI-style tool names cannot contain "."
)

# Exception types raised by the openai client for transport failures, as they appear in the
# traceback BFCL stores. Anything else (bad JSON in tool arguments, a missing field) is the
# model's answer and stays scored as BFCL scores it.
TRANSPORT_ERRORS = (
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "APIStatusError",
)
RETRY_ROUNDS = 3

result_dir = Path(root) / "result" / model.replace("/", "_")


def run(*args: str) -> None:
    # typer apps raise SystemExit; standalone_mode=False makes failures propagate instead.
    cli(list(args), standalone_mode=False)


def drop_transport_errors() -> int:
    """Remove entries that failed on a transport error so the next `generate` reruns them."""
    dropped = 0
    for path in result_dir.rglob("*_result.json"):
        kept = []
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            failed = str(entry.get("result", "")).startswith("Error during inference")
            trace = entry.get("traceback", "")
            if failed and any(name in trace for name in TRANSPORT_ERRORS):
                dropped += 1
            else:
                kept.append(line)
        path.write_text("".join(k + "\n" for k in kept))
    return dropped


def model_errors() -> int:
    return sum(
        str(json.loads(line).get("result", "")).startswith("Error during inference")
        for path in result_dir.rglob("*_result.json")
        for line in path.read_text().splitlines()
    )


for round_ in range(RETRY_ROUNDS + 1):
    run("generate", "--model", model, "--test-category", ",".join(categories),
        "--num-threads", threads)
    dropped = drop_transport_errors()
    if not dropped:
        break
    print(f"bfcl: {dropped} transport failures, regenerating", file=sys.stderr)
else:
    sys.exit(f"bfcl: entries still failing on transport errors after {RETRY_ROUNDS} retries")

run("evaluate", "--model", model, "--test-category", ",".join(categories))



def flat_sum(value) -> int:
    """Multi-turn entries store token counts as a list per turn of a list per step."""
    return sum(map(flat_sum, value)) if isinstance(value, list) else value


scores = {}
tokens = {"input": 0, "output": 0}
for path in sorted((Path(root) / "score" / model.replace("/", "_")).rglob("*_score.json")):
    header = json.loads(path.read_text().splitlines()[0])
    category = path.name.removeprefix(f"{VERSION_PREFIX}_").removesuffix("_score.json")
    scores[category] = {k: header[k] for k in ("accuracy", "correct_count", "total_count")}
for path in result_dir.rglob("*_result.json"):
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        for key, total in (("input_token_count", "input"), ("output_token_count", "output")):
            tokens[total] += flat_sum(entry.get(key, 0))

(Path(root) / "scores.json").write_text(
    json.dumps({"categories": scores, "tokens": tokens, "model_errors": model_errors()})
)
