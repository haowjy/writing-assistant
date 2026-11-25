#!/usr/bin/env bash
# IFEval, HumanEval+ and multi-turn BFCL against one OpenAI-compatible endpoint.
#
#   scripts/guardrails.sh <endpoint-name> <out-dir>
#
# Reads the endpoint's base_url, model and api_key_env from models.toml at the repository
# root (the key from the environment, else the root .env), runs the three evaluations in the
# pinned image built from scripts/guardrails/Dockerfile, and writes
# <out-dir>/guardrails.json. Each tool's raw output stays under <out-dir>/guardrails/.
# Exits non-zero, leaving no guardrails.json, when any evaluation fails.
#
# Needs docker and python3 >= 3.11 on the host; nothing here uses the GPU. Model-written code
# runs only inside the container (scripts/guardrails/.context/CONTEXT.md has the sandbox and the pins).
#
# GUARDRAILS_CONCURRENCY (default 16) sets parallel requests to the endpoint. It changes
# wall time, not scores.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <endpoint-name> <out-dir>" >&2
  exit 2
fi
endpoint=$1
out=$(mkdir -p "$2" && cd "$2" && pwd)
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
here=$repo/scripts/guardrails
concurrency=${GUARDRAILS_CONCURRENCY:-16}

# A stale result must not survive a failed rerun looking complete.
rm -f "$out/guardrails.json"
work=$out/guardrails
rm -rf "$work"
mkdir -p "$work"/{ifeval,humaneval/gen,humaneval/exec,bfcl}

# --- endpoint ------------------------------------------------------------------------------
{ read -r base_url; read -r model; read -r key_env; } < <(python3 - "$repo/models.toml" "$endpoint" <<'EOF'
import sys, tomllib
endpoints = tomllib.load(open(sys.argv[1], "rb")).get("endpoints", {})
e = endpoints.get(sys.argv[2])
if e is None:
    sys.exit(f"guardrails: no [endpoints.{sys.argv[2]}] in {sys.argv[1]}")
if e.get("kind") != "openai" or not e.get("base_url"):
    sys.exit(f"guardrails: {sys.argv[2]} is not an OpenAI-compatible chat endpoint with a base_url")
print(e["base_url"].rstrip("/"), e["model"], e.get("api_key_env", ""), sep="\n")
EOF
)

if [[ -z $key_env ]]; then
  api_key=EMPTY  # local servers ignore the key, but every client requires one
elif [[ -n ${!key_env:-} ]]; then
  api_key=${!key_env}
else
  # Read only the one line; never source .env.
  api_key=$(sed -n -E "s/^(export[[:space:]]+)?${key_env}=[\"']?([^\"']*)[\"']?[[:space:]]*$/\2/p" \
    "$repo/.env" 2>/dev/null | tail -1)
  [[ -n $api_key ]] || { echo "guardrails: $key_env is not set and not in $repo/.env" >&2; exit 1; }
fi
export OPENAI_API_KEY=$api_key OPENAI_BASE_URL=$base_url

# --- image ---------------------------------------------------------------------------------
image=wa-guardrails:$(sha256sum "$here/Dockerfile" | cut -c1-12)
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "guardrails: building $image" >&2
  docker build -q -t "$image" "$here" >&2
fi

# Every container: this user's uid, no capabilities, read-only root, only $work writable.
# Steps that call the endpoint need a network; a loopback endpoint means the host's network.
sandbox=(--rm --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges
  --read-only --tmpfs /tmp:rw,size=4g --memory 8g --pids-limit 1024 -e HOME=/tmp
  -v "$here:/opt/guardrails:ro")
case $base_url in
  http://localhost[:/]* | http://127.* | https://localhost[:/]* | https://127.*) net=(--network host) ;;
  *) net=(--network bridge) ;;
esac
with_endpoint=("${sandbox[@]}" "${net[@]}" -e OPENAI_API_KEY -e OPENAI_BASE_URL)

declare -A seconds
timed() {  # timed <name> <command...>: run, record wall seconds, fail loudly
  local name=$1 start=$SECONDS
  shift
  echo "guardrails: $name" >&2
  if ! "$@"; then
    echo "guardrails: $name failed; no guardrails.json written" >&2
    exit 1
  fi
  seconds[$name]=$((SECONDS - start))
}

# --- IFEval (lm-eval; no model code is executed) --------------------------------------------
# `datasets` takes lock files even when reading its cache, so the baked cache is copied to /tmp.
timed ifeval docker run "${with_endpoint[@]}" -v "$work/ifeval:/work" -e HF_HOME=/tmp/hf "$image" \
  sh -c 'cp -r /opt/hf /tmp/hf && exec "$@"' sh \
  /opt/lm-eval/bin/lm_eval --model local-chat-completions \
  --model_args "model=$model,base_url=$base_url/chat/completions,num_concurrent=$concurrency,max_retries=10,tokenized_requests=False,timeout=600" \
  --tasks ifeval --apply_chat_template --output_path /work --log_samples

# --- HumanEval+ (evalplus) -------------------------------------------------------------------
# Generation needs the endpoint and only parses the code. evalplus retries a failing request
# forever, so the step is bounded.
timed humaneval_gen docker run "${with_endpoint[@]}" -v "$work/humaneval/gen:/work" "$image" \
  timeout 3600 /opt/evalplus/bin/evalplus.codegen --model "$model" --dataset humaneval \
  --backend openai --base-url "$base_url" --greedy --root /work
samples=$(find "$work/humaneval/gen" -name '*.jsonl' ! -name '*.raw.jsonl')
[[ $(wc -l < "$samples") -eq 164 ]] || { echo "guardrails: HumanEval+ generation incomplete: $samples" >&2; exit 1; }
cp "$samples" "$work/humaneval/exec/samples.jsonl"
# Execution: no network, and the only writable mount holds a copy of the samples.
timed humaneval_exec docker run "${sandbox[@]}" --network none -v "$work/humaneval/exec:/work" "$image" \
  /opt/evalplus/bin/evalplus.evaluate --dataset humaneval --samples /work/samples.jsonl \
  --parallel 8 --i-just-wanna-run

# --- BFCL multi-turn (bfcl-eval; eval()s model-named calls, so sandboxed too) -----------------
timed bfcl docker run "${with_endpoint[@]}" -v "$work/bfcl:/work" "$image" \
  /opt/bfcl/bin/python /opt/guardrails/bfcl_run.py "$model" /work "$concurrency"

# --- guardrails.json -------------------------------------------------------------------------
docker run "${sandbox[@]}" --network none -v "$work:/work:ro" "$image" \
  python3 /opt/guardrails/summarize.py --endpoint "$endpoint" --model "$model" \
  --base-url "$base_url" --image "$image" \
  --commit "$(git -C "$repo" rev-parse HEAD)$(git -C "$repo" diff --quiet HEAD -- scripts || echo -dirty)" \
  --seconds "ifeval=${seconds[ifeval]}" \
  --seconds "humaneval_plus=$((seconds[humaneval_gen] + seconds[humaneval_exec]))" \
  --seconds "bfcl=${seconds[bfcl]}" > "$work/guardrails.json"
mv "$work/guardrails.json" "$out/guardrails.json"
cat "$out/guardrails.json"
