# The guardrail evaluation tools, pinned. One venv per tool because their dependency pins
# conflict (bfcl-eval needs numpy==1.26.4 and tree_sitter==0.21.3; evalplus needs
# tree-sitter>=0.22). Datasets are baked in so a run reads the same items offline.
# Changing any pin here changes the image tag guardrails.sh derives from this file, and the
# scores are then a different measurement: say so beside any comparison across the change.
FROM python:3.11-slim-bookworm@sha256:ef56ae98435cc40d8a27020b0b32a9941515acd5840f0fba34858bb34dd362d1
COPY --from=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /usr/local/bin/uv

ARG RESOLVED_BEFORE=2026-11-25T00:00:00Z
ENV UV_EXCLUDE_NEWER=${RESOLVED_BEFORE} UV_NO_CACHE=1 \
    HF_HOME=/opt/hf XDG_CACHE_HOME=/opt/cache NLTK_DATA=/opt/nltk_data

RUN uv venv -q --python /usr/local/bin/python3 /opt/lm-eval \
 && uv pip install -q --python /opt/lm-eval "lm-eval[api,ifeval]==0.4.13"
RUN uv venv -q --python /usr/local/bin/python3 /opt/evalplus \
 && uv pip install -q --python /opt/evalplus "evalplus==0.3.1"
# soundfile: qwen-agent, which bfcl-eval imports unconditionally, imports it without
# declaring it.
RUN uv venv -q --python /usr/local/bin/python3 /opt/bfcl \
 && uv pip install -q --python /opt/bfcl "bfcl-eval==2026.3.23" soundfile

# IFEval prompts (google/IFEval at the revision lm-eval's `main` resolves to; fail the build
# if upstream ever moves it) and the NLTK data its checkers download on first import.
ARG IFEVAL_REVISION=966cd89545d6b6acfd7638bc708b98261ca58e84
RUN /opt/lm-eval/bin/python -c "import datasets, sys; \
from huggingface_hub import HfApi; \
sha = HfApi().dataset_info('google/IFEval').sha; \
sys.exit(0 if sha == '${IFEVAL_REVISION}' else f'google/IFEval main is {sha}, expected ${IFEVAL_REVISION}'); " \
 && /opt/lm-eval/bin/python -c "import datasets; datasets.load_dataset('google/IFEval', revision='main')" \
 && /opt/lm-eval/bin/python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/nltk_data')" \
 && /opt/lm-eval/bin/python -c "import lm_eval.tasks.ifeval.instructions_util" \
 && chmod -R a+rX /opt/hf /opt/nltk_data

# HumanEval+ v0.1.10 (evalplus 0.3.1's default) and its ground-truth outputs, which evalplus
# computes by running the reference solutions and caches; at run time the cache is read-only.
RUN /opt/evalplus/bin/python -c "from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash; \
from evalplus.evaluate import get_groundtruth; \
p = get_human_eval_plus(); get_groundtruth(p, get_human_eval_plus_hash(), [])" \
 && chmod -R a+rX /opt/cache

# What summarize.py stamps into guardrails.json: each tool and the HTTP client it calls the
# endpoint through.
RUN /opt/lm-eval/bin/python -c "from importlib.metadata import version as v; print(v('lm-eval'), v('aiohttp'))" > /opt/v-lm \
 && /opt/evalplus/bin/python -c "from importlib.metadata import version as v; print(v('evalplus'), v('openai'))" > /opt/v-ep \
 && /opt/bfcl/bin/python -c "from importlib.metadata import version as v; print(v('bfcl-eval'), v('openai'))" > /opt/v-bfcl \
 && python3 -c "import json; r = lambda f: open(f).read().split(); \
(lm, aio), (ep, ep_oai), (bf, bf_oai) = r('/opt/v-lm'), r('/opt/v-ep'), r('/opt/v-bfcl'); \
json.dump({'lm-eval': lm, 'lm-eval/aiohttp': aio, 'evalplus': ep, 'evalplus/openai': ep_oai, \
'bfcl-eval': bf, 'bfcl-eval/openai': bf_oai, 'resolved_before': '${RESOLVED_BEFORE}', \
'ifeval_dataset_revision': '${IFEVAL_REVISION}', 'humaneval_plus_dataset': 'v0.1.10'}, \
open('/opt/versions.json', 'w'), indent=2)" \
 && rm /opt/v-lm /opt/v-ep /opt/v-bfcl

ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
WORKDIR /work
