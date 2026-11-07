# Setup rationale

The package supports Python research scripts and retains TOML/CLI smoke workflows. Inference runs behind
a backend interface so the runtime can change without changing task execution.
The research script uses Transformers/PyTorch and PEFT through the optional
`inference` extra. The core has no third-party runtime dependencies. Training runs
through explicit entry points that import the model stack only when executed:
`training.py` (QLoRA SFT) and Phase 8's task-graph DAPO trainer
([task-graph training](../../docs/task-graph-training.md)). The former standalone DAPO
training path is retired; its code remains at commit `9cb9944`, in `main`'s history.

## References and choices

The documentation reviewed on 2026-09-11 informed these choices. The scaffold was
not copied from a research repository.

| Reference | Choice in this codebase |
|---|---|
| [Python Packaging Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) | `src/` layout and an installed CLI entry point. |
| [Inspect evaluation logs](https://inspect.aisi.org.uk/eval-logs.html) | Separate run directories containing inputs, configuration, traces, and scores. |
| [Inspect sandboxing](https://inspect.aisi.org.uk/sandboxing.html) | Separate task workspaces. The local text tools do not provide Inspect's container isolation. |
| [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) | Export conversational `messages` and `tools`; `training.py` trains on them with `SFTTrainer`. |
| [PEFT quantization](https://huggingface.co/docs/peft/v0.17.0/developer_guides/quantization) | QLoRA for SFT in `training.py`; SFT GPU training remains unverified. |
| [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation) | Local checkpoint generation, adapted to the shared tool loop. |
| [vLLM server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/) | Optional HTTP transport for existing workflows. The research script defaults to local inference. |

The run format is local to this package and is not Inspect-compatible. The uv
lockfile pins development and optional data/metric/model dependencies; `pyproject.toml` bounds the build-backend
version. Select and verify training dependencies against the chosen GPU environment.

See [evaluation contracts](CONTEXT.md) for module responsibilities and current
limits, and [training experiments](../../work/research-plan/training-experiments.md)
for the proposed adaptation methods.
