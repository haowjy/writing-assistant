# Implementation milestones

## Present scaffold

- Installable core package, CLI, TOML configs, lockfile, CI checks.
- Five text tools, bounded agent loop, scripted and HTTP inference backends.
- Independent task directories, trace logs, run manifests, basic outcome checks.
- Five synthetic smoke tasks; trajectory validation and reviewed-record export.

## Next: a real baseline

Follow the [Gemma baseline comparison](baseline-comparison.md): compare pretrained
and instruction-tuned 12B/E4B checkpoints, then selected community derivatives.
Define the scores and verify writing-only and agent execution before freezing
the held-out suite.

## Next: first unified SFT

1. Replay and review a small trajectory pilot, with author/work splits fixed upfront.
2. Select and lock a compatible TRL/Transformers/PEFT/bitsandbytes environment.
3. Implement QLoRA loading and assistant-mask checks; overfit a tiny diagnostic batch.
4. Run a small unified adapter experiment and evaluate under the same inference budget.
5. Add split planner/writer adapters only once the baseline comparison is trustworthy.
