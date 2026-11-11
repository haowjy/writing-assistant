# Creative Writing Agent

This repository researches how to train and evaluate a conversational agent that
collaborates with writers on long-running creative projects. The agent should use
external project files to plan, write, and revise while preserving the author's
decisions and story continuity.

The published artifact is the generated task list and the scripts that fetch,
clean, and catalog its sources. Do not commit model weights or book corpora.

Start from [TODO.md](TODO.md): the active work order, including training work that
lives on unmerged branches, so `main` is not the newest code for it.

- [Research wiki](wiki/index.md): goals and shared concepts.
- [Research work](work/index.md): deferred work and experiment plans.
- [Source guidance](src/AGENTS.md): implementation constraints and code navigation.
- [README](README.md) and [docs/](docs/): user-facing setup, usage, and data formats.

Read the local `AGENTS.md` when entering a directory. Its sibling `.context/` holds
detailed contracts, architecture, and rationale; for example,
[source context](src/.context/CONTEXT.md) and [wiki conventions](wiki/.context/CONTEXT.md).
