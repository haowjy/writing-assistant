# Creative Writing Agent

A co-writer for long-running fiction: an agent that works over a project folder (the
manuscript plus a knowledge base of character, world, timeline and decision notes), keeps
committed canon apart from tentative ideas, tracks what each character knows, writes in the
author's style and proposes genuinely different directions.

The project restarted in November 2026. There is no code yet. Work proceeds in this order:

1. A creative-writing benchmark and an RL environment, built and debugged on open-weight models
   (free to run); once the environment works, frontier models join as subjects and judges.
2. Training, once both work.
3. Comparing model-development ideas against them, on a held-out split never used for
   selection.

Project knowledge lives in the Meridian KB, not in this repo: run `meridian context --name kb`
and start at its `index.md` (goals, vocabulary, evaluation principles, training lessons). Work
items and their decisions are under `meridian context --name work`.

Compute: an RTX 3090 on this machine (read `~/.context/MACHINE/gpu.md` before GPU work),
Google Colab credits for training, and a 48 GB M4 Pro Mac mini on the tailnet (`j-mini48`)
for inference and judge models.

## Where things run

Run locally first (the 3090, `j-mini48`). Plan every cloud run in the work directory
before it starts: what runs, expected cost, what stops it. One that costs money needs
Jimmy's approval; free usage does not.

## Google services

Use Google's CLIs through their skills, or a Google-hosted MCP server if connected:
`colab` (`colab-operator` skill) for Colab, `gwscli` (`gwscli` skill) for Drive and other
Workspace data. No direct REST calls or `rclone`: Google requires agents to use these
from February 2027. Signing in is Jimmy's to do.

The code and history before the restart are in the private
`haowjy/writing-assistant-archive` repository, and a bundle of every ref is in
`/data/jimyao/research/writing-assistant/archive/`. Do not commit model weights or book
corpora.
