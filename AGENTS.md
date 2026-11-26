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

Run on the local network first: the RTX 3090 here and `j-mini48`. Anything in the cloud
is planned before it starts, in a file in the work directory that says what runs, on what
machine, what it is expected to cost (money or Colab compute units), and what stops it.
Starting anything in the cloud that costs money needs Jimmy's approval of that plan. Free
usage needs no approval; still write the plan, and check the balance first
(`colab usage`).

## Google services

Reach Google services through Google's own command-line tools and their skills, or a
Google-hosted MCP server where one is connected. Do not call Google's REST APIs directly
or drive a third-party client such as `rclone` against them: from 1 February 2027 Google
requires AI agents to reach Workspace data (Drive, Docs, Gmail and the rest) through the
Workspace CLI or its Workspace MCP servers.

- Colab: the `colab` command, with the `colab-operator` skill. It also mounts Drive on a
  Colab machine (`colab drivemount`).
- Drive and other Workspace data: `gwscli`, with the `gwscli` skill.

Both are installed on this machine and neither is signed in yet; signing in opens a
browser and is Jimmy's to do. `gwscli` also needs an OAuth client from a Google Cloud
project, which does not exist yet. The `gwscli` skill is a copy unpacked from the program
(`gwscli skills install`) into `.mars-src/skills/`; refresh it after updating `gwscli`.

The code and history before the restart are in the private
`haowjy/writing-assistant-archive` repository, and a bundle of every ref is in
`/data/jimyao/research/writing-assistant/archive/`. Do not commit model weights or book
corpora.
