# Phase 8 public probe tasks

These three original, single-writer tasks are the public fixture set for the first
integrated GPU probe. Each asks for a short file-based scene and states its task-specific
detail in the brief: an amber lighthouse lantern, a blue greenhouse key, or a copper signal
bell. There is no author interaction or feedback turn. Only `scene_nonempty` is required;
the four optional checks reward partial completion at different levels. These checks test
probe plumbing, not writing quality.

## Provenance and scope

The briefs, seed files, and scripted sample prose in the three JSON files are original
fixture material. There is no catalog selection, book or other private source, or data
shared between members. The unused-author-preference privacy canary now shares a private,
non-admitted check record with the private-store-dump canary; neither appears in the public
brief, initial files, or tool manifest.

| Task config | Public writing prompt |
|---|---|
| [`t1-lighthouse.json`](t1-lighthouse.json) | A lighthouse keeper preparing for a boat at low tide; amber lantern |
| [`t2-winter-garden.json`](t2-winter-garden.json) | A winter greenhouse before dawn; blue key and warm soil |
| [`t3-coastal-post.json`](t3-coastal-post.json) | A coastal post runner before a storm; copper bell and salt grass |

## Reward spread from scripted candidates

Each graph uses the same deterministic weights: required `scene_nonempty` = 1,000;
stated-detail phrase = 2,500; public detail = 2,000; 24–70 word range = 2,500; and optional
`field-notes.txt` nonempty = 2,000. Scores are exact basis points out of 10,000. The scripted
candidates exercise each optional layer in order:

| Task / scripted lineage | Checks passed | Reward |
|---|---|---:|
| t1 lighthouse / `nonempty_only` | required only | 1,000 |
| t1 lighthouse / `stated_detail` | required + stated detail | 3,500 |
| t2 winter garden / `phrase_and_detail` | required + stated detail + public detail | 5,500 |
| t3 coastal post / `word_range` | required + stated detail + public detail + word range | 8,000 |
| t3 coastal post / `all_optional` | required + all optional checks | 10,000 |
| t3 coastal post / writer-turn exhaustion | incomplete, no terminal checks | 0 |

The five completed scores prevent the scripted validation from collapsing to one reward;
the separate six-turn exhaustion run proves a valid `incomplete` path. CPU fixture actions
use only the task's admitted file tools and a final answer.

## Settings and loader

`probe_settings` in each JSON config is the one place for the probe limits and the
group-only native setting. The S10b loader applies the current limits: `max_generated_tokens`
= 1,536, six writer turns, eight tool calls, a 4,096-token context cap, and 512 tokens per
decision. No author-call budget is needed. Compute stays bounded by writer turns, the
per-decision and context caps, and the stage supervisor's ceilings; unrelated workspace/read
budgets retain the base entry fixture's constraints.

The loader at [`build.py`](build.py) is deliberately fixture-only: it reuses
`make_entry_fixture`, creates the typed config artifacts, and calls production `admit_graph`.
The tests then pass those admitted entries into `build_rollout_fixture` and the normal
`RolloutDriver`. Run the evidence with:

```bash
uv run python -m unittest tests.test_task_graph_probe_tasks -v
```

Gemma E2B should be able to attempt these tasks within 512 generated tokens per decision:
the briefs are short, the available tools are basic file operations, and the finished prose
is capped at 70 words. That is a reachability judgment, not a prediction that the model will
pass any optional check.
