# Phase 8 public probe tasks

These three original, single-writer tasks are the public fixture set for the first
integrated GPU probe. Each asks for a short file-based scene, exposes one decision through
`ask_author`, and supplies one scripted feedback turn. Only `scene_nonempty` is required;
the four optional checks reward partial completion at different levels. These checks test
probe plumbing, not writing quality.

## Provenance and scope

The briefs, seed files, feedback, public decision packets, and scripted sample prose in the
three JSON files are original fixture material. There is no catalog selection, book or
other private source, or data shared between members. The public author
packet answer is also in each committed config. The builder wraps that public value in the
runtime's `AuthorPacketV1` artifact and uses the scripted author only behind the existing
role interface; it does not introduce hidden answer data.

| Task config | Public writing prompt |
|---|---|
| [`t1-lighthouse.json`](t1-lighthouse.json) | A lighthouse keeper preparing for a boat at low tide |
| [`t2-winter-garden.json`](t2-winter-garden.json) | A winter greenhouse before dawn |
| [`t3-coastal-post.json`](t3-coastal-post.json) | A coastal post runner arriving before a storm |

## Reward spread from scripted candidates

Each graph uses the same deterministic weights: required `scene_nonempty` = 1,000;
disclosed-decision phrase = 2,500; public detail = 2,000; 24–70 word range = 2,500; and
optional `field-notes.txt` nonempty = 2,000. Scores are exact basis points out of 10,000.
The scripted candidates exercise each optional layer in order:

| Task / scripted lineage | Checks passed | Reward |
|---|---|---:|
| t1 lighthouse / `nonempty_only` | required only | 1,000 |
| t1 lighthouse / `decision_phrase` | required + disclosed phrase | 3,500 |
| t2 winter garden / `phrase_and_detail` | required + phrase + public detail | 5,500 |
| t3 coastal post / `word_range` | required + phrase + detail + word range | 8,000 |
| t3 coastal post / `all_optional` | required + all optional checks | 10,000 |
| t3 coastal post / writer-turn exhaustion | incomplete, no terminal checks | 0 |

The five completed scores prevent the scripted validation from collapsing to one reward;
the separate six-turn exhaustion run proves a valid `incomplete` path. Every completed
scripted path asks the author and receives both the public answer and feedback.

## Settings and loader

`probe_settings` in each JSON config is the one place for the probe limits and the
group-only native setting. The S10 loader applies fields present in the current contracts:
`max_generated_tokens=1536`, `max_total_tokens=26112`, six writer turns, eight tool calls,
and two author calls (one answer plus one feedback). The aggregate total-token allowance is
the no-earlier-stop ceiling `6 × 4096 + 1536`; unrelated workspace/read budgets retain the
base entry fixture's constraints.

Two declared values cannot yet bind to runtime records: `max_context_tokens=4096` waits for
S4's context-limit field, and `training_mode="native"` waits for S3's group-contract field.
Both values are already in `probe_settings`; S4 maps the former into the budget, and the
later group builder maps the latter into `GroupSpecV1`. The 512-token per-decision cap is
also declared there for the native decoding descriptor. No graph or production contract was
extended ahead of those core steps.

The loader at [`build.py`](build.py) is deliberately fixture-only: it reuses
`make_entry_fixture`, creates the typed config artifacts, and calls production `admit_graph`.
The tests then pass those admitted entries into `build_rollout_fixture` and the normal
`RolloutDriver`. Run the evidence with:

```bash
uv run python -m unittest tests.test_task_graph_probe_tasks -v
```

Gemma E2B should be able to attempt these tasks within 512 generated tokens per decision
before any task-graph update:
the briefs are short, there is one bounded public choice, the available tools are basic
file operations, and the finished prose is capped at 70 words. That is a reachability
judgment, not a prediction that the model will pass any optional check.
