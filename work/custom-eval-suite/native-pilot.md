# E2B-IT native tool rerun

One of three cases executed tools. None completed the requested file task.

| Case | Task | Executed tools | Observed result |
|---|---|---|---|
| F2-06 | Notes to comedy scene | 0 | Asked for notes/source.md contents; created no scene. |
| F4-08 | Chapter to wiki, with revisions | 0 | Asked for chapter contents or confirmation across the three scripted turns; created no wiki. |
| F5-05 | Wiki to fantasy scene | 2 | Read kb/index.md, skipped its linked pages, then wrote Drafts/scene.md rather than the requested lowercase drafts/scene.md. |

The original pilot executed zero tools in these cases. This rerun demonstrates
native tool execution in a real suite case, but tool selection, navigation, and
path accuracy remain unreliable. Three samples do not establish a success rate.

The checkpoint, NF4/BF16 loading, prompts, temperature 0.7, seed 42, and budgets
match the original pilot; the harness uses the native Gemma protocol at commit
`8b5ae84`. Generation loops took about 38.2 seconds in total, excluding loading.
There were no execution errors, automatic retries, or Astra calls. Execution
completion only means the loop ended. Semantic and prose quality remain ungraded.

- [All conversations, model inputs, raw outputs, and files](../../runs/pilot-e2b-it-native-2026-09-13/review.md).
- [Recorded results](native-pilot.json).
- [Original pilot](pilot-e2b.md).
- [Native protocol smoke](native-inference.md).

The next diagnostic should test whether explicit workspace instructions change
tool selection while preserving these original results. The wiki case also needs
review of retrieval behavior: reading an index alone does not expose its linked
facts to the model.
