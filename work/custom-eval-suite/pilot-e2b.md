# E2B-IT five-case pilot

Executed on 2026-09-12 after checkpoint commit `802df4a`. Five cases ran once each
on `google/gemma-4-E2B-it`, pinned to
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, with the `writing-tools-v2` protocol.
The full comparison was not run. No Astra calls or automatic retries occurred.

Start with the [artifact review index](../../runs/pilot-e2b-it-2026-09-12/review.md).
It links each conversation, raw trace, file snapshot, extracted prose, and scorecard.
[Exact inputs and private labels](../../runs/pilot-e2b-it-2026-09-12/selection.json)
are also saved. Raw artifacts remain local under the git-ignored `runs/` directory;
this summary and the [machine-readable results](pilot-e2b.json) are versioned.

| Case | Task | Observed outcome |
|---|---|---|
| F1-01 | Explicit literary prose | Returned a scene; failed the declared word-range check. Semantic judgments remain pending. |
| F2-06 | Loose comedy file authoring | Asked the user to provide the notes instead of reading them. No manuscript file was created. |
| F3-03 | Explicit historical brainstorming | Returned three directions. Semantic usefulness, diversity, and adherence remain ungraded. |
| F4-08 | Loose science-fiction wiki build/update | Asked for source contents across the scripted conversation. No KB was created. |
| F5-05 | Explicit fantasy writing from a KB | Returned a draft in chat without retrieving the KB or writing the requested manuscript file. |

All five execution loops ended normally, but all three file/tool tasks failed
artifact delivery. There were seven model responses and zero tool calls. This is
an observation about this model under the current prompted tool protocol, not a
claim about its capability under native function calling or other harnesses.

Next: review the saved outputs and inputs, then investigate why tool availability
was not acted on before expanding the experiment. The relevant software is
[the pilot script](../../scripts/pilot_e2b.py); its default is inspection. The
`review_results` function rebuilds the review from saved results without generation.
