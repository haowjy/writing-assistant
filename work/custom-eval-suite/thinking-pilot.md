# Five-case E2B-IT pilot with thinking

All three workspace cases used tools. All five execution loops completed without
errors, with seven tool calls in total. Semantic judgments remain pending.

| Case | Task | Tools | Result |
|---|---|---|---|
| F1-01 | Direct prose | 0 | Returned prose; no failed mechanical checks. |
| F2-06 | Notes to scene | 2 | Read notes and saved drafts/scene.md; no failed mechanical checks. |
| F3-03 | Brainstorming | 0 | Returned ideas; no failed mechanical checks. |
| F4-08 | Chapter to wiki | 3 | Read chapter, wrote kb/index.md, patched the accepted revision; no failed mechanical checks. |
| F5-05 | Wiki to prose | 2 | Read only the index, wrote drafts/scene.md; failed two evidence-retrieval checks and the word budget. |

F4 produced a single Markdown page. Manual inspection found questionable
interpretations, including treating the dried stamp pad as a clue. Passing its
mechanical checks does not establish extraction accuracy or wiki quality.

The run used the same five prompts, E2B-IT checkpoint, NF4/BF16 loading, sampling
settings, and budgets as the previous pilot. Thinking was enabled, and the harness
preserved it between tool calls. Total generation-loop time was about 392.3 seconds
(6.5 minutes), excluding loading and report generation. Thinking and answer tokens
share the 2,048-token per-call output budget. There were no automatic retries,
Astra calls, or external benchmark runs.

The earlier native run with thinking disabled used tools in one of its three
workspace cases; this run used tools in all three. This is a small observed
comparison, not a stable estimate of improvement.

- [Review all five cases, thinking, raw prompts/outputs, and files](../../runs/pilot-e2b-it-thinking-2026-09-13/review.md).
- [Recorded summary](thinking-pilot.json).
- [Native tools without thinking](native-pilot.md).
