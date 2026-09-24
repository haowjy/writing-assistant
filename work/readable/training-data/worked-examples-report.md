# Do the 20 reference answers pass their checks?

This report checks 20 hand-written reference answers against the project's own automated checks and records the limits of that verification. All 41 declared deterministic checks pass, but the authors mark two of the 43 required checks unmet because those checks describe a later turn than the answers cover. No broken scenario check was established: the two unmet requirements belong to the complete conversation, and all assessments of meaning remain pending in the repository scorer rather than independently judged.

## What passed, and what remains unverified

A deterministic check computes something directly, such as a word count, file edit, retrieved string, or link. A semantic check assesses meaning, such as whether a revision preserves story continuity. The 41 deterministic passes and the 43 required checks are different inventories: the report supplies author-written evidence for 41 required checks and marks two unmet; it does not claim that all 43 required checks received automated passing scores.

No API call, training, model-candidate run, or evaluation suite was used. The authored replies were put through file-tool replays and the mechanical scorer. This verifies delivery and retrieval feasibility, not a model-generated trajectory—a model's actual sequence of responses and tool calls. Independent human review of craft and semantic evidence is still appropriate before treating these assistant-authored answers as approved references. There is no remaining implementation blocker.

The examples come from existing development cases, not newly authored final-test scenarios. The five family codes describe the activity: F1 is direct prose; F2 is writing or revising files; F3 is planning; F4 is building a knowledge base (KB), meaning linked project-reference pages; and F5 is writing from that knowledge base. A case ID such as F4-04 means case 04 in the KB-building family.

In F4-04, the author marks `consequential-state` and `accepted-update` as `UNMET` at turn 0, the initial response. Both labels require the antenna to point east. The first brief forbids turning it until the ice thaws, and followup 2 supplies the thaw and authorizes the turn. The reference answer preserves the initial state and its Markdown file explains what both followups will test. The labels are not broken for the full multi-turn case; they cannot honestly be claimed satisfied by this first-turn answer.

F3-02 also answers only its first brief. Its note explains that the later followup tests reduced physical action, greater interpersonal tension, preserved diversity, and an explanation of changes.

## What the examples cover

There are 5 F1 examples, 5 F2, 3 F3, 3 F4, and 4 F5. Nine deliver in the reply and eleven in files. Fourteen stage action or dialogue rather than summarize a scene. All 11 explicit scene budgets of 120–220 words pass, with actual lengths of 152–172 words; the other three scenes are 170, 177, and 184 words.

The KBs for F4-01 and F4-04 contain 171 and 174 words respectively, below their explicit 500-word caps. Loose case F4-09 uses 175 words. “Loose” means that the brief leaves some choices to the writer; five such cases state their assumptions: F2-07, F2-10, F3-07, F4-09, and F5-07.

F5-03, F5-05, and F5-07 retrieve information from files. F5-02 is the existing condition where the KB is supplied and the answer must be a reply; it has no tools. Selecting existing development cases cannot supply the missing condition where reply delivery and tool access coexist.

Every selected artifact—the reply passage or written file being assessed—exists. Required-check evidence covers exactly each source case's required check IDs; F3-07 correctly has no required checks and an empty evidence list. Every declared artifact reference matches an original prose selector, the rule that identifies the output to score. Checks concerning paths, the whole workspace, or meaning without an artifact field keep their original contract. No selector was invented for F3 or F4.

The later role/source correction was applied as a scope check. The selected scenarios retain role `development` and their existing source identities. These are records in the requested worked-example format, so neither a new source catalog nor the scenario compiler, `compile_scenarios`, is appropriate. No compiler rejection is being hidden.

## How verification worked

`verify_examples.py` imports the repository's `mechanical_score` to compute checks, `extract_prose` to select outputs, `Workspace` and `dispatch` to execute file tools, and `markdown_graph` to inspect KB links. It seeds a temporary workspace beneath the allowed work path, performs real permitted reads, searches, and writes, records their observations, and wraps the authored response in the requested simulated result. It deletes each temporary workspace after scoring. Five real searches appear in `tool-replays.json`.

The verifier checks every replayed tool call, read-token total, conservative step total, allowed path change, selected prose extraction, and wiki link. Each example has exactly the requested eight fields. Each required-evidence entry has a Boolean `met` flag and a quotation; unmet entries include the explicit `UNMET` marker. Opening-clause comparisons normalize numbering and Markdown first. Those lexical checks support the manual craft review but cannot replace it.

All three wiki graphs have `valid_fraction = 1` and `reachability = 1`: all checked links are valid and all pages are reachable, with no orphan pages. This is an extra check for loose F4-09, whose source labels do not require `wiki-links`. Source citations use literal project paths rather than graph links because the graph checker receives only KB pages.

The output below uses `PASS` for successful mechanical checks and `semantic_pending` for the number of meaning-based checks that still lack scorer judgments. `prose_words` and `kb_words` are word counts. The summary's `author_unmet` list records the author's two first-turn failures, separately from automated scoring. Its opening counts show 20 unique primary openings and 20 unique response openings.

## Recorded scorer command and output

This is the original verification record; no scoring was rerun for this editorial rewrite.

Run from `/home/jimyao/gitrepos/research/writing-assistant`:

```bash
UV_CACHE_DIR=/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/.uv-cache PYTHONPATH=src uv run python3 /home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/verify_examples.py > /home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/verification-output.txt
```

Exit code: 0. The command redirects stdout to verification-output.txt; its complete contents follow.

```text
F1-01: word-budget=PASS; prose_words=166; semantic_pending=3
F1-02: word-budget=PASS; prose_words=171; semantic_pending=3
F1-03: word-budget=PASS; prose_words=165; semantic_pending=3
F1-04: word-budget=PASS; prose_words=172; semantic_pending=3
F1-05: word-budget=PASS; prose_words=169; semantic_pending=3
F2-02: saved-scene=PASS, edit-scope=PASS, word-budget=PASS; prose_words=167; semantic_pending=3
F2-04: saved-scene=PASS, edit-scope=PASS, word-budget=PASS; prose_words=168; semantic_pending=3
F2-05: preserve-ending=PASS, saved-scene=PASS, edit-scope=PASS, word-budget=PASS; prose_words=152; semantic_pending=3
F2-07: saved-scene=PASS, edit-scope=PASS; prose_words=170; semantic_pending=2
F2-10: saved-scene=PASS, edit-scope=PASS; prose_words=184; semantic_pending=2
F3-02: none declared; planning response; semantic_pending=1
F3-05: none declared; planning response; semantic_pending=1
F3-07: none declared; planning response; semantic_pending=0
F4-01: wiki-words=PASS, wiki-links=PASS, wiki-index=PASS, wiki-only=PASS; kb_words=171; semantic_pending=3
F4-04: wiki-words=PASS, wiki-links=PASS, wiki-index=PASS, wiki-only=PASS; kb_words=174; semantic_pending=4
F4-09: wiki-index=PASS, wiki-only=PASS; kb_words=175; semantic_pending=3
F5-02: word-budget=PASS; prose_words=159; semantic_pending=3
F5-03: retrieve-state=PASS, retrieve-belief=PASS, preserve-kb=PASS, word-budget=PASS; prose_words=163; semantic_pending=3
F5-05: retrieve-state=PASS, retrieve-belief=PASS, preserve-kb=PASS, word-budget=PASS; prose_words=161; semantic_pending=3
F5-07: retrieve-state=PASS, retrieve-belief=PASS, preserve-kb=PASS; prose_words=177; semantic_pending=2
SUMMARY {"author_unmet": ["F4-04:consequential-state", "F4-04:accepted-update"], "channels": {"file": 11, "reply": 9}, "deterministic_failed": 0, "deterministic_passed": 41, "examples": 20, "families": {"F1": 5, "F2": 5, "F3": 3, "F4": 3, "F5": 4}, "loose_examples": 5, "required_checks": 43, "scene_examples": 14, "search_calls": 5, "unique_primary_openings": 20, "unique_response_openings": 20}
```

## Problems encountered during verification

The first unmodified `PYTHONPATH=src uv run python3` invocation exited 2 before Python ran:

```text
error: failed to open file `/home/jimyao/.cache/uv/sdists-v9/.git`: Read-only file system (os error 30)
```

Putting `UV_CACHE_DIR` inside the work directory resolved the cache problem. The first completed scoring pass already passed all 41 checks, but the separate response-opening audit exited 1 because it treated the numbered-list prefix `1.` as a shared opening clause in F3-02 and F3-05. Normalizing list markers fixed that auxiliary audit, and the recorded later run passed. No example check was suppressed or relabeled.

## What the craft review examined

The 14 scene openings use a command, negative statement, conditional question, infinitive subject, interrupted task, tactile location, prohibition, timed pen movement, spatial inversion, verbless inventory, unidentified visitor, remembered promise to oneself, cleft construction, and temporal clause. Their first images are distinct. Planning alternatives start with different decisions, and wiki headings identify different projects; required list or document formatting is not treated as a prose opening.

The table preserves the exact primary openings used in that comparison. It shows the observed opening variety, rather than an independent judgment of literary quality.

| Case | Primary opening |
|---|---|
| F1-01 | then tell me which shelf |
| F1-02 | no answer came from the vault office |
| F1-03 | if i ask the captain |
| F1-04 | to answer would take one press of nessa's thumb |
| F1-05 | halfway through tying the deposition bundle |
| F2-02 | under suri's fingertip |
| F2-04 | don't use my name on the radio |
| F2-05 | dara set her pen across the inkwell before the last word could dry |
| F2-07 | beyond the garden wall lay a road yara had never walked |
| F2-10 | two saucers for tasting |
| F3-02 | test the boundary of a report |
| F3-05 | a place held open |
| F3-07 | for these options |
| F4-01 | the tidal archive |
| F4-04 | winter radio |
| F4-09 | the hollow clock |
| F5-02 | whoever stood beyond the orchard gate had kept out of the light |
| F5-03 | one more crossing |
| F5-05 | it was from beneath the witness bench that the courthouse spoke |
| F5-07 | by the time belen asked about the letter |


Manual continuity review compared each scene with its own visible notes, not information disclosed only in another variant. No protected reserve is spent, no sealed original is accessed without permission, and untested theories remain attributed. Local revision F2-05 preserves its final sentence outside the prose delimiters. File-completion notes correspond to actual replayed writes.

## Delivered files

The combined deliverable is `worked-examples.json`, using schema version 1. Twenty individual Markdown files, beginning with `worked-examples/F1-01.md` and including the 20 linked cases, contain the brief, exact reply, complete written-file contents, quoted required-check evidence, craft rationale, and any followup notes. `worked-examples.md` indexes them, and the original `worked-examples-report.md` reports the work.

`verification-results.json` and `verification-output.txt` hold complete mechanical scorecards, counts, the opening audit, and actual output. `tool-replays.json` records offline tool observations and workspace snapshots before and after changes. `example-notes.json` supplies the first-turn scope annotations. The helpers `build_examples.py`, `verify_examples.py`, and `finish_examples.py` make authoring, rendering, and verification reproducible.

All authored files are under the requested work path. No source, tests, scenario, training-data, or runs files were edited, and existing unrelated work artifacts were preserved.

## Recorded final parse and existence check

Exact command (the script contains the complete `PYTHONPATH=src uv run python3` parse loop and assertions):

```bash
bash /home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/validate_delivery.sh
```

Exit code: 0. Actual stdout:

```text
JSON_PARSE_OK worked-examples.json
JSON_PARSE_OK example-notes.json
JSON_PARSE_OK verification-results.json
JSON_PARSE_OK tool-replays.json
MARKDOWN_OK 20 case files, 20 index targets, nonempty index and report
DELIVERY_NONEMPTY_OK 31 authored files checked
CONTRACT_OK 20 examples, 5 families, 2 channels, 43 required checks accounted for
SCORING_OK 41 deterministic passes, 0 failures; 2 first-turn semantic requirements explicitly unmet
```

The 31 checked files comprise four JSON artifacts, 20 individual examples, the index and report, the captured scorer output, three Python helpers, and `validate_delivery.sh`. Tool-cache internals are not deliverables. The JSON/Markdown content equality checks are in `verify_examples.py`.
