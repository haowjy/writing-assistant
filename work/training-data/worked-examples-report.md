# Worked-example delivery and verification

**Delivered 20 examples; all 41 declared deterministic checks pass.** There are 43 required checks: 41 have quoted author evidence and two are explicitly unmet at the requested first turn. Semantic assessments are not independent judge results. The repository scorer leaves all semantic checks pending; no API, training, candidate run, or evaluation suite was used.

## Coverage and decisions

- Families: F1 = 5, F2 = 5, F3 = 3, F4 = 3, F5 = 4.
- Channels: 9 reply, 11 file. Fourteen examples stage action or dialogue rather than summarize a scene.
- All 11 explicit 120–220-word scene budgets pass (actual range: 152–172 words). The remaining three scenes are 170, 177, and 184 words.
- Explicit wiki caps: F4-01 = 171 words; F4-04 = 174 words; both below 500. Loose F4-09 uses 175 words.
- Five loose cases state their assumptions: F2-07, F2-10, F3-07, F4-09, F5-07.
- F5-03, F5-05, and F5-07 retrieve from files. F5-02 is the existing supplied-KB reply condition, which has no tools. Selecting existing dev cases cannot fill the absent reply-with-tools cell.
- Every selected artifact exists. Required-check evidence covers exactly the source case’s required check IDs; F3-07 correctly has an empty list.
- Every declared artifact reference matches a source prose selector. Path/global and semantic checks without an artifact field retain their original contract; no selector was invented for F3 or F4.

## Scope exceptions

**F4-04: consequential-state and accepted-update are UNMET at turn 0.** Both labels require the antenna to point east. The first brief says it cannot turn until the ice thaws; followup 2 supplies the thaw and authorizes turning it. The example preserves the first-brief state and reports these labels openly. They are not broken for the complete multi-turn case. The Markdown example explains what both followups will test.

F3-02 answers only its first brief. Its example notes that the followup tests reduced physical action, increased interpersonal tension, preserved diversity, and an explanation of the changes.

The later role/source correction was applied as a scope check: the selected source scenarios retain role `development` and their existing source identities. These records are the requested worked-example schema, not newly authored scenarios, so no catalog or compile_scenarios call is appropriate. No compiler rejection is being concealed.

## Verification method

verify_examples.py imports the repository’s mechanical_score, extract_prose, Workspace, dispatch, and markdown_graph. It seeds a temporary workspace beneath the allowed work path, performs real allowed reads/searches/writes, records their actual observations, and creates the requested fake result around the authored response. Temporary workspaces are deleted after scoring. Five real search calls appear in tool-replays.json. This establishes delivery and retrieval feasibility; it does not represent a model-generated trajectory.

All replay tool calls, read-token totals, conservative step totals, allowed path changes, selected prose extraction, and wiki links are checked. Each record has the exact requested eight fields. Each required evidence entry has a Boolean met flag and a quote; unmet entries contain an explicit UNMET marker. Opening-clause checks normalize numbering and Markdown before comparison. These lexical checks support, but do not replace, the manual craft review.

All three wiki graphs have valid_fraction = 1 and reachability = 1, with no orphan pages. For loose F4-09 this is an additional check, because its source labels do not declare wiki-links. The source citations are literal project paths rather than graph links, since the graph checker only receives KB pages.

## Exact verification command and actual stdout

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

## Earlier verification issues

The first unmodified `PYTHONPATH=src uv run python3` invocation exited 2 before running Python:

```text
error: failed to open file `/home/jimyao/.cache/uv/sdists-v9/.git`: Read-only file system (os error 30)
```

Setting UV_CACHE_DIR inside this work directory resolved it. The first completed scoring pass passed all 41 checks but the auxiliary response-opening audit exited 1: it treated the numeric list prefix `1.` as a clause shared by F3-02 and F3-05. Normalizing list markers fixed that audit; the later run above passed. No example check was suppressed or relabeled to fix it.

## Opening and craft review

The 14 scene openings use a command, a negative statement, a conditional question, an infinitive subject, an interrupted task, a tactile location, a prohibition, a timed pen movement, spatial inversion, a verbless inventory, an unidentified visitor, a remembered promise to oneself, a cleft construction, and a temporal clause. Their first images are distinct. Alternatives begin with different decisions; wiki headings identify different projects. Required document/list formatting is not counted as a prose opening.

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

Manual continuity review checked each scene against its own visible notes, not against information available only in other variants. No protected reserve is spent, no sealed original is accessed without permission, and untested theories remain attributed. Local revision F2-05 keeps its final sentence outside the prose delimiters. File completion notes correspond to actual replayed writes.

## Files created

- worked-examples.json: combined schema-version-1 deliverable.
- worked-examples/F1-01.md through the 20 linked case files: brief, exact reply, complete written-file contents, quoted required-check evidence, craft rationale, and followup notes where applicable.
- worked-examples.md: short index.
- worked-examples-report.md: this report.
- verification-results.json, verification-output.txt: complete mechanical scorecards, counts, opening audit, and actual stdout.
- tool-replays.json: actual offline file-tool observations and before/after snapshots.
- example-notes.json: first-turn scope annotations used in the readable files.
- build_examples.py, verify_examples.py, finish_examples.py: reproducible authoring/rendering and verification helpers.

All authored files are under the requested work path. No source, tests, scenario, training-data, or runs files were edited. Existing unrelated work artifacts were preserved. No remaining implementation blocker; independent human review of craft and semantic evidence remains appropriate before treating these assistant-authored targets as approved reference material.

## Final delivery parse and existence check

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
