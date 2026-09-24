# What the model does when the brief leaves choices open

This report examines whether vague briefs lead the model to ask questions, state assumptions or silently choose defaults. In all 40 loose-brief attempts outside the alternatives family, it silently chooses: it asks no clarifying question and states no default, while usually producing longer work than it does under explicit instructions.

The study compares Google's `google/gemma-4-E2B-it` on the same 50 scenarios and seeds in two runs, one attempt per scenario per run. `nf4` means weights compressed to 4 bits; `bf16` means bfloat16, the full-precision comparison. The nf4 run is `runs/custom50-e2b-it-2026-09-14` and has language-model grades. The bf16 run is `runs/custom50-e2b-it-2026-09-21-bf16` and has no such grades. These 100 attempts are called the `custom50` runs.

Instruction specificity means how many decisions the brief settles. Explicit briefs state more of the length, viewpoint, style and format; loose briefs leave more choices open. Point of view (POV) means whose experience the narration follows. Close third person uses third-person narration centered on one character.

The task families are F1, scenes in the reply; F2, revision or drafting saved to a file; F3, alternative directions or feedback; F4, building or updating a Markdown knowledge base (KB); and F5, using project facts supplied in the brief or retrieved from files. An identifier such as F2-08 names case 08 within family F2. A deliverable, also called an artifact in the records, is the requested scene, set of directions or knowledge-base files. The reply channel is the chat message; the file channel is a workspace file. Two further distribution runs repeat the explicit F1-01 brief 25 times per format to measure variation when the brief stays fixed.

The source reads briefs in `visible.json`, behavior and before/after workspace state in `result.json`, and the actual workspace files. It changes no source, tests or run files. Fractions retain their original denominators; some later tables exclude the one crash.

## Loose briefs omit length and viewpoint, but often still name a file

The 50 briefs share an identical `selection.json` across formats. Their measured differences show what the experiment varies: the loose briefs drop quantitative limits and narration choices, while file access remains almost evenly distributed.

| Property of the brief | explicit (n=25) | loose (n=25) |
| --- | --- | --- |
| mean brief length | 108.2 words | 77.5 words |
| brief length range | 57–185 words | 25–135 words |
| states a word-count target ("120–220 words") | 20 / 25 | 0 / 25 |
| states a POV / narration distance ("close third … centered on X") | 15 / 25 | 0 / 25 |
| names a file / deliverable path (`drafts/…`, `kb/…`, `source/…`, `*.md`) | 13 / 25 | 12 / 25 |
| scenario ships `initial_files` (a workspace to read) | 13 / 25 | 12 / 25 |
| offers tools to the model | 13 / 25 | 12 / 25 |
| has scripted follow-up turns | 4 / 25 | 6 / 25 |

All 25 loose briefs omit a length target and viewpoint specification. Twenty of 25 explicit briefs state “120–220 words,” and 15/25 state viewpoint. Loose wording substitutes requests such as “feel more alive,” “rework it” or “what are some directions” for those details. Twelve loose briefs still name a file, compared with 13 explicit briefs; a loose brief therefore need not leave the delivery destination open.

The source compares these F1 examples, described as the same action/scene condition:

> Explicit F1-01: “… Write 120–220 words about Mara and Ilan in a tidal archive, in restrained, concrete prose. Use close third person centered on Mara. End before the central uncertainty is resolved. … Stage a disagreement over the object. Return only the prose, without a title or commentary.”

> Loose F1-06: “… Could you turn this into a scene? I'd like it to feel more alive. Just the story, please.”

Explicit F5-01 says “Save prose in `drafts/scene.md`; keep the KB unchanged.” Loose F5-09 also names the destination: “Could you write a scene using the notes in `kb/`? Save it in drafts/scene.md. Keep the notes as they are for now.” It leaves the choice of notes open. The source connects this kind of directory-level request to the directory-read error described below, but the actual error case is F5-07, not F5-09.

Loose F2-10 asks for a scene from notes, saved to `drafts/scene.md`, with the notes left alone; it specifies neither length nor viewpoint. Loose F4-09 asks for a small Markdown wiki from `source/chapter.md`, starting at `kb/index.md`, with things easy to find. It gives no page, word or structural limit. The compared explicit F4-01 asks for at least one linked topic page and no more than 500 words.

The comparison is not fully controlled. Within F2, `explore_then_write`—reading notes before drafting—has two explicit and three loose cases. F3 `alternatives` has three explicit and two loose; F4 `construction`, creating a wiki, has three explicit and two loose; F5 `retrieved_kb`, reading facts from files, has three explicit and two loose. These one-case imbalances mean a family-level comparison also changes the mix of conditions. F2's other condition, `local_revision`, revises an existing file; F3 also has feedback cases, and F4 has update cases.

F3 is a request about possible story directions at both levels. Explicit briefs ask for “three numbered” directions explaining choice, consequence and fit. Loose briefs ask “I'm not sure where to take this. What are some directions I could try?” Questions and recommendations in F3 therefore answer an invitation to discuss choices; they are not evidence that the model independently notices a missing instruction.

## Nearly every attempt finishes, without stopping to clarify

Ninety-nine of 100 attempts complete and reach a final answer. The one exception is nf4 F4-02, an explicit wiki-update task whose tool-call payload triggers `ValueError: json: could not parse after dialect transforms`, leaving `status: "error"` and `output: ""` after three issued calls.

| Weight format | level | completed | reached a final answer |
| --- | --- | --- | --- |
| nf4 | explicit | 24 / 25 | 24 / 25 |
| nf4 | loose | 25 / 25 | 25 / 25 |
| bf16 | explicit | 25 / 25 | 25 / 25 |
| bf16 | loose | 25 / 25 | 25 / 25 |

The source treats the crash as a protocol or quantization artifact rather than a behavioral response to the brief. Parsing failure is observed; the proposed quantization explanation is not tested here. No attempt asks a question and then stalls without producing anything: every completed attempt provides reply prose or a workspace file.

## Loose briefs usually produce longer deliverables

The length measurement uses the reply when `prose.kind == "reply"`, `drafts/scene.md` when `prose.kind == "file"`, and the sum of knowledge-base pages for F4. The table shows means of 487 versus 288 words in nf4 and 399 versus 240 in bf16: loose deliverables average approximately 1.7 times as long. The mechanism proposed by the source is the removal of explicit limits, allowing the model's longer default output.

| Weight format | level | mean | median | range |
| --- | --- | --- | --- | --- |
| nf4 | explicit | 288 | 274 | 0–661 |
| nf4 | loose | 487 | 424 | 96–902 |
| bf16 | explicit | 240 | 235 | 9–486 |
| bf16 | loose | 399 | 399 | 92–898 |

The family means reveal an exception that the source's initial “every level/family” claim overlooks: F4 wikis are shorter under loose briefs in both formats. All other families are longer.

| Family | Nf4 explicit mean | Nf4 loose mean | Bf16 explicit mean | Bf16 loose mean |
|---|---:|---:|---:|---:|
| F1 scenes | 256 | 543 | 206 | 486 |
| F2 file revision/drafting | 62 | 293 | 139 | 245 |
| F3 directions | 555 | 841 | 413 | 689 |
| F4 wiki | 366 | 283 | 220 | 211 |
| F5 fact-based scenes | 217 | 474 | 223 | 362 |

Each family/level has five scenarios. The source describes most scene/direction comparisons as roughly 1.5–2.4 times longer; its cited comparisons include nf4 F1, F3 and F5, and bf16 F1, F2, F3 and F5. It excludes the much larger nf4 F2 ratio from that description. Missing or unchanged destination files affect F2's artifact lengths, so its low explicit mean is not simply a preferred scene length. F1 is the clearest example: 206 versus 486 words in bf16 and 256 versus 543 in nf4, each based on five cases per level. F4 differs and has its own page budget; these observations do not isolate the reason for its reversal.

## Extra turns are mostly built into the scenarios

The source reports slightly more multi-turn attempts under loose briefs, six of 25 versus three or four of 25 explicit attempts. Tool volume is nearly flat. Scripted follow-ups are also more common in loose cases—six versus four—so more turns do not by themselves show the model seeking clarification.

| Weight format | level | mean turns | mean tool calls | attempts with 0 tool calls | multi-turn attempts | total tool errors |
| --- | --- | --- | --- | --- | --- | --- |
| nf4 | explicit | 1.16 | 1.44 | 12 / 24 | 3 / 24 | 0 |
| nf4 | loose | 1.36 | 1.48 | 13 / 25 | 6 / 25 | 1 |
| bf16 | explicit | 1.24 | 1.72 | 12 / 25 | 4 / 25 | 0 |
| bf16 | loose | 1.36 | 1.68 | 13 / 25 | 6 / 25 | 1 |

The table mixes denominators as reported: nf4 explicit zero-call and multi-turn rates use 24 completed attempts, whereas the section's general description says 25 per level. The source does not fully explain the denominator for its means.

The only tool error in each format occurs in loose F5-07: `read_file(path="kb/")` receives `[Errno 21] Is a directory: …/workspace/kb`. The model then uses `list_dir`, which lists files, and completes. The source attributes this to a brief naming a directory without choosing a note file. That supplies the opportunity for the mistake; it does not demonstrate that loose wording necessarily causes it.

## The model asks about directions, not omitted length or viewpoint

Reading final replies and excluding question marks inside fictional dialogue yields no user-directed question in any explicit case: 0/25 in each format. Loose replies end with such questions in 3/25 bf16 cases (F3-07, F3-08, F3-10) and 2/25 nf4 cases (F3-07, F3-10). All are in F3, whose task already asks for directions.

Outside F3, no loose attempt asks the user to settle omitted length, viewpoint or scope. “Let me know if you'd like revisions” appears in some loose F2/F4 replies, but offers later service rather than asking about a missing decision, so it is not counted.

The source separately classifies each completed attempt by what it does with the most open decision. Its labels mean:

- ASK: it asks the user to make the unstated choice.
- SAY: it chooses a default and tells the user that choice.
- SILENT: it chooses without mentioning the choice.
- IGNORE: it fails to address or produce the deliverable.

For loose F1 the decision is scene length, viewpoint or structure; for F2, how to make or rework the scene, including length/viewpoint; for F4, wiki structure or page budget; for F5, which notes and what scene length/viewpoint; and for F3, which direction to pursue. Explicit cases are recorded as `SILENT (specified)`: the decision is already stated, so no disclosure is expected. That label must not be interpreted as a defect in explicit cases.

The classification covers 99 final-answer attempts, excluding the nf4 F4-02 crash. Almost all are silent; every ASK or SAY belongs to loose F3.

| Weight format | ASK | SAY | SILENT | IGNORE | total final-answer attempts |
| --- | --- | --- | --- | --- | --- |
| nf4 | 3 | 2 | 44 | 0 | 49 |
| bf16 | 4 | 1 | 45 | 0 | 50 |
| both | 7 | 3 | 89 | 0 | 99 |

Nf4 ASK cases are F3-06, F3-07 and F3-10, or 3/49 completed attempts. Bf16 ASK cases are F3-06, F3-07, F3-08 and F3-10, or 4/50. Nf4 SAY cases are F3-08 and F3-09 (2/49), which recommend a direction; bf16 SAY is F3-09 (1/50), which recommends a blend. Nf4 SILENT combines all 20 non-F3 loose cases with all 24 completed explicit cases, giving 44/49. Bf16 combines 20 and 25, giving 45/50. IGNORE is zero.

The decisive comparison therefore uses only the 40 non-F3 loose attempts: ASK 0/40, SAY 0/40 and SILENT 40/40. The ASK totals are larger than the earlier direct-question counts because they also include F3-06 in each format. The source does not provide an operational explanation for that difference, so the two counts are retained as separate measures.

## Examples show what asking, stating and silently choosing look like

Bf16 loose F3-08 illustrates ASK:

> “Which direction feels most compelling to you? Do you want the tension to come from a secret they share (Direction 1), a secret one of them is hiding (Direction 2), or the external pressure forcing them to choose their bond (Direction 3)?”

Bf16 loose F3-09 illustrates SAY:

> “If you want immediate momentum, I recommend blending Direction 2 and 3. Start with the psychological tension (Pia vs. Ren) to establish the immediate stakes, and use the object mystery (the watch/key) as the mechanism …”

Bf16 loose F2-08 illustrates SILENT. Its brief asks for a scene from notes saved to `drafts/scene.md`, leaving the notes alone. The final reply is six words: “Scene created and saved to `drafts/scene.md`.” The file is 2,289 bytes, approximately 275 words. Length, viewpoint, tense and scene boundaries are chosen without being disclosed.

The source also quotes model deliberation showing that a missing choice can be noticed but left unstated. Bf16 F5-09 considers whether to read `story-context.md` or `canon.md`, or ask for clarification, then decides the request to write from `kb/` implies proceeding. It reads one file and gives a 15-word completion note. Nf4 F2-09 similarly says the unspecified way to rework the text calls for clarification or a reasonable creative choice, but interprets the instruction as asking it to do the rework. Nf4 F5-07 considers listing the directory, asking which file was meant, or assuming a file inside it; it chooses listing as the helpful next step.

These are three quoted examples in the original, not a count of how frequently that deliberation occurs. They support the proposed mechanism: the model notices ambiguity but gives the imperative to proceed greater weight than discussing the choice. The source's broader “frequently” claim is unquantified.

No completed attempt is classified IGNORE even when it violates the delivery instruction. Nf4 F2-01, F2-03 and F2-05 put tagged prose in chat and leave the file unchanged; nf4 F5-03 writes zero words to the requested file. They still produce prose and final answers, so the classification records SILENT on the open decision and a separate failure on the stated delivery rule. A zero IGNORE count therefore does not imply successful file delivery.

## Explicit instructions are only partly followed

For the 15 explicit prose cases in each format, the source extracts text as the scenario declares: tagged `<prose>…</prose>` content for F2/F5-02/F5-04 replies, whole replies for F1, or the scene file when designated. It reports 3/15 nf4 and 7/15 bf16 outputs inside 120–220 words, a combined 10/30, or 33%.

This differs from the failure report's scored-check counts of one nf4 pass and six bf16 passes, with missing prose excluded from scored denominators. The source's extraction wording also mentions replies for cases whose contract requires files. The exact reconciliation needs a common extraction and counting procedure; this rewrite does not choose one result over the other.

The source describes out-of-range values of 233–296 words in nf4 and 225–241 in bf16, mostly just above the cap. It cites bf16 F1-03 at 227 and F1-04 at 238 as narrow misses, although the failure report counts 226 and 237. These show overshoot rather than complete disregard for the target, but not every missing-file case has a measurable over-length deliverable.

For the three explicit F2 local-revision cases in each format, the contract requires replacement prose between `<prose>` and `</prose>` in the file, retaining the final sentence after the closing tag. Only bf16 F2-03 succeeds: 1/6 overall, 0/3 nf4 and 1/3 bf16. Its changed file has 245 words, tags and the retained final line; total file words include more than the extracted replacement prose.

Nf4 F2-01/F2-03/F2-05 and bf16 F2-01/F2-05 instead put the tagged block in the reply and leave the two-line draft alone. Nf4 F2-01 begins its reply `<prose>Mara's hand settled on the cold iron of the latch…` while the file remains `"Mara opened the door.\nThe harbor bell rang twice.\n"`. The model generates the requested text but selects the wrong delivery route.

For explicit F5 retrieval cases F5-01, F5-03 and F5-05, the knowledge base is unchanged in all six attempts. The scene file is written in 2/3 nf4 and 3/3 bf16 cases. Nf4 F5-03 places a 248-word scene only in its reply and zero words in the file. This separates a respected preservation rule from a failed delivery rule.

All six measured explicit F4 construction cases stay under the 500-word total knowledge-base budget: nf4 328–439 words, bf16 126–249. Explicit F4 also consistently cites `source/chapter.md` and leaves source files unchanged. The model can therefore comply with a budget and preservation instructions in this task, despite frequent failures on scene length and routing.

## The update findings conflict with another report

F4 update cases first receive “Draft-only possibility… Do not update canon from this suggestion,” then an accepted revision. Canon means accepted story facts. Nine attempts reach final answers: bf16 F4-02, F4-04 in both formats, and loose F4-06/F4-08/F4-10 in both formats. This source reports 9/9 applying the accepted revision and says all nine also preserve the draft-versus-accepted distinction.

It cites bf16 F4-02 storing the proposal on a separate page labelled “DRAFT ONLY” and “not” canon. It also attributes a `kb/clues.md` entry tagged “(Draft)” to nf4 F4-06. The tool-use report attributes that patch to bf16 F4-06 and describes nf4 F4-06 as creating `drafts/chapter_suggestions.md` instead.

More seriously, tool-use records bf16 F4-08 replacing the established fact that Noor retains her residence papers with an unaccepted alternative in which she loses them. That directly conflicts with this report's “all nine” conclusion. Applying an accepted revision and retaining all required history are also different tests: the failure report records three nf4 accepted-update failures. The source's 9/9 application count is retained, but it cannot establish flawless canon discipline or full accepted-update check success.

## Viewpoint appears similar, but pronoun counts cannot prove close third

Fifteen of 25 explicit briefs specify close third centered on a named character; zero of 25 loose briefs do. First-person pronouns in F1 range from zero to two per explicit attempt and zero to three per loose attempt in both formats; loose occurrences are inside dialogue. The source concludes that close third is the default in either case, so specifying it makes little observed difference.

The evidence is limited: counting first-person pronouns can distinguish some surface narration choices, but cannot establish whose knowledge third-person narration enters. The failure report gives a possible viewpoint slip in F5-02. The source's practical conclusion is preserved as an interpretation, not presented as a validated viewpoint test.

## Repeating one explicit brief shows a weight-format difference

The distribution runs repeat F1-01, which specifies 120–220 words, close third on Mara and prose-only output. Each has 25 completions, all with one turn and zero tool calls. The fixed brief produces much tighter length compliance in bf16:

| Weight format | completed | mean artifact words | median | range | in 120–220 |
| --- | --- | --- | --- | --- | --- |
| nf4 | 25 / 25 | 238 | 239 | 197–280 | 7 / 25 |
| bf16 | 25 / 25 | 185 | 187 | 160–211 | 25 / 25 |

Bf16 stays in range on all 25 repetitions; nf4 stays in range on seven. The source describes nf4 as tending to overshoot by approximately 20–60 words, although its range also includes compliant outputs. It concludes that bf16 can meet a well-specified length request and that quantization contributes to explicit-length failures.

The source also claims this control confirms that the loose-versus-explicit length gap is caused by specificity rather than scenario noise. Repeating only one explicit scenario does not isolate that causal effect: there is no repeated loose counterpart, and the cross-level task mix differs. The control demonstrates within-scenario variation and a format difference; the stronger claim remains unresolved in the audit.

## What a later training comparison should measure

The proposed training target is to make the model ask about an important open choice or state the assumption it is using on loose briefs. The observed baseline is zero disclosures in 40 non-F3 loose attempts. On explicit briefs, the source proposes improving on its reported 33% length-compliance rate and controlling separately for nf4 overshoot. A valid later comparison will need to reconcile the extraction/counting differences above and score file delivery independently from generating prose somewhere.

The original analysis is [specificity.md](../../transcript-analysis/specificity.md). The conflicting update examples, unmatched counting methods and limits of the repeated-brief control are listed for research review in [the audit](../lane-a-audit.md).
