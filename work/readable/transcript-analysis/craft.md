# Why the generated scenes read as weak

This report explains what is weak in the model's sentences and scenes, using quotations and counts so the problems are visible. The recurring problem is that the model describes a mood several times instead of developing an event: it adds explanations, familiar images and abstract endings where an action or exchange could change the scene.

The model is Google's `google/gemma-4-E2B-it`, called E2B in the source report. It was tested with weights stored in two formats: `nf4`, a compressed 4-bit format, and `bf16`, short for bfloat16, the full-precision comparison. A run is one execution of the test set in one format. The main evidence here comes from nf4; bf16 supplies comparison texts, not a second set of prose-quality grades.

The language-model judge gave the nf4 prose 2.20 out of 5 overall. Higher scores mean better prose. Its lowest named dimensions were redundancy, 2.00; language, 2.20; characterization, 2.24; and pacing, 2.28. The patterns below explain how the text earns those low scores, although this reading study does not experimentally isolate causes of the scores.

## What was read and counted

The main sample contains 25 selected pieces, read in full. The ASTRA grading pipeline—the process that selects outputs and has a language model judge them—selected one piece per scenario. Its prose-quality measure is called Q2; the dimension scores are stored in `scores.Q2.dimensions` in `runs/custom50-e2b-it-2026-09-14/astra-graded/<ID>/scorecard.json`. Here `<ID>` is the scenario identifier, and the selected text is `artifacts[0].text`.

The test families used in this report are F1, scene writing in the chat reply; F2, revision or drafting delivered to a file; F3, alternative directions; F4, building or updating a Markdown knowledge base; and F5, using project facts, either supplied in the brief or retrieved from files. A knowledge base, abbreviated KB, is the project's reference folder. In an identifier such as F1-01, the prefix names the family and the final number names the case.

The selected sample consists of all 10 F1 cases, six F2 cases (F2-04, F2-06, F2-07, F2-08, F2-09 and F2-10), and nine F5 cases (F5-01, F5-02 and F5-04 through F5-10). F3 asks for “three numbered … next-scene directions”; F4 asks for a Markdown continuity wiki. Neither produces the kind of scene prose studied here, and F3 has no Q2 grade. They serve only as controls. Claims about prose therefore concern three families, not all five.

The comparison material contains the same 25 cases in bf16, stored in `bf16_prose.json`, plus 50 outputs from the distribution runs `distribution-e2b-it-2026-09-21-nf4` and `distribution-e2b-it-2026-09-21-bf16`. These repeat F1-01 25 times per format to expose repetition and variation. There are 100 available pieces in total: 25 read line by line and approximately 35 read closely. The source does not clarify whether those reading counts overlap.

Unless another denominator is stated, a fraction below counts pieces out of the 25 graded nf4 texts. Pattern-matching counts were run with `PYTHONPATH=src uv run python3`; the source lists `corpus.json`, `bf16_prose.json` and `graded_prose/` as reproduction material. An explicit brief specifies choices such as length, viewpoint and format. A loose brief leaves more of those choices to the model.

## Openings repeat an atmosphere before anyone acts

Twenty-one of 25 pieces, or 84%, begin with a version of “The air/silence/static/light/heat/bell/chill…” followed by two or three sensory descriptions, often using “thick” or “heavy.” This makes unrelated scenes sound alike and delays the action that would distinguish them.

- F1-01: “The damp air of the archive clung to Mara, heavy with the scent of brine and decaying paper.”
- F1-03: “The salt-laced air of the night ferry hung thick, heavy with the scent of brine and something older—something tied to the stone.”
- F1-09: “The air in the clockmaker's workshop was thick—a cloying mixture of fine brass polish, aged oil, and the dry, metallic scent of centuries of stopped time.”
- F1-10: “The air in the restaurant was thick, heavy with the ghosts of last night's patrons—a cloying mix of fryer oil, stale linen, and something vaguely sweet, like old sugar.”
- F2-04: “The static was the only constant, a thin, high shiver against the silence of the cabin.”
- F5-01: “The air in the archive was thick, heavy with the scent of brine and decaying paper.”

The clockmaker's opening starts with identifiable smells, then asks the reader to smell “centuries of stopped time.” The restaurant opening similarly moves from fryer oil and linen to “ghosts.” Those additions announce significance without giving the reader another observable event.

The repetition persists when one scenario is generated repeatedly. Among the 25 bf16 repetitions of F1-01, 12 begin “The damp air of the archive,” 12 contain “clung to Mara,” and 12 contain “decaying paper.” These counts are separate and should not be added. Near-identical openings include “The damp air of the archive clung to Mara's coat, heavy with the smell of brine and decaying paper…”. The nf4 repetitions vary more, but 11/25 still use the same opening construction and 11/25 contain “brine and”.

This is a model default: the explicit briefs request “restrained, concrete prose” or “spare prose,” not an atmospheric introduction. The competing images help explain the language score of 2.20. The source's proposed edit is to start with an action carrying the conflict, for example: “Mara set the brass ledger's empty tray down on Ilan's workbench and said, 'You sold it.'” That is an editorial illustration, not another observed model output or an established story fact.

## Sentences keep adding explanations after they have made their point

All 25 pieces contain a comma followed by an “-ing” phrase that elaborates on the main statement: 122 occurrences, roughly five per piece in a task often requesting 120–220 words. Other recurring forms rename a noun after a comma or dash. An appositive is such a renaming phrase; a participial phrase uses a verb form such as “tasting” or “turning” to add description. The weakness is the repeated elaboration, not the grammatical forms themselves.

The counts show how widely these sentence endings recur. The categories can overlap.

| Form | Pieces containing it | Occurrences |
|---|---:|---:|
| Trailing comma and “-ing” phrase | 25/25 | 122 |
| Comma followed by “his/her/its/their…” renaming or description | 23/25 | 74 |
| Any em dash | 23/25 | 75 |
| Em dash followed by “a/an/the…” renaming phrase | 20/25 | 44 |

Examples show what gets added:

- F1-01: “The air between them thickened, tasting of salt and unsaid futures.” The ending replaces a simple suggestion of tension with an image the reader cannot locate in the scene.
- F1-08: “The atmosphere thickened, turning the sterile archive into a stage for inevitable, charmingly clumsy misunderstandings.” The ending explains the intended comic effect before events produce it.
- F1-06: “…something faintly metallic—the ghost scent of old film.” The second description decorates the first rather than clarifying it.
- F5-01: “Mara felt the familiar knot tighten in her chest—the suspicion that lingered like the moisture in the air.” Suspicion is described twice, first as a knot and then as moisture.
- F5-10: “He smiled—a wide, genuine smile that didn't quite reach his eyes.” The explanation both calls the smile genuine and qualifies that impression with a familiar cliché.

The source connects this default to language 2.20 and redundancy 2.00: the sentence adds a second image or repeats an emotion instead of advancing the scene. The F1-03 judge lists “shields, knots, ghosts, snapping threads, shattering, drowning” as competing images. The proposed edit is to let one image stand. “The air between them thickened.” can end there; the next action can carry the tension.

## Feelings are named instead of made particular to a character

“Felt” occurs in 21/25 pieces, with 47 uses. “Seemed” occurs in 13/25, and “knew” or “knowledge” in 12/25. The repeated problem is that the narration identifies an emotion or explains knowledge without showing a decision, gesture or exchange specific to that character.

- F1-01: “Mara felt the familiar, hollow accusation rise, unproven and unverified.” “Unproven” and “unverified” repeat the same qualification; neither shows what Mara does with her suspicion.
- F1-07: “She watched the sky… and felt the familiar knot of worry tighten in her chest.” The stock bodily image could belong to almost anyone.
- F1-09: “Pia felt the familiar, sharp prickle of suspicion, a feeling she couldn't quite name, only identify.” The sentence names suspicion and then says she cannot name it.
- F5-08: “The knowledge was heavy, a lead weight in Kavi's chest.” The weight metaphor announces distress without revealing a distinctive response.
- F1-08: “She knew Kavi was already spiraling into the trap—the perfect defense mechanism for an overwhelmed clerk.” The explanation supplies the psychological reading instead of letting conduct reveal it.

These defaults help explain characterization 2.24. The F5-08 judge says “narration explains feelings rather than revealing individual motives”; the F1-08 and F5-08 judgements say dialogue “largely delivers premise information.” The source suggests replacing a feeling-noun with a character-specific action, illustrating the change with “Mara turned the ledger tray face-down so Ilan could not see it was empty.” Again, this is a proposed edit, not corpus evidence.

## The same stock phrases travel between unrelated stories

“Felt the familiar knot tighten” appears verbatim in 4/25 pieces; “familiar knot” appears in 6/25. “Heavy with the scent of” and “her gaze fixed on the” each appear in 4/25. Reusing the same body part, metaphor and verb gives different characters the same emotional language.

F1-03 says “Oren felt the familiar knot tighten.” F1-04, F5-01 and F5-04 use “felt the familiar knot tighten in her chest,” with F5-01 adding the moisture comparison quoted above. F5-09, including its bf16 version, says “Pia felt a familiar knot tighten in her chest.” The phrase appears across a literary-fiction archive, a science-fiction radio station and a fantasy courthouse.

Shared five-word sequences include “felt the familiar knot tighten” in four documents, “familiar knot tighten in her” in four, “her gaze fixed on the” in four, “heavy with the scent of” in four, and “the scent of brine and” in three. The source calls this a “memorised emotional idiom.” The observable evidence is repetition across stories and formats; it does not establish that a particular training passage was memorized. The repetition contributes to language 2.20 and redundancy 2.00 by presenting a familiar expression as if it were a fresh observation. The proposed remedy is to remove the stock phrase and give apprehensive characters different, specific responses.

## Mood repeats while the situation stays still

The source reports 16.8 abstract nouns per 1,000 words, though it does not define the noun list used. “Thick” or “heavy” appears in 22/25 pieces, with 42 uses; “tension,” “pressure” or “dread” in 17/25; and “the silence” in 20/25, with 29 uses.

F5-08 opens with four consecutive assertions of dread:

> “The silence of the apartment was the worst part… the heavy, thick quiet that seemed to absorb every other noise… The knowledge was heavy, a lead weight in Kavi's chest… the dread was starting to seep past the logic… The implication hung heavy…”

Silence, quiet, heavy knowledge, dread and a heavy implication tell us the same thing. They do not introduce a new fact, obstacle or response. The F5-08 judge describes “Recurrent heavy silence, cold, waiting, and dread” that “often restate the same emotional condition.” The F1-07 judge similarly says “repeated assertions of mystery substitute for developing tension.”

The briefs sometimes supply abstract framing themselves, such as “Their everyday work carries the weight of that unspoken choice.” The model amplifies that framing instead of turning it into an event. The source identifies this as the largest contributor to redundancy 2.00 and proposes allowing at most one direct statement of mood per scene, after an action has earned it; any further mood sentence should become a new fact or obstacle. This is an editorial recommendation, not a tested intervention.

## Endings tell the reader that the uncertainty remains

Twenty-two of 25 pieces finish on a named abstraction or suspended uncertainty; seven literally end with “waiting for…”. Explicit briefs do request an ending before the central uncertainty is resolved. The model's additional choice is to explain that non-resolution instead of stopping on an object, action or line of dialogue.

- F1-02 leaves the question “hanging unresolved in the humid air”.
- F1-03 ends “waiting for the moment when the uncertainty would either shatter the calm or drown them both”.
- F5-01 says the tide is “promising only more saturation, never resolution”.
- F5-04 leaves “the real story suspended, waiting for the thaw that might never come”.
- F5-05 says the question of true culpability “remained suspended, unresolved”.
- F5-08 ends “They were always waiting for someone to speak the truth.”

The final moment becomes commentary on the scene's theme. That deprives the ending of a concrete change or image and helps explain pacing 2.28 and language 2.20. The source counts only three exceptions: F2-08, “The stranger was close.”; F1-06, “The cinema wasn't just closing down; it was waking up.”; and F1-09, which ends in dialogue. It classifies these as concrete images or actions, although the cinema sentence is also criticized below for abstraction. That classification tension remains unresolved.

The proposed edit is to cut the sentence explaining what remains unresolved and stop at the preceding object or gesture.

## Balanced phrases add weight without adding information

“Not just” appears in 10/25 pieces, with 17 uses; “, but” in 20/25, with 35 uses; and “wasn't just … it was” in 7/25, with eight uses. The source calls this antithesis: a paired contrast between two descriptions.

Examples are “The cinema wasn't just closing down; it was waking up” (F1-06); “The heat in 1936 was not just a temperature; it was a physical weight…” (F1-07); “It wasn't just dust; it was the palpable memory of scarcity” (F2-07); and “The reunion wasn't just a memory; it was the seed of a new hope…” (F5-07). F5-01 has cold seeping “not just through the fabric of her tunic, but into the bone…”.

These are defaults. The paired construction promises a revelation, but the second half often adds an abstraction instead of an event or fact. That makes the prose sound portentous and contributes to language 2.20 and redundancy. The source suggests choosing one half, for example “The heat was a physical weight,” instead of building a two-part contrast around it.

## Scenes summarize situations instead of staging exchanges

Words associated with repeated or habitual events—“every,” “always,” “each,” “often” and “would”—occur in 22/25 pieces, with 42 uses. Dialogue accounts for only 137 of 597 sentences, or 23%. These are indicators of the observed tendency, not proof that each occurrence is a defect.

F1-10 jumps to “For the next week, Ada and Milo lived inside the walls of the family restaurant…” and then says “Every clatter of dishes sounded amplified, every distant siren a potential alarm.” This summarizes a week rather than showing one encounter. F2-07 gives “Every grain of dust that stirred seemed to carry the weight of the drought…”. F5-07 describes a communal garden “once a vibrant tapestry of green” becoming “a landscape of resignation.” Those descriptions supply conditions without showing an interaction that changes them.

Even briefs requesting dialogue get mostly narration. F1-03, F1-04 and F5-04 ask for “Let most of the scene unfold through dialogue” or “dialogue-led prose”; their dialogue shares are 24%, 26% and 37%. F1-05 has 11% dialogue despite asking “Show an interrupted practical task rather than summarizing.” These failures help explain pacing 2.28 and characterization 2.24: the F5-08 judge says dialogue delivers premise information, and the F5-09 and F5-10 judges say scenes stall. The suggested edit is to convert a narrated passage into an exchange that changes what someone knows.

## Narration explains motives that the scene could reveal

“Knew” or “knowledge” occurs in 12/25 pieces. Representative examples are F1-06, “He was supposed to be methodical, practical—a skeptic facing the inexplicable”; F1-08's explanation of Kavi's “perfect defense mechanism”; and F5-10, “He knew the script. He knew the mistake was inevitable, and the ensuing chaos was the main event.”

The model supplies the interpretation before conduct can establish it. The F5-10 judge says narration “repeatedly announces confusion and Milo's success instead of making them evident through action.” This default contributes to characterization 2.24 because it replaces an individual inner response with an explanation of the role a person plays. The source suggests deleting the explanatory ending after a dash. Its example keeps “He smiled—a wide, genuine smile…” and drops “that didn't quite reach his eyes,” identifying the latter as a cliché.

## Loose briefs invite copied notes and script formatting

Four of 25 pieces reuse at least six consecutive words from the brief; three contain commentary resembling stage directions; four use script formatting, all from loose briefs. These are more sensitive to instructions than the sentence habits above: explicit output rules suppress script format, while an open request to “make a scene” leaves the model to choose it.

F5-06 copies “must be checked before anything is screened” and “is scheduled to close on Friday” into statements about the projector and cinema. F5-08 repeats “their own voices answering from empty rooms.” The source also cites F1-08, “authorize a corrected official copy,” and F1-01, “above the tide line,” as verbatim reuse. The latter quoted fragments contain fewer than six words; the report does not show the longer matches behind its six-word count. It calls F1-08 explicit here, although another report classifies it as loose.

Loose F2-06 opens with “Setting:” and “Characters:” and gives “MR. HENDERSON (Clearing his throat loudly)…”. Loose F2-10 uses “Setting: The back kitchen…”, “(Scene Start)” and “(Scene End)”. Loose F2-08 starts “Setting: A sparsely lit, dusty archive room…”. F5-07 and F2-06 end with “(Scene End)” or “The scene ends on the tension between formality and absurdity.)”. In F5-09 the project-document frame enters the fiction itself: “The collector's square key, mentioned in the notes, felt heavy in the context of the delicate escapement.” “Mentioned in the notes” addresses the writer's source material rather than the character's experience.

Explicit instructions such as “Return only the prose, without a title or commentary” and “Put only the replacement prose between `<prose>`” contain this default: none of the four script-format cases is explicit. The proposed edit is to treat notes as facts to obey, not language to paste, and remove comments such as “mentioned in the notes” and “the scene ends.”

## The extra length largely repeats the same material

Nine of the 10 explicit pieces in this selected sample exceed the requested 120–220 words. Their median is 268 words. The only piece inside the range, F1-01, is reported here as 218 words. The failure report counts it as 216; these measures have not been reconciled.

The loose pieces are longer even without a formal ceiling. The comparison shows that removing the limit gives the repeated descriptions more room.

| Brief type | Selected pieces | Median words | Pieces over 220 words |
|---|---:|---:|---:|
| Explicit | 10 | 268 | 9/10 |
| Loose | 15 | 428 | 13/15 |

Every explicit F1/F2/F5 brief requests 120–220 words, including sentence-expansion tasks. Exceeding that ceiling is instruction non-compliance. The source attributes most of the excess to trailing explanations, named feelings and repeated mood, and proposes cutting an atmospheric opener, an explanatory afterthought and a mood sentence to reach the ceiling. That attribution is a reading judgement; no ablation measures how much each cut would remove.

## More instructions improve format more than craft

The same sentence habits occur in both brief types. This table shows why the source concludes that explicit instructions improve content discipline without making the writing more concrete.

| Habit | Explicit, 10 pieces | Loose, 15 pieces |
|---|---:|---:|
| Atmospheric opening | 8/10 | 13/15 |
| “Thick” or “heavy” | 9/10 | 13/15 |
| “Felt” | 9/10 | 12/15 |
| “The silence” | 8/10 | 12/15 |
| “, but” contrast | 7/10 | 13/15 |
| Trailing comma and “-ing” phrase | 10/10 | 15/15 |
| Comparison using “like” or “as if” | 4/10 | 10/15 |
| “Tension,” “pressure” or “dread” | 6/10 | 11/15 |
| “Familiar knot” | 4/10 | 2/15 |

Explicit instructions suppress screenplay formatting and title commentary, reduce verbatim note copying in the most constrained pieces, and supply a word ceiling that is mostly exceeded. F1-01 and F1-08 are cited as remaining copying examples, subject to the F1-08 classification conflict above. Both weight formats still tend to open with atmosphere, repeat emotions and close with abstractions. The source's proposed local edits address those mechanisms: remove redundant descriptions and turn narrated information into action or dialogue. Their effectiveness has not been tested here.

## Limits and findings the evidence does not support

Resolving the conflict too quickly is not the dominant problem. Twenty-two of 25 endings suspend uncertainty or introduce a new threat. The one clear early resolution is the F5-07 romance, which consummates the relationship: “The touch was everything—a bridge between the past longing and the uncertain future… the seed of a new hope, planted right there in the cracked soil.”

Hedges and adverbs occur, but the source ranks them below repeated elaboration, named feelings and repeated mood. It counts 166 “-ly” adverbs in 9,259 words, or 17.9 per 1,000 words; “seemed” occurs in 13/25 pieces. Its phrase “orders of magnitude less defining” expresses that ranking, not a demonstrated orders-of-magnitude numerical difference. Thin dialogue is supported by the 23% sentence share and the failure to increase it under dialogue-led instructions.

The 25 graded pieces span approximately eight genres, too few to support claims about individual genres. The source reports checking the habits across F1, F2 and F5 and both formats and finding them consistent. That supports persistence in this sample, not a population estimate or a causal explanation of the model's training.

## Evidence files

The original analysis is [craft.md](../../transcript-analysis/craft.md). Its reproduction list names `corpus.json`, containing all 150 attempts with outputs, briefs, workspace files and metadata; `graded_prose/<ID>.md`, containing the 25 selected pieces; `bf16_prose.json`, containing their 25 counterparts; `graded_dims.json`, containing Q2 rationales and dimension reasons; and `all_texts.md`, containing all 150 attempt texts. These names refer to the original `work/transcript-analysis/` directory, not this rewrite directory. The claimed `graded_prose/` directory was absent when this rewrite was prepared; the audit records that reproduction gap.
