# Why the task vocabulary is being expanded

This proposal explains why each new genre, prose style, trope, situation and continuity challenge was chosen for the task generator. It also explains six new composition rules intended to address repeated settings and wording, unstated creative assumptions, continuity mistakes and excessive length.

The proposed additions go into `data/training/variation-catalog-v1.json`. The [builder](../../training-data/build_extension.py) copies existing entries unchanged and appends the new entries in [vocabulary-extension.json](../../training-data/vocabulary-extension.json). Genres increase from 60 to 80 (+20), styles from 20 to 36 (+16), tropes from 40 to 80 (+40), situations from 32 to 80 (+48), continuity challenges from 12 to 30 (+18), and composition rules from 7 to 13 (+6). The 15 `book_candidates`, or candidate books, remain unchanged.

The original rationale calls the broadening goal `BREADTH`: add physical settings, social settings, story forms and prose registers beyond the archive, attic, manor and lighthouse cluster and the existing literary emphasis. It links clarification rules to a finding that 40/40 attempts under loose briefs, excluding tasks that explicitly request alternative directions, silently chose defaults. It links the length rule to 9/10 explicit prose pieces exceeding their budget in the craft-analysis sample. These are motivations for the proposal; the document does not measure whether the additions correct either problem.

## Twenty genres add story forms and traditions

Genres are kinds of story. These additions include subgenres and traditions outside English-language literature; they do not add blends. The `genre_blend_size` field controls how many genres are combined.

1. “police procedural” adds case logs and chains of evidence, giving investigative scenes a professional structure missing from the existing 60 genres.
2. “medical drama” adds hospitals and clinical decisions, a high-stakes setting missing from the pool.
3. “campus novel” adds seminars, departments and dormitories, making academic institutions available as more than generic literary realism.
4. “epistolary fiction” lets tasks tell stories through letters, logs and memos as well as narrated scenes.
5. “sports fiction” adds stadiums, teams, training and spectators, a major social world previously absent.
6. “military science fiction” adds the language of orders, hardware and military units, which the existing speculative genres lack.
7. “dystopian fiction” adds political and administrative oppression as a story tradition distinct from survival after an apocalypse.
8. “weird fiction” adds stories in which reality becomes unstable, distinct from cosmic and gothic horror.
9. “picaresque” adds an episodic, satirical story about a person of low social standing, giving the catalog a non-heroic narrative shape.
10. “war fiction” grounds conflict in logistics, civilian lives and aftermath, extending beyond epic fantasy and adventure.
11. “Nordic saga” adds terse prose driven by ancestry and feuds from a tradition outside English-language literature.
12. “Russian psychological realism” adds inward moral conflict and close social observation from a tradition outside English-language literature.
13. “French naturalism” adds a tradition outside English-language literature in which environment and class shape people's lives.
14. “Japanese I-novel (shishōsetsu)” adds a Japanese confessional tradition: first-person accounts of everyday life with little plot.
15. “Chinese wuxia” adds a Chinese martial-arts tradition driven by honour, networks of relationships and obligations, expanding beyond the stock fantasy quest.
16. “Latin American boom novel” adds a Latin American tradition that combines non-linear time with political family history.
17. “West African oral epic” adds the voice of a griot, an oral storyteller, together with genealogy and communal performance from a tradition outside English-language literature.
18. “Italian neorealism” adds a tradition outside English-language literature centred on ordinary postwar lives and material scarcity.
19. “German Bildungsroman” adds the German coming-of-age tradition, explicitly naming a story structured around personal development over time.
20. “Japanese light novel” adds a popular Japanese form with fast pacing, a strong narrative voice, serial publication and prominent dialogue.

## Sixteen styles make plain and popular prose available

Styles are prose registers. These additions counter the literary emphasis of the existing 20 by making plain prose and voices suited to particular genres available.

1. “brisk and plot-forward” asks for events to take priority over atmosphere, reducing prose that merely pads out a scene.
2. “workmanlike and transparent” allows competent, unobtrusive prose for genre fiction.
3. “plain declarative” provides simple statements suited to documentary and procedural writing.
4. “procedural and step-by-step” makes the order of operations explicit when it matters in police, medical or laboratory scenes.
5. “crisp newsroom clarity” adds a clear reporting voice for civic, courtroom and market scenes.
6. “terse hard-boiled” adds the terse voice associated with noir and crime, genres already present without a matching style.
7. “warm and reassuring” supports cosy and domestic writing without literary ornament.
8. “chatty first-person” allows a distinctive, digressive first-person narrator, extending beyond the existing “colloquial and conversational” style.
9. “briskly comic” asks for faster comic timing than the existing “dryly comic” style.
10. “clinical and detached” adds an unsentimental voice for hospitals, autopsies and institutions.
11. “urgent present-tense” uses present-tense narration to help pace thrillers and emergencies.
12. “pulpy and propulsive” allows energetic genre entertainment without treating literary prose as a correction it needs.
13. “reader-friendly and formula-aware” allows the writer to use genre conventions competently without always subverting them.
14. “sensory but unfussy” retains physical detail while countering the excessively ornate prose encouraged by “expansive and sensory”.
15. “precise scientific register” adds a laboratory or technical voice alongside the existing “voice shaped by a particular profession”.
16. “measured expository clarity” provides plain explanation for the rules of a fictional world and for backstory.

## Forty tropes create relationship, work and civic conflicts

Tropes are familiar story shapes. These additions cover relationships, professions, domestic life, civic life and comedy, deliberately avoiding the usual fantasy-quest set.

1. “a secret kept to protect someone else's reputation” puts loyalty in conflict with honesty in a professional or family relationship.
2. “a mentor's advice that misfires” turns professional guidance into a liability, giving characters a problem they must act on.
3. “a favor that must be repaid at the worst time” makes a relational obligation fall due when repaying it costs most.
4. “a shared joke that becomes a private language” shows intimacy through a shared idiom, allowing relational comedy without announcing the closeness.
5. “a rival who is right about the facts” forces a professional rival to concede a fact without making either rival a villain.
6. “a parent living through a child's success” entangles parental love with the status a child's success brings.
7. “a sibling who kept the family business afloat” gives a shared family enterprise a source of quiet resentment.
8. “a marriage negotiated like a contract” makes the terms of an intimate domestic relationship explicit.
9. “an office alliance that outlives its purpose” leaves professional or civic loyalty in place after the goals that created it have changed.
10. “a manager promoted past their competence” makes organisational failure personal and potentially comic.
11. “a well-meaning lie told to spare a friend” lets an act of kindness between friends create a later cost.
12. “a whistleblower with mixed motives” complicates a morally significant civic act with self-interest.
13. “a patient who knows more than the doctor” reverses medical expertise and leaves a meaningful decision about disclosure.
14. “a customer who becomes a confidant” creates intimacy within a commercial relationship.
15. “a committee that cannot decide” makes collective indecision carry real civic consequences and comic possibilities.
16. “a town's unofficial historian” puts local memory, authority and contested truth into the same civic role.
17. “a retirement nobody has planned for” creates a domestic change of identity with quiet stakes.
18. “an inheritance that splits siblings” uses property and grief to put pressure on sibling relationships.
19. “a neighborhood watch with divided loyalties” turns community policing against itself through conflicting loyalties.
20. “a colleague who takes credit by accident” creates professional resentment and comedy over credit without requiring a villain.
21. “a contractor who finds the previous worker's shortcut” turns workmanship and liability into a concrete professional discovery scene.
22. “a shift worker covering for an absent colleague” shows solidarity between workers through a small act with a real cost.
23. “a teacher's pet who resents the role” puts classroom status in conflict with belonging.
24. “a coach who benches the star” pits a coach's authority against an athlete's talent.
25. “a nurse who breaks a small rule for a patient” puts medical care in conflict with procedure.
26. “a chef whose signature dish is not their own” links culinary authorship to professional reputation.
27. “a landlord and tenant who need each other” creates a domestic and civic relationship in which dependence is mutual but unequal.
28. “an estranged cousin who arrives with a claim” brings a relative back into a domestic setting with both legal and emotional stakes.
29. “a wedding guest with an uninvited truth” puts a social ceremony at risk through candour, creating relational and comic possibilities.
30. “a funeral that reveals a hidden debt” interrupts family grief with an obligation.
31. “a charity gala with a hidden agenda” lets public charity become a means of exerting civic influence.
32. “a local election decided by one vote” makes the stakes of local democracy easy to see.
33. “a union rep caught between members and management” puts a labour representative under conflicting obligations to members and management.
34. “a neighbor who hears everything through the wall” turns domestic proximity into surveillance, with comic possibilities.
35. “a best friend tired of being the sidekick” requires friends to renegotiate an unequal relationship.
36. “a couple who cannot agree on a name” uses a small domestic disagreement to expose larger divisions, with comic possibilities.
37. “a family recipe treated as a trade secret” connects family inheritance with professional commerce.
38. “a group project where one person does the work” puts fairness and credit at issue in a classroom situation that can become comic.
39. “a public apology that makes things worse” shows a civic attempt at repair through speech making the harm worse, with comic possibilities.
40. “a regular customer who notices a change” lets an outsider's observation change a commercial relationship and drive the story.

## Twenty-four situations leave consequential choices open

Situations are concrete predicaments to dramatise. These 24 leave a material choice unresolved, so an unstated assumption can change the story. The catalog labels this purpose `CLARIFY`: the collaborator should ask or state a default. The predicament alone does not specify which response is appropriate.

1. “a price sign is wrong and the stallholder must choose which price to honor” leaves the price to honour unresolved; silently choosing it can change the stallholder's reputation.
2. “two customers claim the last crate before the market opens” requires an explicit allocation decision before narration gives one customer the crate.
3. “a courier must decide whether to open an undeliverable package” puts privacy and safety in conflict through a decision the prose must dramatise.
4. “a shop owner recognizes a returned item as stolen” offers three materially different actions: accuse the customer, absorb the loss, or report the theft.
5. “a foreman must choose between a deadline and an unlogged safety check” exposes “ship anyway” as a default that the collaborator should state rather than silently adopt.
6. “a night shift finds the previous shift's log incomplete” leaves a choice between reporting the gap and quietly completing the log, with different consequences.
7. “a triage nurse must assign the last free bed” requires an authored decision about who receives care; the narration cannot treat that decision as inevitable.
8. “a doctor's order contradicts a patient's stated wish” requires the collaborator to declare which instruction governs rather than assume it.
9. “an ambulance arrives with no record of who called” leaves a choice between proceeding and first tracing the caller.
10. “a clinic runs out of a routine medicine during a rush” requires the writer to choose rationing criteria.
11. “a patient's relatives disagree about how much to tell them” leaves disclosure policy unresolved.
12. “a jury is told to ignore testimony it has already heard” leaves open whether the instruction actually governs the jury's decision.
13. “a clerk finds a filing error that could void a decision” offers a choice between reporting the error and correcting it silently.
14. “a council vote ties and the chair must break it” sets out the chair's choices explicitly; the original rationale claims this makes a silent default impossible.
15. “a permit is revoked while the work is half done” offers three actions: stop, continue, or appeal.
16. “a pilot must choose between the schedule and a passenger's need” makes competing duties an explicit choice.
17. “a driver is offered a load with vague papers” offers three actions: accept, refuse, or inspect.
18. “a restaurant critic arrives unannounced on the busiest night” leaves a choice between ordinary treatment and intervention, either of which changes the scene.
19. “a server shows an unauthorized access nobody reported” leaves a choice between disclosure and quiet investigation.
20. “a prison visit is denied for a reason not stated” offers three actions: accept the denial, demand a reason, or escalate.
21. “a guard must decide whether to report small contraband” puts a concrete rule in conflict with a relationship.
22. “a teacher finds a note that names a student” leaves the timing of action and the teacher's duty open.
23. “a well runs dry while a caravan waits” offers three actions: ration, detour, or negotiate.
24. “a hiker's only map is contradicted by a marked trail” requires an explicit choice between trusting the document and trusting what is on the ground.

## Twenty-four situations supply concrete scenes

These 24 situations provide a place, a task and sensory detail that can be turned into prose. The catalog labels this purpose `PROSE`, meaning that the pool should favour scene writing. They extend beyond the concentration of archive and lighthouse settings.

1. “the night market loses power during its busiest hour” uses darkness to change how people occupy and navigate a market.
2. “a factory line halts because one worker refuses to sign off” makes an industrial stoppage something the writer can dramatise.
3. “a construction crew uncovers an unmarked pipe” interrupts labour with a discovery at a construction site.
4. “a mine elevator stalls with visitors still below” puts visitors underground with claustrophobic industrial stakes.
5. “a warehouse inventory counts one pallet too many” starts a logistics story with a concrete discrepancy.
6. “a cleaner finds a discarded chart in the stairwell” places institutional action in a hospital's service areas.
7. “a witness is asked a question nobody anticipated” creates a live courtroom exchange around an unexpected question.
8. “a public defender meets a client minutes before the hearing” compresses legal action into a specific place and a short time.
9. “a ferry manifest lists one passenger too many” turns a shipping document into a source of suspense.
10. “a cargo log disagrees with the port records” puts maritime trade records in conflict with dockside reality.
11. “a train is held at a signal with no explanation” turns enforced waiting on public transport into a scene.
12. “a border crossing closes while a line of travelers waits” uses a waiting crowd to give a civic and geographical boundary physical presence.
13. “irrigation from one farm floods the neighbor's field” turns agricultural water use into a boundary conflict.
14. “a harvest crew finds a crop that is not theirs” connects farm labour to a property dispute.
15. “a kitchen runs out of an ingredient mid-service” puts restaurant work under immediate time pressure.
16. “canning-line labels are switched before shipping” shows an error spreading through a food-production process.
17. “a sample is mislabeled on the last day of a study” puts the integrity of laboratory evidence at risk.
18. “a result cannot be reproduced before a deadline” builds research suspense around whether a result can be confirmed.
19. “a code deploy fails minutes before a demo” adds a modern software workplace that the pool under-represents.
20. “a student's data does not match the lab notebook” connects laboratory or classroom mentoring with record-keeping.
21. “a stadium announcer is handed a last-minute substitution” uses a public announcement to dramatise an event in a stadium.
22. “a festival stage loses power mid-performance” brings a crowd, a performance and a failure together at a festival.
23. “a substitute is handed a lesson plan for the wrong class” requires a teacher to improvise authority in a classroom.
24. “a wildfire forces a choice about which animals to release” adds urgent physical stakes in a desert or wildfire setting.

## Eighteen continuity challenges control what becomes known and when

Continuity challenges are consistency problems for the writer. Here `CONTINUITY` means managing disclosure, learning and later consequences: a detail may need to remain inactive until its payoff, rather than merely avoid contradiction.

1. “delay a revelation until its physical evidence exists on the page” requires action on the page to establish the evidence before the revelation.
2. “let a character learn a fact only after acting on the wrong one” shows a character acting on a mistaken belief before learning the truth.
3. “plant a detail that must stay dormant until its later payoff” makes a planted detail a promise of a later payoff while keeping it inactive for now.
4. “pay off a consequence established earlier in the same scene” requires cause and effect within the scene.
5. “reveal information to the reader before the POV character can act on it” creates dramatic irony by giving the reader information before the point-of-view character can use it, with a later action required.
6. “withhold a name until every speaker has used it wrongly at least once” uses mistaken names to control continuity across speakers.
7. “stage a discovery at the moment it changes the character's next choice” makes a discovery change the character's next decision.
8. “return an earlier promise as an obligation now due” turns an earlier promise into a present obligation.
9. “let an earlier lie constrain what a character can say now” limits what a character can say because an earlier lie must still be maintained.
10. “reveal the cost of a choice before its benefit” puts the cost of a decision ahead of its benefit so the stakes face forward.
11. “keep a secondary character ignorant of a fact the reader already knows” requires the writer to track what a secondary character does not know even when the reader knows it.
12. “introduce a detail that must not matter until a later chapter” keeps a detail inactive until its scheduled payoff in a later chapter.
13. “force a choice between revealing a secret and protecting a relationship” makes disclosure a choice with consequences for a relationship.
14. “let an object's history surface only when it is handled again” connects information about an object to the physical act of handling it again.
15. “withhold the reason for an action until its consequence is visible” delays the motive until its visible consequence has increased the pressure.
16. “make a character notice a change they cannot yet interpret” prepares a later revelation without explaining it prematurely.
17. “delay an explanation until the person who needs it has stopped asking” uses the timing of an explanation to create pressure after the request for it has ended.
18. “let a consequence arrive from a decision made before the scene begins” requires an earlier, off-page decision to have a consequence in the present scene.

## Six composition rules govern how ingredients become tasks

Composition rules tell the generator how to use the vocabulary. The exact proposed rules are retained below because they are part of the proposal's technical content.

1. “When a length budget is specified, treat it as a hard ceiling and verify the draft against it before submission.” This is the `LENGTH` rule. It targets the 9/10 overshoot finding by making word-budget compliance a checked requirement rather than a preference.
2. “When a brief leaves a material choice open, the task must ask the choice or state the assumed default explicitly; it must never resolve the ambiguity silently.” This `CLARIFY` rule targets the 40/40 silent-default finding. Its wording leaves unclear whether the generated task or the answering agent must ask; that ambiguity remains in the proposal.
3. “Across a batch, vary the opening construction, dominant image, and physical setting; repeated stems or clustered settings count as a coverage failure.” This `BREADTH` rule targets repeated openings such as “The damp air of the archive” and the concentration of settings.
4. “A sampled trope must press on a concrete decision in the scene rather than decorate the setting.” This `BREADTH` rule makes the trope affect a character's choice. The rationale attributes interchangeable prose to treating tropes as mere flavour.
5. “In a genre blend, each contributing genre's conventions must remain load-bearing in the scene; a blend that only relabels a single genre is not admissible.” Here “load-bearing” means that each genre must affect how the scene works. This `BREADTH` rule responds to the stated lack of grounding in genre blends.
6. “A sampled continuity challenge that creates positive pressure must name the detail, the moment of disclosure or payoff, and the consequence of mistiming it.” Here “positive pressure” means an event the writer must arrange, such as a revelation or payoff. This `CONTINUITY` rule turns the new challenges into specific instructions about what happens and when.

## Preservation and limits of this proposal

All existing items are preserved byte-for-byte and in their original order; additions follow them. No addition duplicates an existing item: `build_extension.py` fails on overlap, and the merged file was checked for duplicates within each list. The proposal was written only to the original agent's work directory; it made no changes under `src/`, `tests/`, `data/`, or `runs/`.
