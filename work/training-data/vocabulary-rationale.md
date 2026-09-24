# Vocabulary Extension — Per-Item Rationale

Proposed addition to `data/training/variation-catalog-v1.json`. Existing items are
copied through unchanged by `build_extension.py`; every item below is **NEW** and
appears in `vocabulary-extension.json` after the existing items of its list.

Merged counts: genres 60→80 (+20), styles 20→36 (+16), tropes 40→80 (+40),
situations 32→80 (+48), continuity_challenges 12→30 (+18), composition_rules 7→13 (+6);
book_candidates unchanged (15).

Problem labels used below:

- **BREADTH** — opens new physical/social worlds/genres/registers, breaking the
  archive/attic/manor/lighthouse cluster and the literary-style skew.
- **CLARIFY** — clarification pressure; the brief plausibly leaves a real decision
  open, so a good collaborator must ask or declare a default instead of silently
  assuming (solves the 40/40 silent-default finding).
- **PROSE** — prose-weighted concrete scene someone can dramatise (the pool should
  lean prose-heavy).
- **CONTINUITY** — positive continuity pressure: reveal/withhold timing, learning,
  payoff, dormancy (not just "don't contradict").
- **LENGTH** — enforces explicit length discipline (solves the 9/10 overshoot).

---

## BREADTH — new genres (+20)

Sub-genres and non-Anglophone traditions. No blends here; blends stay in
`genre_blend_size`.

1. **police procedural** — adds a case-log/evidence-chain structure absent from the 60, giving a professional, investigative scaffold for prose scenes.
2. **medical drama** — opens hospitals and clinical decision-making; a high-stakes setting the pool currently lacks.
3. **campus novel** — adds the academic-institution world (seminar, department, dorm) distinct from generic literary realism.
4. **epistolary fiction** — supplies a document-mediated register so tasks can use letters, logs and memos rather than only scene narration.
5. **sports fiction** — brings stadiums, teams, training and spectators, a major social world entirely missing.
6. **military science fiction** — a concrete genre-competent register (orders, hardware, units) absent among the speculative genres.
7. **dystopian fiction** — separates the political/administrative dystopia tradition from post-apocalyptic survival.
8. **weird fiction** — adds the reality-instability register distinct from cosmic and gothic horror.
9. **picaresque** — an episodic, low-born, satirical tradition giving non-heroic narrative shape.
10. **war fiction** — grounds conflict in logistics, civilians and aftermath rather than epic fantasy or adventure.
11. **Nordic saga** — non-Anglophone tradition: terse, lineage- and feud-driven prose with a distinct register.
12. **Russian psychological realism** — non-Anglophone tradition: interior moral pressure and social observation.
13. **French naturalism** — non-Anglophone tradition: deterministic environment and class detail.
14. **Japanese I-novel (shishōsetsu)** — non-Anglophone tradition: confessional, quotidian, low-plot first person.
15. **Chinese wuxia** — non-Anglophone tradition: martial honour networks and obligation, not stock fantasy quest.
16. **Latin American boom novel** — non-Anglophone tradition: non-linear time and political family history.
17. **West African oral epic** — non-Anglophone tradition: griot voice, genealogy and communal performance.
18. **Italian neorealism** — non-Anglophone tradition: postwar ordinary lives and material scarcity.
19. **German Bildungsroman** — non-Anglophone tradition: formation over time, a structural shape not otherwise named.
20. **Japanese light novel** — genre-competent popular register: fast, voicey, serialized, dialogue-forward.

## BREADTH — new styles (+16)

Genre-competent and plain registers, to counter the literary skew of the existing 20.

1. **brisk and plot-forward** — gives explicit permission to prioritize event over atmosphere and stop scene-padding.
2. **workmanlike and transparent** — a deliberately competent, unshowy register for genre tasks.
3. **plain declarative** — the simplest register, useful for documentary and procedural prose.
4. **procedural and step-by-step** — matches police/medical/lab genres where order of operations carries meaning.
5. **crisp newsroom clarity** — a reporting register for civic, courtroom and market scenes.
6. **terse hard-boiled** — supplies the noir/crime voice the catalog has a genre for but no style for.
7. **warm and reassuring** — supports cosy and domestic registers without literary ornament.
8. **chatty first-person** — enables voicey, digressive narration distinct from "colloquial and conversational".
9. **briskly comic** — a faster comic timing than the existing "dryly comic".
10. **clinical and detached** — an unsentimental register for hospital, autopsy and institutional scenes.
11. **urgent present-tense** — a pacing tool for thrillers and emergency scenes.
12. **pulpy and propulsive** — explicitly sanctions genre-pleasure prose so it is not "corrected" toward literary style.
13. **reader-friendly and formula-aware** — allows competent use of genre conventions rather than subverting them.
14. **sensory but unfussy** — keeps physical detail without the existing "expansive and sensory" purple drift.
15. **precise scientific register** — a lab/technical voice complementing the existing "voice shaped by a particular profession".
16. **measured expository clarity** — the plain explanatory register needed for world-rule and backstory delivery.

## BREADTH — new tropes (+40)

Relational, professional, domestic, civic and comic pressure; deliberately avoids the
stock fantasy-quest set.

1. **a secret kept to protect someone else's reputation** — relational: loyalty vs honesty in a professional or family setting.
2. **a mentor's advice that misfires** — professional: guidance becomes liability, a proactive pressure.
3. **a favor that must be repaid at the worst time** — relational: obligation arriving at maximum cost.
4. **a shared joke that becomes a private language** — relational/comic: intimacy rendered through idiom rather than declaration.
5. **a rival who is right about the facts** — professional: rivalry without villainy, forcing intellectual concession.
6. **a parent living through a child's success** — domestic: status and love entangled.
7. **a sibling who kept the family business afloat** — domestic/professional: quiet resentment in a shared enterprise.
8. **a marriage negotiated like a contract** — domestic: intimate stakes framed by explicit terms.
9. **an office alliance that outlives its purpose** — professional/civic: loyalty stranded by changed goals.
10. **a manager promoted past their competence** — professional/comic: organisational failure made personal.
11. **a well-meaning lie told to spare a friend** — relational: kindness creating later cost.
12. **a whistleblower with mixed motives** — civic: moral action complicated by self-interest.
13. **a patient who knows more than the doctor** — medical/professional: expertise reversed, an open decision about disclosure.
14. **a customer who becomes a confidant** — market/relational: intimacy in commercial space.
15. **a committee that cannot decide** — civic/comic: collective paralysis with real consequences.
16. **a town's unofficial historian** — civic: memory, authority and contested local truth.
17. **a retirement nobody has planned for** — domestic: identity change and quiet stakes.
18. **an inheritance that splits siblings** — domestic: property and grief as relational pressure.
19. **a neighborhood watch with divided loyalties** — civic: community policing turned against itself.
20. **a colleague who takes credit by accident** — professional/comic: credit and resentment without a villain.
21. **a contractor who finds the previous worker's shortcut** — professional: workmanship and liability, a concrete discovery scene.
22. **a shift worker covering for an absent colleague** — labor/relational: small solidarity with real cost.
23. **a teacher's pet who resents the role** — classroom/relational: status and belonging.
24. **a coach who benches the star** — sports/professional: authority vs talent.
25. **a nurse who breaks a small rule for a patient** — medical: care vs procedure.
26. **a chef whose signature dish is not their own** — kitchen/professional: authorship and reputation.
27. **a landlord and tenant who need each other** — domestic/civic: unequal dependence.
28. **an estranged cousin who arrives with a claim** — domestic: family return with legal/emotional stakes.
29. **a wedding guest with an uninvited truth** — relational/comic: ceremony under threat from candour.
30. **a funeral that reveals a hidden debt** — domestic: grief interrupted by obligation.
31. **a charity gala with a hidden agenda** — civic: public beneficence as leverage.
32. **a local election decided by one vote** — civic: small democracy, legible stakes.
33. **a union rep caught between members and management** — labor/civic: representation under conflicting mandates.
34. **a neighbor who hears everything through the wall** — domestic/comic: proximity as surveillance.
35. **a best friend tired of being the sidekick** — relational: unequal friendship renegotiated.
36. **a couple who cannot agree on a name** — domestic/comic: small decision exposing larger fault lines.
37. **a family recipe treated as a trade secret** — domestic/professional: inheritance and commerce.
38. **a group project where one person does the work** — classroom/comic: fairness and credit.
39. **a public apology that makes things worse** — civic/comic: speech that cannot repair harm.
40. **a regular customer who notices a change** — market/relational: observation by an outsider as narrative pressure.

---

## CLARIFY — situations that leave a real decision open

Each of these embeds a material choice the brief plausibly does not settle, so guessing
silently is visibly costly.

1. **a price sign is wrong and the stallholder must choose which price to honor** — a quote/price default with reputational stakes if guessed wrong.
2. **two customers claim the last crate before the market opens** — allocation must be decided openly, not by the first narration.
3. **a courier must decide whether to open an undeliverable package** — privacy vs safety, a decision the prose must stage.
4. **a shop owner recognizes a returned item as stolen** — accuse, absorb the loss, or report: three legible options.
5. **a foreman must choose between a deadline and an unlogged safety check** — the default (ship anyway) is exactly the silent assumption to surface.
6. **a night shift finds the previous shift's log incomplete** — report or complete it quietly; consequences differ.
7. **a triage nurse must assign the last free bed** — who deserves care is a decision that cannot be narrated as inevitable.
8. **a doctor's order contradicts a patient's stated wish** — which instruction governs must be declared, not assumed.
9. **an ambulance arrives with no record of who called** — proceed or trace the caller first.
10. **a clinic runs out of a routine medicine during a rush** — rationing criteria are an authored decision.
11. **a patient's relatives disagree about how much to tell them** — disclosure policy is the open question.
12. **a jury is told to ignore testimony it has already heard** — whether the instruction holds is a live decision.
13. **a clerk finds a filing error that could void a decision** — report or correct silently.
14. **a council vote ties and the chair must break it** — the explicit choices make a silent default impossible.
15. **a permit is revoked while the work is half done** — stop, continue, or appeal.
16. **a pilot must choose between the schedule and a passenger's need** — duty allocation stated as a choice.
17. **a driver is offered a load with vague papers** — accept, refuse, or inspect.
18. **a restaurant critic arrives unannounced on the busiest night** — treat as ordinary or intervene; both change the scene.
19. **a server shows an unauthorized access nobody reported** — disclose or investigate quietly.
20. **a prison visit is denied for a reason not stated** — accept, demand reason, or escalate.
21. **a guard must decide whether to report small contraband** — rule vs relationship, a concrete decision.
22. **a teacher finds a note that names a student** — act now or hold it; timing and duty are open.
23. **a well runs dry while a caravan waits** — ration, detour, or negotiate.
24. **a hiker's only map is contradicted by a marked trail** — trust the document or the ground; the choice must be made explicit.

## PROSE — new concrete scenes / physical and social worlds

Each implies a specific place, task and sensory texture suitable for prose dramatisation,
breaking the archive/lighthouse cluster.

1. **the night market loses power during its busiest hour** — market world; darkness re-orders social space.
2. **a factory line halts because one worker refuses to sign off** — industrial world; a stoppage is a scene.
3. **a construction crew uncovers an unmarked pipe** — construction site; discovery halts labour.
4. **a mine elevator stalls with visitors still below** — subterranean industrial world, claustrophobic stakes.
5. **a warehouse inventory counts one pallet too many** — logistics world; a discrepancy as inciting object.
6. **a cleaner finds a discarded chart in the stairwell** — hospital back-of-house; institutional prose.
7. **a witness is asked a question nobody anticipated** — courtroom; a live verbal beat.
8. **a public defender meets a client minutes before the hearing** — legal world; compressed time and place.
9. **a ferry manifest lists one passenger too many** — ship world; a bureaucratic detail becomes suspense.
10. **a cargo log disagrees with the port records** — maritime trade; documents vs dockside reality.
11. **a train is held at a signal with no explanation** — transit world; enforced waiting as scene.
12. **a border crossing closes while a line of travelers waits** — civic/geographic threshold; crowd as prose texture.
13. **irrigation from one farm floods the neighbor's field** — agricultural land; water and boundary conflict.
14. **a harvest crew finds a crop that is not theirs** — farm world; labour and property.
15. **a kitchen runs out of an ingredient mid-service** — restaurant back line; time-pressure prose.
16. **canning-line labels are switched before shipping** — food factory; error propagating through a process.
17. **a sample is mislabeled on the last day of a study** — laboratory; evidence integrity at risk.
18. **a result cannot be reproduced before a deadline** — research world; epistemic suspense.
19. **a code deploy fails minutes before a demo** — software/code world; a modern, under-represented setting.
20. **a student's data does not match the lab notebook** — lab/classroom; mentorship and record-keeping.
21. **a stadium announcer is handed a last-minute substitution** — stadium world; public address as scene.
22. **a festival stage loses power mid-performance** — festival world; crowd, performance, failure.
23. **a substitute is handed a lesson plan for the wrong class** — classroom; improvised authority.
24. **a wildfire forces a choice about which animals to release** — desert/wildfire world; urgent physical stakes.

---

## CONTINUITY — positive-pressure challenges (+18)

Reveal/withhold timing, learning, payoff, and dormancy — beyond "don't contradict".

1. **delay a revelation until its physical evidence exists on the page** — forces the reveal to be earned by on-page action.
2. **let a character learn a fact only after acting on the wrong one** — dramatises the gap between belief and knowledge.
3. **plant a detail that must stay dormant until its later payoff** — turns a detail into an intentional obligation.
4. **pay off a consequence established earlier in the same scene** — local cause-and-effect discipline.
5. **reveal information to the reader before the POV character can act on it** — builds dramatic irony with a required later beat.
6. **withhold a name until every speaker has used it wrongly at least once** — makes naming an active continuity device.
7. **stage a discovery at the moment it changes the character's next choice** — binds revelation to decision.
8. **return an earlier promise as an obligation now due** — converts prior text into present pressure.
9. **let an earlier lie constrain what a character can say now** — continuity as positive constraint on dialogue.
10. **reveal the cost of a choice before its benefit** — forces forward-looking stakes rather than retrospective contradiction.
11. **keep a secondary character ignorant of a fact the reader already knows** — manages knowledge states deliberately.
12. **introduce a detail that must not matter until a later chapter** — intentional dormancy with a scheduled payoff.
13. **force a choice between revealing a secret and protecting a relationship** — makes disclosure a live decision.
14. **let an object's history surface only when it is handled again** — binds continuity to physical action.
15. **withhold the reason for an action until its consequence is visible** — delay motives to increase pressure.
16. **make a character notice a change they cannot yet interpret** — seeds a later reveal without pre-empting it.
17. **delay an explanation until the person who needs it has stopped asking** — timing of disclosure as craft pressure.
18. **let a consequence arrive from a decision made before the scene begins** — enforces off-page setup that must cash out.

---

## LENGTH — composition rule (+1)

1. **"When a length budget is specified, treat it as a hard ceiling and verify the draft against it before submission."** — directly targets the 9/10 overshoot finding by making budget compliance a checked requirement rather than a preference.

## CLARIFY — composition rule (+1)

1. **"When a brief leaves a material choice open, the task must ask the choice or state the assumed default explicitly; it must never resolve the ambiguity silently."** — encodes the silent-default fix (40/40) as a generation rule.

## BREADTH — composition rules (+3)

1. **"Across a batch, vary the opening construction, dominant image, and physical setting; repeated stems or clustered settings count as a coverage failure."** — targets the "The damp air of the archive" collapse and the setting cluster.
2. **"A sampled trope must press on a concrete decision in the scene rather than decorate the setting."** — stops tropes being used as mere flavour, which produced interchangeable prose.
3. **"In a genre blend, each contributing genre's conventions must remain load-bearing in the scene; a blend that only relabels a single genre is not admissible."** — addresses the noted limitation that genre blends lack grounding.

## CONTINUITY — composition rule (+1)

1. **"A sampled continuity challenge that creates positive pressure must name the detail, the moment of disclosure or payoff, and the consequence of mistiming it."** — makes the new reveal/withhold challenges operational rather than vague.

---

## Notes / caveats

- All existing items are preserved byte-for-byte and in original order; new items are
  appended after them.
- `book_candidates` is unchanged (15 entries).
- No item duplicates an existing item; `build_extension.py` fails on any intersection,
  and the merged file was validated for intra-list duplicates.
- Constraint respected: this is a proposal written only to the spawn work directory;
  nothing under `src/`, `tests/`, `data/`, or `runs/` was modified.
