# Why these 24 story settings were added

This proposal explains how 24 reusable story settings broaden the task pool beyond its existing ten worlds. It describes the voice and continuity problem each setting adds, so the owner can judge whether the collection offers meaningfully different writing tasks.

A world is a reusable setting with two characters and facts the writing must respect. These additions respond to repeated wording in the run using `bf16`, the 16-bit brain-floating-point model-weight format: 12/25 pieces opened with the same “damp air of the archive” stem, and self-BLEU was 0.71. Self-BLEU measures how much wording in each generated piece resembles the others; a higher score means less variety. These observations motivate the proposal but do not show that new worlds improve generated prose.

The [proposed worlds](../../training-data/worlds-extension.json) contain two settings in each of 12 families. All 24 `prose_style` values are distinct, satisfying the limit that no more than two worlds may share a style. None uses an archive, attic, manor or lighthouse, and none reuses a character, object or setting from `worlds.json`.

Each world supplies five `required` facts to preserve; a `belief` that is not established truth; an `incidental` detail that may change; and a `forbidden` assertion the writing must not make. The `update` and `new_state` fields describe a change and its resulting state. A `protected` sentence appears unchanged inside `revision`, the text supplied for a revision task. “Canon” below means established story facts.

The task labels come from `src/writing_agent/development.py`: F1 asks for a scene in the reply; F2 asks for a local revision or for exploration followed by writing; F3 asks for alternatives; F4 asks for knowledge-base (KB) construction or updates; and F5 asks the agent to retrieve information from a supplied KB. Every world below supports F1, F2 and F3. Within each pair, the first also supports F4 and the second also supports F5, preserving the proposal's task-family assignments.

## Industrial/workplace

`foundry_shift` uses a brusque, procedural voice. A hot, noisy workplace governed by rules replaces muted reflection and soft weather imagery with concrete machinery.

`cannery_line` uses a comic, brash voice. Fast physical comedy and workplace dialogue provide alternatives to openings built around quiet dread.


## Medical/emergency

`night_ambulance` uses a staccato, breathless voice. Urgency operates within strict procedures. The writing must not claim that the wrong blood was used without the required laboratory step.

`clinic_triage` uses a warm, domestic voice. Care and fairness meet in a triage queue. Its rule creates an interpersonal moral problem distinct from plots about archive memory.


## Legal/civic

`tenants_hearing` uses a dryly bureaucratic voice. Dull civic procedure replaces formal courtroom drama. A stamp for the wrong building creates a specific continuity error the writer might be tempted to make.

`jury_backroom` uses a tense, claustrophobic voice. A confined room and an unverifiable note create tension while keeping belief separate from established fact, without relying on a manor or old building.


## Maritime/transport

`cargo_inspection` uses a clipped, procedural voice. A working port at dawn broadens the existing night ferry's melancholy. Container and seal facts can be checked and updated in a KB.

`tram_breakdown` uses a comic, commuter voice. A comic group of commuters is stuck in traffic. The fare-refund rule allows different story directions while leaving the central uncertainty unresolved.


## Agricultural/rural

`seed_cooperative` uses a earthy, unhurried voice. Rural life takes an earthy voice rather than a muted pastoral one. A germination-test rule is an established fact that the writer could violate by distributing seeds prematurely.

`veterinary_rounds` uses a blunt, unsentimental voice. Blunt night-time farm work supplies a different voice. A twisted tag leaves both a belief and the identity in the record unresolved.


## Scientific/technical

`storm_lab` uses a spare, precise voice. Instruments and a confirmation procedure determine what can be asserted. The writing must not claim a record storm without preserving the requirement for a second instrument.

`materials_bench` uses a dry, exacting voice. A deadline competes with a test standard. Passing the warped sample is a specific tempting error that can be scored.


## Commercial/market

`fish_market` uses a loud, brash voice. Loud transactions broaden the quiet coastal scenes. An inspector's seal rule and a refund outcome provide concrete, checkable facts.

`hardware_counter` uses a wry, comic voice. Retail misunderstanding supports comedy. An established fact about primer explains a testable mismatch, while a character may believe something else.


## Educational

`night_class` uses a warm, patient voice. A smudged grade raises classroom fairness. Requiring a second reader gives KB and revision tasks a clear established rule to preserve.

`school_play_audition` uses a comic, chaotic voice. Backstage disorder and parental pressure create comedy. The writing must not give someone the lead immediately; callback status stays unresolved.


## Performance/sport

`boxing_gym` uses a hard-edged, punchy voice. Contact sport creates urgency. The commission's wrap-inspection procedure provides an established fact and a tempting disqualification error.

`orchestra_rehearsal` uses a elegant, high-wire voice. A broadcast deadline and a reed problem create tension. Alternative program orders can change the scene without cancelling the broadcast.


## Domestic/family

`kitchen_move` uses a tender, lived-in voice. Family grief operates alongside inventory rules. A price sticker leaves the object's origin uncertain.

`birthday_setup` uses a breezy, comic voice. Warm backyard comedy includes a specific prohibition on retirement. Violating that rule would make a continuity error that can be scored.


## Digital/remote-mediated

`remote_standup` uses a deadpan, screen-lit voice. Online-work satire adds a flat, deadpan voice and facts recorded in logs and tickets. The writing must not reverse the rollback during the call.

`group_chat` uses a breathless, fragmented voice. Fragmented messages provide a different texture. A two-moderator ban rule and an unchecked log allow a belief to remain unproven.


## Religious/ceremonial

`harvest_service` uses a solemn, cadenced voice. Ritual order and a flame rule establish facts the scene must respect. The proposed cold-store cause remains an unchecked belief.

`memorial_walk` uses a quiet, elegiac voice. A torn banner supports civic mourning. Cancellation is a tempting resolution that is forbidden, and wind remains an unproven cause.
