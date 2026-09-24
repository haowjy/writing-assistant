# Worlds Extension — Rationale

Companion to `worlds-extension.json`. It covers the 24 new worlds authored to break the
variational collapse seen in the bf16 arm (12/25 pieces opening with the same "damp air of
the archive" stem; self-BLEU 0.71).

## Coverage summary

- 24 worlds, 2 per setting family, across all 12 required families (see table).
- 24 distinct `prose_style` values (no style shared by more than one world, meeting the
  "at most 2" limit).
- No archives, attics, manors, or lighthouses. No characters, objects, or settings reused
  from `worlds.json`.
- Each world has 5 `required` facts plus `belief`, `incidental`, `forbidden`,
  `update`/`new_state`, and a `protected` sentence embedded verbatim in `revision`.
- "Suits" maps to the five scenario families produced by `src/writing_agent/development.py`:
  F1 reply scene, F2 local revision / explore-then-write, F3 alternatives, F4 KB
  construction/update, F5 retrieval with supplied KB.

## Rationale table

| World id | Setting family | Register | Suits | Collapse risk addressed |
|---|---|---|---|---|
| foundry_shift | industrial/workplace | brusque, procedural | F1, F2, F3, F4 | Breaks the muted reflective register with a hot, noisy, rule-bound industrial scene; concrete machinery replaces soft weather imagery. |
| cannery_line | industrial/workplace | comic, brash | F1, F2, F3, F5 | Adds a fast comic workplace voice so scenes need not open on quiet dread; physical comedy gives dialogue-driven alternatives. |
| night_ambulance | medical/emergency | staccato, breathless | F1, F2, F3, F4 | Supplies high-stakes urgency and hard procedural constraints; forbids the tempting "wrong blood used" beat without the required lab step. |
| clinic_triage | medical/emergency | warm, domestic | F1, F2, F3, F5 | Pairs care with triage fairness; the queue rule gives an interpersonal moral pressure distinct from archive memory plots. |
| tenants_hearing | legal/civic | dryly bureaucratic | F1, F2, F3, F4 | Replaces formal courtroom drama with dull civic procedure; the wrong-building stamp is a specific, tempting continuity error. |
| jury_backroom | legal/civic | tense, claustrophobic | F1, F2, F3, F5 | Trapped-room tension and an unverifiable note; keeps the belief/established-fact split sharp without a manor or old building. |
| cargo_inspection | maritime/transport | clipped, procedural | F1, F2, F3, F4 | Working port at dawn replaces the night ferry's melancholy; container/seal facts are checkable and support KB updates. |
| tram_breakdown | maritime/transport | comic, commuter | F1, F2, F3, F5 | Comic ensemble traffic stop; the fare-refund rule lets alternatives diverge without resolving a central uncertainty. |
| seed_cooperative | agricultural/rural | earthy, unhurried | F1, F2, F3, F4 | Rural but not pastoral-muted; germination-test rule supplies a concrete canon fact and a tempting premature-distribution error. |
| veterinary_rounds | agricultural/rural | blunt, unsentimental | F1, F2, F3, F5 | Night farm work with blunt stakes; the twisted tag lets belief and record-identity stay unresolved in a distinct register. |
| storm_lab | scientific/technical | spare, precise | F1, F2, F3, F4 | Instrumentation and confirmation protocol; forbids the record-storm claim, forcing the second-instrument rule to survive. |
| materials_bench | scientific/technical | dry, exacting | F1, F2, F3, F5 | Deadline pressure against a test standard; the warped-sample pass is a specific tempting error that supports scoring. |
| fish_market | commercial/market | loud, brash | F1, F2, F3, F4 | Loud transactional scene breaks the coastal hush; inspector-seal rule and refund outcome are concrete and checkable. |
| hardware_counter | commercial/market | wry, comic | F1, F2, F3, F5 | Retail misunderstanding comedy; the primer fact lets belief diverge from an established, testable mismatch cause. |
| night_class | educational | warm, patient | F1, F2, F3, F4 | Classroom fairness over a smudged grade; second-reader policy is a clean canon constraint for KB and revision tasks. |
| school_play_audition | educational | comic, chaotic | F1, F2, F3, F5 | Backstage chaos and a parent's pressure; forbids the instant-lead outcome and keeps callback status unresolved. |
| boxing_gym | performance/sport | hard-edged, punchy | F1, F2, F3, F4 | Contact-sport urgency; commission wrap inspection supplies a procedural fact and a tempting disqualification error. |
| orchestra_rehearsal | performance/sport | elegant, high-wire | F1, F2, F3, F5 | Broadcast deadline and a reed problem; lets alternatives play on program order without cancelling the broadcast. |
| kitchen_move | domestic/family | tender, lived-in | F1, F2, F3, F4 | Family grief and inventory rules; the price sticker lets object provenance stay genuinely uncertain. |
| birthday_setup | domestic/family | breezy, comic | F1, F2, F3, F5 | Warm backyard comedy; the no-retirement rule is a specific tempting violation that makes continuity scoreable. |
| remote_standup | digital/remote-mediated | deadpan, screen-lit | F1, F2, F3, F4 | Online work satire gives flat deadpan voice and log/ticket facts; forbids an in-call rollback reversal. |
| group_chat | digital/remote-mediated | breathless, fragmented | F1, F2, F3, F5 | Fragmented message-board texture; the two-moderator ban rule and unchecked log let belief stay unproven. |
| harvest_service | religious/ceremonial | solemn, cadenced | F1, F2, F3, F4 | Ritual sequence and flame rule supply strong canon; the cold-store cause stays an unchecked belief. |
| memorial_walk | religious/ceremonial | quiet, elegiac | F1, F2, F3, F5 | Civic elegy over a torn banner; forbids cancellation, a specific tempting resolution, while the wind cause stays unproven. |

## Family tally

Industrial/workplace 2; medical/emergency 2; legal/civic 2; maritime/transport 2;
agricultural/rural 2; scientific/technical 2; commercial/market 2; educational 2;
performance/sport 2; domestic/family 2; digital/remote-mediated 2; religious/ceremonial 2.
