# Steer Scripts - Coverage Companion

Reusable steer scripts for the conversational writing collaborator. Each steer is a user message that continues a session after an opening brief. Placeholders (`{protagonist}`, `{other}`, `{object}`, `{place}`) keep a script composable with any world.

Artifact: `steer-scripts.json` (schema_version 1). Scripts: **36**. Steers: **210**.

## Depth x kind coverage

| band | ADD_CONSTRAINT | CHANGE_DIRECTION | PUSH_BACK | CLARIFY_ANSWER | REQUEST_REVISION | NEW_MATERIAL | SCOPE_CUT | APPROVE | INTERRUPT | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 2-3 | 7 | 3 | 6 | 2 | 2 | 1 | 2 | 3 | 1 | 27 |
| 4-6 | 12 | 5 | 8 | 4 | 7 | 5 | 3 | 6 | 2 | 52 |
| 8-12 | 27 | 14 | 15 | 8 | 17 | 13 | 9 | 20 | 8 | 131 |
| ALL | 46 | 22 | 29 | 14 | 26 | 19 | 14 | 29 | 11 | 210 |

## Response pressure

| pressure | scripts |
|---|---|
| REQUIRES_ASK | 9 |
| PERMITS_DEFAULT | 15 |
| MUST_NOT_ASK | 12 |

## Other checks

- Terse steers (under 12 words): **191/210** (91%), required >= 25%.
- Escalating-conflict scripts (no fast resolution): **6** (SS-A08, SS-B04, SS-B12, SS-C05, SS-C06, SS-C09), required >= 6.
- Scripts by group: standard=26, silent_assumption=10.

## The 10 noticed-a-silent-assumption scripts

These are the exact behaviour the project has no data for: the model silently assumed something, the writer notices, and the first steer is the correction. Each row names what the script tests.

| id | band | pressure | what the model assumed / what is tested |
|---|---|---|---|
| SS-A01 | 2-3 | PERMITS_DEFAULT | Model silently assumed {protagonist} and {other} meet for the first time; steer forces it to revise the relationship without asking what it should be. |
| SS-A02 | 2-3 | REQUIRES_ASK | Model silently assumed a romance. Script checks that the agent stops, names the assumption, and treats the unresolved status as a decision to ask about. |
| SS-A03 | 2-3 | MUST_NOT_ASK | Model silently assigned an action to the wrong character. A clear correction should be executed, not questioned. |
| SS-B01 | 4-6 | REQUIRES_ASK | Model silently made {protagonist} an unreliable narrator. Script tests detecting an unauthorized craft device, reverting it, and asking next time. |
| SS-B02 | 4-6 | PERMITS_DEFAULT | Model silently set the scene at night. Script tests correction of an ambient default the writer never specified. |
| SS-B03 | 4-6 | MUST_NOT_ASK | Model silently killed {other} off-screen. Script tests reversing an unapproved plot event and re-deriving the actual stakes. |
| SS-C01 | 8-12 | MUST_NOT_ASK | Model silently treated {place} as abandoned, contradicting the world. Script tests a long run where the correction must persist across many steers. |
| SS-C02 | 8-12 | REQUIRES_ASK | Model silently granted {protagonist} knowledge of a secret. Script tests knowledge-state correction plus an explicit instruction to ask in future. |
| SS-C03 | 8-12 | PERMITS_DEFAULT | Model silently chose a season. Script tests adopting a stated default and then tracking it as a hard constraint across the revision. |
| SS-C04 | 8-12 | MUST_NOT_ASK | Model silently invented a major backstory death. Script tests reverting canon, re-deriving dependent objects, and holding the new state. |

