# Follow-up messages for longer writing collaborations

This document explains the 36 reusable scripts in [steer-scripts.json](../../training-data/steer-scripts.json), which supply 210 user messages after an opening writing request. It shows which kinds of feedback they contain, how long the conversations are, and how ten scripts test recovery when the writer notices an assumption the agent never stated.

A “steer” is one user follow-up message. Each script is a sequence of steers. The JSON uses `schema_version: 1`. Placeholders keep scripts reusable across worlds: `{protagonist}` is the main character, `{other}` the second character, `{object}` the story object and `{place}` the setting.

## All nine kinds of feedback occur at every conversation length

A depth band counts follow-up messages in a script: 2–3, 4–6 or 8–12. The table shows that every kind occurs in every band; the longer scripts supply more messages overall. Each kind's exact JSON label is retained so it can be found in the data.

| User action and JSON kind | 2–3 steers | 4–6 steers | 8–12 steers | Total |
|---|---:|---:|---:|---:|
| Add a requirement (`ADD_CONSTRAINT`) | 7 | 12 | 27 | 46 |
| Change the creative direction (`CHANGE_DIRECTION`) | 3 | 5 | 14 | 22 |
| Object to the agent's choice (`PUSH_BACK`) | 6 | 8 | 15 | 29 |
| Answer a clarification (`CLARIFY_ANSWER`) | 2 | 4 | 8 | 14 |
| Ask to revise existing work (`REQUEST_REVISION`) | 2 | 7 | 17 | 26 |
| Supply new story material (`NEW_MATERIAL`) | 1 | 5 | 13 | 19 |
| Reduce the requested scope (`SCOPE_CUT`) | 2 | 3 | 9 | 14 |
| Approve the work or direction (`APPROVE`) | 3 | 6 | 20 | 29 |
| Interrupt the current task (`INTERRUPT`) | 1 | 2 | 8 | 11 |
| All kinds | 27 | 52 | 131 | 210 |

## Scripts differ in whether the agent should ask a question

“Response pressure” is the JSON label for the expected response to an unresolved choice. Nine scripts use `REQUIRES_ASK`: the agent needs to ask about a consequential unresolved decision. Fifteen use `PERMITS_DEFAULT`: the agent may state a default and proceed. Twelve use `MUST_NOT_ASK`: the instruction already supplies or delegates the choice, so another question would obstruct the work.

There are 26 ordinary scripts (`standard`) and ten scripts in which the writer notices an unstated assumption (`silent_assumption`). Of all messages, 191/210 contain fewer than 12 words, reported here as 91%; this exceeds the required 25%. Six scripts sustain or escalate conflict without a quick resolution, meeting the minimum of six: SS-A08, SS-B04, SS-B12, SS-C05, SS-C06 and SS-C09. These are script identifiers, not additional feedback categories; the A, B and C groups correspond to the three depth bands below.

## Ten scripts begin with the writer correcting an unstated assumption

The project lacked examples of this interaction: the agent silently invents a fact or creative choice, and the writer's next message corrects it. The following scripts test both the immediate correction and whether later work respects it.

### Short conversations: 2–3 steers

- `SS-A01` permits a stated default (`PERMITS_DEFAULT`). The agent assumed the two characters were strangers. The correction requires revising their relationship without asking what it should be.
- `SS-A02` requires a question (`REQUIRES_ASK`). The agent assumed a romance. It must stop, name the assumption and ask about the relationship that remains undecided.
- `SS-A03` requires action without a question (`MUST_NOT_ASK`). The agent assigned an action to the wrong character; it should execute the clear correction.

### Medium conversations: 4–6 steers

- `SS-B01` requires a question (`REQUIRES_ASK`). The agent made the protagonist an unreliable narrator. It must detect and reverse this unapproved device and ask next time.
- `SS-B02` permits a stated default (`PERMITS_DEFAULT`). The agent set the scene at night without instruction. The script tests correction of that background choice.
- `SS-B03` requires action without a question (`MUST_NOT_ASK`). The agent killed the other character off-screen. It must reverse the unapproved event and work out what is now at stake.

### Long conversations: 8–12 steers

- `SS-C01` requires action without a question (`MUST_NOT_ASK`). The agent treated the place as abandoned, contradicting the world. The correction must persist across many later messages.
- `SS-C02` requires a question (`REQUIRES_ASK`). The agent gave the protagonist knowledge of a secret. It must correct who knows what and respect an explicit instruction to ask in future.
- `SS-C03` permits a stated default (`PERMITS_DEFAULT`). The agent chose a season. It must adopt a stated default and preserve that season as a firm constraint through revision.
- `SS-C04` requires action without a question (`MUST_NOT_ASK`). The agent invented a major death in the backstory. It must restore the established facts, reconsider objects affected by that death and maintain the corrected state.
