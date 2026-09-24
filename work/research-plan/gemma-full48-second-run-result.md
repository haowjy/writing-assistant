# Second intact full48 run — stopped after checkpoint 31

**The v6-qualified fresh-base run is terminal at a sampled, uncommitted group 32.**
It reached 31 complete optimizer checkpoints and 124 committed attempts. Group 32
(index 31, `wave1-train-032`) sampled all four attempts, then stopped before update
32 because one reward was unavailable. The supervisor and worker exited normally
under their failure policy; no timeout or automatic retry fired. The GPU returned
to idle. There is no checkpoint at the attempt level.

Production identity:
`9c5f32959ecf5bccceead3783e67db8144defee6cba3b8c9aa748acb1ed63e80`.
Initial source: `7828a8f`; controlled v6 fit:
[fit result](gemma-full48-fit-v6-result.md). Source-bound v2 evidence must not be
edited or resumed by the existing runner. The planned stop at checkpoint 48 and
second pass did **not** occur. This is not an intact 96-group result.

| Group 32 slot | Recorded outcome | Reward |
| --- | --- | --- |
| 000 | Candidate-invalid final response | 0 (available) |
| 001 | Candidate-invalid final response | 0 (available) |
| 002 | Unsupported raw stopping boundary before scheduled user follow-up | Unavailable |
| 003 | Candidate-invalid tool arguments | 0 (available) |

Slot 002 made four valid tool calls ending in `<|tool_response>` (token ID 50).
Its fifth sampled response was a final answer ending in `<eos>` (token ID 1).
The next scheduled user turn requires `<turn|>` (ID 106) to extend a native Gemma
transcript. `native_suffix` correctly refused to fabricate that boundary but the
backend classified the result as infrastructure/unavailable. Its exact token
ledger, response trace, workspace, and original unavailable reward survive under
`trainer/groups/step-000031-87b060b85fc043198c5ef1f8807f47c5/attempt-002/`.
All four ledgers passed `verify_tokens` in a read-only audit; checkpoint 31 passed
`verify_checkpoint`. Scoring an *explicitly reclassified copy* of slot 002's saved
result with the original mechanical callback returned available reward 0. Thus
all four recorded candidate outcomes could be zeros in a **new, documented
policy**, but the old unavailable record remains unchanged.

A focused regression (red before the change, green afterward) now checks that an
EOS-ended final answer with a scheduled follow-up becomes a candidate-invalid
failure **before another model action**, preserving its sampled token IDs. The
same EOS answer with no follow-up remains a completed answer. A group-level
regression checks that four attempts are scored together rather than pended;
unknown framing and host failures remain unavailable. This is a code-path test,
not live-model qualification or permission to mutate v2.

For exploratory continuation, decide explicitly whether to build a new-identity
fork importing checkpoint 31 **and exactly these four recorded token ledgers**, or
to archive group 32 and allow a documented resample. Neither recovery path exists
in the current runner; normal resume refuses any sampled uncommitted group.
A clean fresh-base rerun is a different objective and need not be assumed for
exploratory code hardening. Mechanical rewards cannot establish prose quality.

Raw run evidence:
`/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/full48-production-v2/`.
Supervisor `supervision/f8ec5cb4bbe24b21870a7ab38f5e8437/` ended `failed`
with `GroupPending: Unavailable reward/infrastructure: whole group pending`.
