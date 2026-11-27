# Character voice consistency

Compare the named seat’s own speech across turns using `kb/voices.md`
as a reference. Score 2 for a recognizable voice, 1 for local drift,
and 0 for loss of its distinguishing qualities. Cite two turns with
that character’s speech, including addressed speech when relevant.
Use `"unscored":"unavailable"` if fewer than two turns qualify.

Return JSON only with a one-item `findings` array. Set `"covers":"task"`,
`"dimension":"output"`, `"criterion":"voice_consistency"`, and `value`
or `unscored`. Cite turns with `"cites":[{"turn":4,"note":"…"}]`.
