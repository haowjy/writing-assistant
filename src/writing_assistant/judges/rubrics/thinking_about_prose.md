# Thinking about prose

Judge how the target turn’s recorded thinking informs the prose.
Score 2 for a specific writing choice reflected in the prose, 1 for
a relevant consideration with a partial or unclear effect, and 0 for a
material writing problem the thinking misses or worsens.
Use `"unscored":"not_applicable"` when the turn has no prose work.

Return JSON only with a one-item `findings` array. Set `"covers":{"turn":N}`
to the target turn, `"dimension":"thinking"`, `"criterion":"writing_decision"`,
and `value` or `unscored`. Cite the record with
`"cites":[{"turn":4,"note":"…"}]` or `"evidence":[{"path":"…","revision":1}]`.
