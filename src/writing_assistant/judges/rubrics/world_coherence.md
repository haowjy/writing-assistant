# World coherence

Judge the named seat from the saved record. Distinguish established facts
from testimony and proposals. Track what the character could know at the time.

Score `continuity` as 2 for coherent work that follows the request, 1 for a
local or repaired error, and 0 for a material contradiction or missing work.
Score `visibility` as 2 for respecting who knows what, 1 for a questionable
inference, and 0 for using or disclosing private knowledge without basis.
Use `"unscored":"not_applicable"` if no knowledge boundary matters.

Return JSON only with a `findings` array containing both criteria. Each has
`"covers":"task"`, `"dimension":"output"`, `criterion`, and `value` or
`unscored`. Cite the record with `"cites":[{"turn":4,"note":"…"}]` or
`"evidence":[{"path":"…","revision":1}]`.
