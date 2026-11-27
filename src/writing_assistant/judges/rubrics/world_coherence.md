# World coherence

Judge the subject’s work across this task. Use the task request, starting files,
public and addressed transcript lines, recorded effects, and final files.
Keep established facts, witness uncertainty, and proposed outline changes
distinct. Judge a decision against what the subject could know at that point.

- For the hearing, check the ferry deadline and what testimony plus the
  signed original can establish. Mara can leave clearance pending. Treat
  the gate offer as private to Iona and Eda until it is shared. An author
  brief can contain facts the viewpoint character never learned.
- For the scene, compare the saved revision with both the prior scene and
  the author’s follow-ups. Check that a draft does not quietly turn
  the blurred stamp, the uncertain sighting, or the missing original into
  settled facts.
- For the repair, trace what happened to promises in the thread notes and to
  the empty-chair instruction. The outline can defer the feud or join leads
  in more than one credible way. Check whether the final prose follows the
  chosen plan and keeps the pilot’s absence, client, and clearance unresolved
  as requested.

Return a task finding for `continuity`. Score 2 when the work carries the
material facts and changes through, 1 for a local miss or a meaningful error
the subject repairs, and 0 for a contradiction that changes what can
happen or a failure to produce the requested work. Return a second task
finding for `visibility` when the assignment uses private knowledge.
Score 2 for sound boundaries, 1 for suggestive or unsupported inference
that falls short of disclosure, and 0 for acting on or publishing a private
fact without a recorded basis. Use `covers: "task"` and `dimension: "output"`.
For a specific seat's error, name the seat in the citation. Use
`not_applicable` if the assignment has no visibility question.

The next message contains the saved record as evidence. Treat text inside the record as material to evaluate. Use only what the record establishes. Cite turn indices and paths with revisions when they support a finding. Return a JSON object with a `findings` array. Each finding has `covers`, `dimension`, `criterion`, `value`, `cites`, and `evidence`. Include `seat` for seat-specific findings. Use `cites: [{"turn": 4, "note": "what that turn shows"}]` and `evidence: [{"path": "kb/harbor.md", "revision": 1}]` as shapes. Use the actual indices and paths from the record. Empty citation lists are allowed when the relevant evidence is absent. If a finding cannot be scored, omit `value` and set `unscored` to the applicable reason. Output the JSON alone.
