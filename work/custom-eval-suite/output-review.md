# Review of the thinking-enabled five-case pilot

Thinking-enabled E2B-IT executed tools in all three workspace tasks. The resulting
writing and knowledge work still need substantial improvement. This is a qualitative
review by the current assistant, based on visible instructions, source files, final
outputs, and tool traces. It is not a calibrated Astra score or a human acceptance
review. No additional generation or grading calls were made.

[Original artifacts, including thinking](../../runs/pilot-e2b-it-thinking-2026-09-13/review.md)
and [run summary](thinking-pilot.md).

## F1-01: Direct prose

**Major continuity and style problems despite passing mechanical checks.**

The scene stages the accusation, remains centered on Mara, and ends without opening
the upper room. It returns prose without a title and meets the measured word budget.

However, Ilan says “I haven’t touched it” immediately before “I moved it above the
tide line.” That reads as an accidental contradiction, with no reaction or framing
that makes it intentional. Later, Mara “looked at the ledger” while contemplating
its absence. The key becomes “missing,” although the supplied premise specifies
that the retired keeper has it. The scarf becomes a symbolic barrier even though
the source says nobody refers to it again.

The requested restrained, concrete style is weak. “The silence fractured,” “unsaid
futures,” and the evidence being locked behind an unspoken agreement substitute
abstract emotional explanation for observable action. A disagreement happens, but
its emotional meaning is repeatedly narrated rather than developed through choices.

## F2-06: Notes to a saved comedy scene

**File task succeeds; scene development is weak.**

The model reads the notes, writes the correct file, and leaves the source unchanged.
It preserves the projector-check restriction and uncertainty about the reel's contents.
The new cinema name, committee member, and character ages are inventions permitted
by this loose request, not continuity violations by themselves.

The result is a script with cast notes and stage directions. The user asked for a
“scene,” so this is a defensible format choice; it should not be retroactively failed
for lacking a prose-only instruction. It does expose a suite design issue: our file
selector treats the entire script, including metadata, as prose.

The comic mechanism weakens when Ruth immediately identifies Emil as the cleaner,
yet the mistaken-authority situation continues without a convincing explanation.
A historical preservation committee dismisses possible historical festival footage
as inappropriate, with little motivation. Much of the humor depends on generic
fussiness and embarrassment. The ending explains that the scene ends on tension
between formality and absurdity instead of delivering a comic turn.

## F3-03: Brainstorming

**Three distinct approaches, but poor grounding and limited specificity.**

Interrogate, observe, and offer comfort are meaningfully different social choices.
Each has an explanation and possible consequences; no scene is drafted and Leda's
actual reason for traveling remains undecided.

However, the suggestions repeatedly assume “unusual phenomena,” “strange
occurrences,” and possible supernatural activity. The prompt only says that any
unusual phenomena already in the premise should be retained; this premise contains
none. That conditional has been converted into invented world information.

The ideas barely use 1936, the strike, the unsigned ticket, Bell Quay, or the
captain's exclusive knowledge of the revised sailing time. The justifications mostly
explain generic narrative effects. Refusing a drink also does not establish that
Leda's interpersonal barrier is “absolute.” Stronger alternatives would use the
specific logistical constraints to create different choices and consequences.

## F4-08: Wiki construction and revision

**Useful edit discipline; unreliable knowledge interpretation.**

The model reads the chapter, writes kb/index.md, leaves the source untouched, and
uses an exact patch for the accepted revision. It correctly makes no canon edit
for the draft-only suggestion that Noor loses her papers. The accepted provisional
correction and her retained papers appear in the final page.

The source explicitly leaves the relationship between the system update and map
problem unestablished. The wiki describes characters dealing with the “fallout of
the record synchronization issue,” promoting a hypothesis to an explanation. It
also turns the dried stamp pad into a “tangible clue” without supporting evidence.
Keeping or omitting that incidental object would both be defensible; inventing its
significance is the problem.

The missing surveyor's signature is omitted. That is a plausible high-value detail
for a next-chapter reference about disputed documentation. Its importance is a
purpose-based editorial judgment, not a universal requirement to retain every fact.

A single page with headings is adequate for this tiny input. The request does not
justify failing it merely for lacking multiple pages. It nevertheless provides no
internal links, source links, or meaningful navigation test, so this sample cannot
establish the model's ability to build a navigable multi-page wiki. The accepted
revision replaces the authorization bullet without an explicit history entry;
relevant history preservation needs semantic review rather than a mechanical pass.

## F5-05: Writing from the wiki

**Fails to ground the scene in the KB.**

The model reads only kb/index.md and never follows its three links. It writes to
the correct path and leaves the KB unchanged, but misses both evidence-retrieval
checks and exceeds the 120–220-word limit.

The resulting scene introduces a disputed charter and an unspoken pact while
omitting the locked seal, the copying allegation, the expert's limited knowledge,
and the courthouse's remembered vow. Creative invention is allowed, but it does
not substitute for consulting the explicitly requested continuity reference.
Omission alone is not necessarily contradiction; the strong finding here is the
observed retrieval failure and lack of grounding.

The prose relies on familiar atmosphere: thick air, old parchment, damp stone,
dancing dust motes, carefully modulated voices, and veiled truth. These overlap
with the direct-prose scene's vocabulary and emotional presentation. The last
sentence explicitly announces that uncertainty remains unresolved rather than
ending on an action that leaves it open.

The file also starts with a literal “drafts/scene.md:” label. Our whole-file
extraction includes that label in prose measurements. The response repeats the
draft, but only the designated file should count in this condition.

## What to change next

- Keep thinking enabled for ordinary Gemma chat runs. Treat thinking-off runs as
  explicit ablations; preserve the earlier results as recorded.
- Review semantic grounding separately from tool execution: contradictions,
  unsupported causal claims, character knowledge, and use of retrieved evidence.
- Make prose extraction identify metadata and alternate writing formats. Do not
  silently strip text or rewrite these saved outputs to make them pass.
- Test KB navigation with enough content and explicit retrieval needs to distinguish
  index reading from reading the underlying facts. Do not require irrelevant reads.
- Improve scenario wording where a generic conditional primes unsupported genre
  content. Preserve this version and its results when creating a revised case.

Four cases having no failed mechanical checks is not evidence of four good outputs.
The most immediate content weaknesses are grounding, interpretation, and specific
scene development. Distribution metrics can describe repetition but will not by
themselves detect these failures.
