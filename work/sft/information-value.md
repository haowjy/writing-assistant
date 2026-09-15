# Useful information, structure and prose density

Measure useful coverage and unnecessary material separately. Maximal facts per word
is not the objective: terse omissions and unreadable compression can score well on
a raw density ratio. This is a proposed extension, not implemented reward code.

## Measurements by output type

| Output | What counts as useful content | Structural evidence |
|---|---|---|
| Planning | Distinct options, consequences, tradeoffs, uncertainties and actionable decisions | Alternatives are easy to compare; accepted choices and proposals remain distinct |
| Conversation | Answers the current question and provides needed context or clarification | Answer is findable; no repeated explanation or needless digression |
| KB | Supported facts and interpretations relevant to the stated purpose and detail level | Navigation, placement, links, retrieval success and clear source/uncertainty status |
| Prose | Plot movement, characterization, atmosphere, subtext, rhythm or setup serving the intended scene | Scene progression and readability appropriate to the requested style |
| Source-grounded nonfiction | Correct explanations, relevant evidence, necessary qualifications and conclusions | Reader can find and understand the answer without irrelevant detail |

Planning and KB work put more semantic quality weight on selection and structure
than on literary style. Statistical prose-profile weights should not transfer to
these artifacts. Conversational acknowledgments and tool observations remain outside
prose extraction. A mixed message may contain separately scored prose and discussion.

## A proposed content-value profile

1. **Useful-content precision:** relevant, supported substantive units divided by
   all substantive units. Report factual support and relevance separately too, so
   a correct but irrelevant claim can be distinguished from an invented claim.
2. **Important-content coverage:** weighted required/useful units covered divided
   by the task-specific reference total. Set importance from the visible purpose,
   not from every fact available in the source. Permit equivalent explanations and
   alternative defensible selections. Freeze labels before reviewing candidates.
3. **Redundancy and irrelevance:** repeated meanings and off-purpose passages, with
   evidence spans. Exact repetition is deterministic; semantic repetition needs
   judgment. A useful recap, cross-link or repeated motif is not automatically waste.
4. **Information density:** unique useful units per 100 words, reported within
   output-type, purpose and length groups. Treat as a diagnostic until validated;
   do not maximize it directly. Fixed extraction rules and deduplication are needed
   to prevent splitting one fact into many units from inflating the score.
5. **Usability:** can a reader answer the intended question, compare alternatives,
   or retrieve needed information? Exact link checks and controlled retrieval tasks
   supplement semantic organization judgments. Heading count is not structure quality.

Coverage counters the incentive to say very little; relevance counters the incentive
to dump every fact. Empty or substantive-content-free outputs have unavailable
density/precision with failed delivery or coverage, not perfect precision. Missing
judge evidence stays unavailable rather than silently becoming a negative label.

Unit extraction, relevance, salience and equivalence require semantic judgments.
Python can calculate ratios from labels, but that does not make them judge-free
measurements. Record labels, source evidence, uncertainty and scoring versions.
Unknown or unverifiable claims need explicit status; opinions and hypotheses are
not false facts merely because they lack external verification.

## Prose needs a different notion of contribution

A description can establish unease without adding an explicit fact. A pause can
develop a relationship. Repeated phrasing can create a deliberate motif. Review
whether passages serve the requested scene and style, not whether each sentence
advances the plot or adds an entity. The judge can explain which passage seems
redundant and what role it might serve; this is a literary assessment rather than
a validated automatic unit count.

The contemplated prose reward splits the current 40% quality component into 30%
literary quality and up to 10% calibrated prose profile. It is not active: verify
relationships with preference data, length/style controls and adversarial examples
first. For planning/conversation/KB, any analogous statistical subcomponent should
be smaller, with semantic usefulness and organization dominating. No new numeric
weight is selected here. MMD remains a corpus/group diagnostic.

## Research basis and limits

- The [Pyramid method](https://www.cs.columbia.edu/~ani/DUC2005/) uses semantic content
  units and importance derived from multiple reference summaries. Adapting salience
  to a writing task's purpose is our design choice, not the original Pyramid metric.
- [SummEval](https://arxiv.org/abs/2007.12626) provides expert/crowd assessments and
  an evaluation framework for summarization. Its relevance and organization lens
  informs our planning/KB review; short news summaries do not validate novel-length
  knowledge selection or literary subtext.
- [FActScore](https://arxiv.org/abs/2305.14251) evaluates support for atomic claims.
  Factual precision does not establish relevance, completeness or prose quality.
- [VERISCORE](https://arxiv.org/abs/2406.19276) distinguishes verifiable claims in
  diverse long-form outputs. This helps avoid treating every opinion or creative
  statement as an externally checkable fact. Fiction uses supplied canon and accepted
  branch decisions as its authority rather than a real-world factuality test.

## Nonfiction scope

Include a small source-grounded nonfiction slice as a proposal for the initial
collection: summaries, explanations, comparisons and briefs. These use existing
task families; add an output-kind label rather than a sixth family. Planning and
KB prose already exercise some expository skills, but fictional-source summaries
do not establish factual reliability in unrelated real-world domains.

Start from supplied reference passages with an explicit reader goal. Include
contrasting candidates that omit an essential qualification, repeat the same point,
add true but irrelevant detail, or compress so far that an explanation stops making
sense. Preserve useful detail rather than rewarding shortness itself. Do not expand
to an unbounded general-writing corpus before testing transfer to the writing agent.
The 100 assignments are unchanged by this proposal; nonfiction admission and these
measurements remain follow-up work.
