# Reference data for writing rewards

Start with LitBench's training pairs for preference calibration, keep HANNA as an
independent dimensional check, and consider LiteraryTaste for author-specific taste.
Public releases checked 2026-09-15. No new datasets, judge calls or training ran.

| Dataset | Available signal | Proposed role |
|---|---|---|
| [LitBench-Train](https://huggingface.co/datasets/SAA-Lab/LitBench-Train) | About 43,800 prompt/chosen/rejected story pairs, with timestamps and voting metadata | Judge calibration; later reward-model training |
| [HANNA](https://github.com/dig-team/hanna-benchmark-asg) | 1,056 stories from 96 prompts, three raters, six dimensions: relevance, coherence, empathy, surprise, engagement, complexity | Independent dimensional check; retain existing evaluation-reference role |
| [LiteraryTaste](https://github.com/mj-storytelling/LiteraryTaste) | 60 readers each judging 100 short-text pairs, with stated reading tastes | Style and personalization; preserve individual disagreement |
| [WritingPreferenceBench](https://github.com/WritingPreferenceBench/Writing-Preference-Bench) | 1,800 human-validated preference pairs: 1,200 English and 600 Chinese, across eight writing domains | Additional subjective judge evaluation; filter to relevant domains |
| Existing Gutenberg and Tell Me a Story sources | Human prose without equivalent quality/preference ratings | Curated style examples and corpus diagnostics, not automatic reward labels |

## What the labels establish

LitBench labels derive from filtered Reddit voting signals. Its
[paper](https://arxiv.org/html/2507.00769v1) describes selection controls and an
additional human study, but popularity remains a proxy for literary preference.
Short-form twists and humor may not transfer to restrained prose or long narrative
development. A chosen story is preferred within its pair, not necessarily excellent
in every respect. Inspect examples before selecting rubric anchors.

LitBench has a separate public test release. The original paper reports 2,480 test
pairs while the current Hugging Face collection displays about 2,380. Pin a revision
and count downloaded records rather than copying the paper's denominator. Keep test
examples out of fitting, prompt examples and rubric optimization.

HANNA supplies 19,008 individual ratings and automatic-metric scores. Its older
candidate systems can make comparisons easier than distinguishing two competent
current models. Do not collapse six dimensions into an unexplained quality score.
Existing cataloged HANNA records remain evaluation-only.

The [LiteraryTaste paper](https://arxiv.org/html/2511.09310v1) studies personal versus
aggregate preferences. Its 6,000 judgments are not 6,000 independent story pairs.
This matters when supporting both spare and ornate writing rather than imposing
one preferred voice on every author.

WritingPreferenceBench includes functional writing and non-fiction. Its README
claims length matching but its statistics table lists appreciably different mean
lengths for chosen and rejected responses. Audit actual pairs before relying on
that control or comparing judge accuracy.

## Use with GLM-5.3 through Reka

1. Group shared prompts and stories in an eligible training release, then separate
   judge-development data from a reserved check set before selecting examples.
   Audit overlap with existing WritingPrompts-derived data, including HANNA.
2. Judge blinded pairs and swap presentation order on a bounded subset. Record
   preference agreement, order sensitivity, length preference, explanations and
   failures. Human preference labels provide external evidence, not an obligation
   to force agreement in every case.
3. Refine rubric anchors or select a few varied examples from development data.
   Do not use the reserved check for those choices or include whole corpora in
   every reward request.
4. Score our generated prose with its own brief and relevant KB/source evidence.
   Keep prose quality, instruction adherence and continuity separately visible.
   These datasets do not verify tool execution, wiki navigation or KB extraction.
5. Later compare a specialized preference model with the API judge. Story-pair
   accuracy alone does not establish good KB reasoning or author personalization.

The [LitBench collection](https://huggingface.co/collections/SAA-Lab/litbench) also
lists generated rationales separately. Preserve artifact-level provenance and reuse
terms. The paper describes the upstream preference dataset as MIT-licensed; do not
silently transfer that label to every downstream artifact. Public availability and
admission to our training inventory are separate statuses.

## References are not rewards by themselves

A preference model or calibrated rubric judge can score new writing. Similarity to
a reference corpus mainly measures resemblance. MMD and n-gram distance remain
corpus diagnostics, not proof that one story is good or follows the requested branch.
Human authorship also does not guarantee quality.

To reduce formulaic prose, inspect repetition, generic dialogue, interchangeable
imagery and redundant emotional explanation in context. Include good examples
across styles; do not substitute a word blacklist or universally terse voice for
literary judgment. Combine that evidence with separate task and continuity checks.
