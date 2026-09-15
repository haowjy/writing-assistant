# Reference data for writing rewards

Start with LitBench's training pairs for preference calibration, keep HANNA as an
independent dimensional check, and consider LiteraryTaste for author-specific taste.
Public releases checked 2026-09-15. No new datasets, judge calls or training ran.

## Existing reward models before custom training

Do not make a new reward-model training project a prerequisite for the first 100
tasks. Compare a pretrained scorer against GLM on suitable preference-development
data and representative training-only outputs first. The task definitions alone
do not contain reward-training labels.

| Released candidate | Interface | Practical role |
|---|---|---|
| [Skywork-Reward-V2-Qwen3-4B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B) | Scalar sequence-classification score | First local throughput/quality candidate |
| [Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B) | Same scoring approach with a larger backbone | Compare if 4B judgments are inadequate; higher memory cost |
| [RM-R1-Qwen2.5-Instruct-7B](https://huggingface.co/gaotang/RM-R1-Qwen2.5-Instruct-7B) | Generates a rationale and pairwise preference | Inspectable local judging alternative; generation adds latency |

These are general reward models, not validated judges of our literary styles or
KB tasks. Skywork recommends inputs within its 16,384-token training length and
omitting system messages from its scoring template. Its raw score is not a calibrated
1–5 literary rating. RM-R1's published interface compares two responses; using it
for single-output rewards requires an explicit comparison/aggregation design.

The LitBench paper reports trained writing verifiers, but the linked public
collections inspected here expose datasets, not confirmed downloadable verifier
checkpoints. StoryAlign/StoryReward describes a planned model release. IP-GRM's
repository claims released weights, but a usable checkpoint link was not confirmed.
Do not list these as installed or ready-to-run models on the strength of a paper
announcement. [LitBench collection](https://huggingface.co/collections/SAA-Lab/litbench),
[StoryAlign paper](https://arxiv.org/abs/2605.04831),
[IP-GRM repository](https://github.com/ShadeCloak/IP-GRM).

A 4B BF16 model has roughly 8 GB of raw parameter storage; an 8B model roughly 16 GB.
These arithmetic estimates exclude runtime memory, input activations and batching.
On the 3090, measure isolated scoring and model-swap overhead instead of assuming
reward inference and writer training can share the GPU efficiently. A scalar model
avoids generating a long critique, but actual throughput and ranking quality remain
unmeasured. Quantization also needs a ranking-agreement check.

Keep mechanical tools/files/navigation checks in code. GLM can initially assess
source-backed interpretation, continuity and prose. A pretrained scalar scorer is
a comparison signal until validated, not an automatic replacement for those passes.
If judging later dominates cost or latency, fine-tune an existing scorer using
eligible public preference pairs plus reviewed, representative in-domain pairs.
Include fluent-but-inconsistent and compliant-but-poorly-written examples so it
cannot learn prose polish as a substitute for task success. Keep the reward model
fixed during a writer-training experiment and evaluate with a separate held-out judge.

No checkpoint was downloaded or executed for this review.

## Available preference and rating data

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
