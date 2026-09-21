# Review: evaluation and reward code, `d809665..HEAD`

Review only. No source file was modified. Every finding below was reproduced by running
the code in `/home/jimyao/gitrepos/research/writing-assistant` with `PYTHONPATH=src` and
the stand-in extractor from `tests/test_sample_distribution.py`. The 164 tests across the
touched modules pass, so none of these are caught by the current suite.

Scope read in full: `reward.py`, `prose.py`, `longform.py`, `longform_suite.py`,
`specificity.py`, `scoring.py`, `inference.py`, `paid.py`,
`scripts/probe_context_budget.py`, `scripts/build_longform_suite.py`, and the new tests.
Upstream EQ-Bench was available at `/tmp/eqbench-lf` (revision matches `EQ_REVISION`).

**Status: the critical and major findings were reproduced and fixed.** The fixes are the
commits immediately after this review; this file is kept as the record of what was wrong
and how it was established, including the claims that were refuted. What remains open is
listed under *Not settled* at the end.

---

## Critical

### C1. `group_advantages` gives a false non-zero signal to an all-tie group

**File:** `src/writing_agent/reward.py:216` (`if std == 0.0`), `:225-226`.

The zero-variance path is a float exact comparison. For several identical reward values
the population mean does not round-trip to the same float, so `std` is a tiny non-zero
number, the branch is skipped, and every advantage is `(value - mean) / std`, which is
`±~1.0`. The docstring claims "an all-tie group returns no signal as a reported fact";
it does not.

Reproduced:

```
group_advantages([Reward("ok", 0.1)] * 3)
-> mean 0.10000000000000002, std 1.3877787807814457e-17,
   zero_variance False, advantages [-1.0, -1.0, -1.0]
group_advantages([Reward("ok", 0.2)] * 3)
-> std 2.7755575615628914e-17, advantages [-1.0, -1.0, -1.0]
```

This is reachable from real rewards, not just synthetic ones. `WEIGHTS` are
`{quality .4, intent .3, continuity .2, mechanics .1}`; three rollouts whose three
semantic ratings are all `1` and whose mechanical checks all pass produce value `0.1`
each (`0.1 * 1.0 = 0.1`), i.e. exactly the group above:

```
rollout_reward(judgment(quality=1,intent=1,continuity=1), [check("delivery", True)])
-> Reward(status="ok", value=0.1)
```

So a batch of three uniformly minimal rollouts emits a strong spurious gradient. This is
the worst kind of "looks meaningful but isn't": a real number, a `zero_variance: False`,
and a `frac_zero_std: 0.0` curriculum signal, all wrong.

**Test that should have caught it:** `tests/test_reward.py::GroupAdvantageTests::test_identical_rewards_give_no_signal`
uses `0.7` four times. `0.7` is exactly representable in this arithmetic
(`0.7 * 4 / 4 == 0.7`), so the test passes while the bug survives. Likewise
`test_advantages_never_contain_negative_zero` uses `0.5`.

**Smallest fix:** don't compare a float to zero. Treat `variance < eps` (e.g. `1e-12`) as
zero variance, or return `advantages = [0.0] * n` whenever `std < eps`. Add a subTest for
`0.1` and `0.2`.

---

## Major

### M1. `session_reward` can leave a critically failed stage uncapped

**File:** `src/writing_agent/reward.py:156-193`, specifically the pending test at `:170`
and the cap at `:180-186`.

`rollout_reward` correctly builds `Reward(value=min(raw, CRITICAL_CAP), critical=True)`
for a declared critical failure, but `session_reward` never looks at `critical`. It only
caps when the caller *re-passes* the same failure through `unresolved`. A stage that
already carries the evidence is not enough.

Reproduced:

```
stage = rollout_reward(judgment(all 5s), [check("d", True)],
                       declared_critical=["canon"], observed_failures=["canon"])
# stage.value == 0.25, stage.critical == True
session_reward([stage], Reward("ok", 1.0))
-> value 0.625, critical False, reason None      # broken rollout scores 0.625
session_reward([stage], Reward("ok", 1.0),
               declared_critical=["canon"], unresolved=["canon"])
-> value 0.25, critical True
```

`CRITICAL_CAP` is `0.25`, so a critically failed session yields `0.625 > CRITICAL_CAP`
unless the caller duplicates the failure. The module's own docstring ("A final average
alone hides a failed crucial stage") states the property the code does not implement.

**Smallest fix:** in `session_reward`, treat any `r.critical` among
`stages + [final_state]` as a failure: e.g.
`failed = critical_failures(declared_critical, unresolved) or [r.reason for r in stages+[final] if r.critical]`.

### M2. `sample_distribution` counts artifacts, not attempts, and pools across models/conditions

**File:** `src/writing_agent/prose.py:428-475`, pooling at `:456`, power applied at `:466`.

`texts = [a["text"] for card in cards for a in card["artifacts"] ...]` flattens every
`ok` artifact of every card, then `sample_power` gates on `len(texts)`. It neither keys on
`(model, condition, feature config)` nor restricts to one artifact per attempt. Both
inflate sample power past the declared floors.

Reproduced:

```
# 2 artifacts per card, 4 cards
sample_distribution(cards, ex, embeddings=False)
-> attempts 4, samples 8
   D4 status "ok", power "low", samples 8      # declared floor for D4 is 8
# same scenario id, three different models/conditions
-> samples 3 pooled with no error
```

This matters directly to the held-out suite: `data/scenarios/longform-v1.json` declares
`samples_per_case: 4` and relies on 4 being *below* the D4 floor of 8
(`documented_sampling` reports D4 `insufficient`). But `LF-01` has three `write` files
(`chapter-2/3/4`), so pooling its 4 attempts yields 12 samples and a D4 the suite says it
cannot support. The suite note ("It cannot support MMD, whose sample floor is 20") is
contradicted by the function that is meant to compute it.

`prose_profile`'s config guard is inert on this path: all generated features come from one
`extractor` instance, so the configs are identical and the "Feature configurations cannot
be pooled" check never fires. It cannot detect model or condition pooling at all.

**Smallest fix:** require a single model/condition/feature-config per pool (raise
otherwise), and gate the floors on `len(cards)` (independent attempts), not `len(texts)`,
or explicitly pool one designated artifact id. At minimum, do not let within-attempt
artifacts count toward the distribution floors.

### M3. The sample floors are not enforced on the paths that produce the metrics

**File:** `src/writing_agent/prose.py:332-340` (`prose_profile` computes D4/D6 directly),
`:618-649` (`score_prose`), `:380-405` (`sample_power`).

`sample_power` is only applied inside `sample_distribution`. Two other reachable paths
emit an unpowered number with no marker:

1. `prose_profile(..., repeated_prompt=True)` directly. With 2 samples:

```
prose_profile(texts(2), feats(2), references=feats, repeated_prompt=True)
-> D4 {"status":"ok","value":0.98,...}   # no "samples", no "power"
   D6 {"status":"ok","value":0.24,...}
```

   The advertised floor for D4/D6 is 8; `prose_profile` has no awareness of it.

2. `score_prose` (the single-attempt evaluation entry point) with references set and one
   artifact:

```
score_prose(card(1 artifact), scenario, {"before":{}}, ex, references=refs)
-> D1 {"status":"ok","value":{...}}      # no "samples"; declared MINIMUM_SAMPLES["D1"] == 2
```

   The same `D1` value is `insufficient_samples` when routed through
   `sample_distribution` with one sample, so the two paths disagree about the same
   quantity.

**Smallest fix:** apply the `sample_power` gate inside `prose_profile` (for D1/D4/D6) so
every producer carries `samples`/`power`, and delete the special-casing in
`sample_distribution`. Add a test that `prose_profile(repeated_prompt=True, n<8)` marks
D4/D6 insufficient.

### M4. `holdout_audit` reports `held_out` when it has nothing to compare against

**File:** `src/writing_agent/longform_suite.py:196-215` and `:219-230`;
caller `scripts/build_longform_suite.py:45-55`.

`claimed_hashes` returns `set()` for a missing catalog, and `holdout_audit` treats an
empty claim as a clean audit. `data/processed/` is gitignored, so on a fresh checkout the
training and development catalogs are absent and the build "proves" a holdout it never
checked:

```
holdout_audit(catalog, claimed={"training": set(), "development_used": set()})
-> {"status": "held_out", "benchmark_hashes": 1,
    "claimed": {"training": 0, "development_used": 0}}
```

The build script's docstring says "The build refuses to complete if ... any benchmark work
is already in use by training or by the development suite." With no catalogs it always
completes and records `held_out`. The current artifact happens to show
`{"training": 15, "development_used": 50}` because the files exist locally, but that is
not reproducible from the repository.

**Smallest fix:** have `build_longform_suite.py` require the catalog files (raise if a
path it intends to claim is missing), or change `holdout_audit` to return/require a status
of `unverified` when a claim set is empty. Do not let `claimed_hashes` silently downgrade
a missing catalog to "nothing claimed". `tests/test_longform_suite.py::HoldoutTests::test_a_missing_catalog_claims_nothing`
actively blesses the dangerous half.

### M5. `longform.py`'s "a dropped or invented metric cannot move the score" is not enforced

**File:** `src/writing_agent/longform.py` module docstring (`:20-21`),
`parse_scores` `:261-292`, `judge_score` `:316-318`, `missing_criteria` `:321-325`.

`parse_scores` defaults to `criteria=()`, which accepts every metric. `judge_score`
normalizes over whatever it was handed. `missing_criteria` is never called by either.
Reproduced:

```
parse_scores("Coherent: 12\nVibes: 20\n")
-> {"Coherent": 12.0, "Vibes": 20.0}          # invented metric admitted
judge_score({"Coherent": 0, "Vibes": 20}, negative=[])
-> 10.0                                        # invented metric moved the score
```

A dropped criterion is the same problem in reverse: `judge_score` re-normalizes over the
returned subset, so dropping the weight-5 `forced poetry or metaphor` criterion silently
raises the score. The property only holds if the caller passes `criteria` to
`parse_scores` *and* separately calls `missing_criteria` and acts on it. Nothing wires
that.

**Smallest fix:** make `criteria` mandatory in `parse_scores` (or return the extra/omitted
names alongside the scores), and have `judge_score`/`task_score` return the
`missing_criteria` report so it cannot be ignored.

### M6. `excludes_all` (and `excludes`) pass vacuously on a missing file

**File:** `src/writing_agent/scoring.py:150-169` (path resolution) and `:165-169` (the
new `excludes_all`), with credit taken in `aggregate_checks` `:41-77`.

The kind raises correctly on an empty/missing `texts` list (verified), but when the
target `path` is absent the check is evaluated against `""` and any "nothing forbidden
here" test passes:

```
checks = [
  {"id":"del","kind":"nonempty","required":True,"path":"manuscript/chapter-2.md"},
  {"id":"no-hot","kind":"excludes_all","required":True,
   "path":"manuscript/chapter-2.md","texts":["hot summer day","sunlight"]}]
mechanical_score(... after={} ...)
-> del status "ok" passed False
   no-hot status "ok" passed True
   scores["Q11"] {"value":1.0,"status":"ok","numerator":1,"denominator":1}
```

A required mechanical check that asserts "this must not be present" is credited 1.0 for a
file that was never written. If no companion `nonempty` check fails, `Q3` is also 1.0.
The session closed the "check citing a fact the candidate was never given" hole; this is
the mirror image, a check crediting a file the candidate never produced.

**Smallest fix:** for path-scoped deterministic checks, when the path is missing from
`after`, mark the check `status="unscored"`/`passed=False` (or raise) rather than
substituting empty text. Add a test for a missing `excludes_all` path.

---

## Minor

### m1. `critical_failures` silently returns nothing when handed the frozen declaration set

**File:** `src/writing_agent/reward.py:94-105`.

`declared_check_set` returns `{"ids": [...], "hash": ...}`; `critical_failures` does
`set(declared) & set(observed)`, which for a dict is the keys `{"ids","hash"}`. So the two
functions do not compose:

```
declared = declared_check_set([{"id":"missing-artifact","passed":False}])
critical_failures(declared, ["missing-artifact"]) -> []
rollout_reward(..., declared_critical=declared, observed_failures=["missing-artifact"])
-> value 0.9, critical False
```

Nothing currently wires them, but `declared_check_set` is otherwise unused and is the
obvious "pre-declared ids" producer. Either accept `{"ids": ...}` explicitly or reject a
non-list/str iterable. Related: `Reward(status="absent")` is treated as *unavailable* by
`session_reward:170` and `group_advantages`, i.e. an absence becomes pending; there is no
current producer, but the constant exists.

### m2. `sample_power` can label an empty value `status: ok`

**File:** `src/writing_agent/prose.py:380-405`, used at `:466`.

`sample_power` only looks at `status`. A pool of 8 containing one empty `ok` artifact makes
`self_bleu` return `None`, and `prose_profile` records `measurement(None)` with the default
`status="ok"`:

```
-> D4 {"value": None, "status": "ok", "samples": 8, "power": "low"}
```

A consumer filtering on `status == "ok"` gets no number, and the reason is not
`insufficient_samples`. Also `sample_distribution` never filters empty `ok` texts before
pooling.

### m3. `--ladder` drops `gemma-4-12B` entirely instead of reporting it as an estimate

**File:** `scripts/probe_context_budget.py:129-130`.

A cached model with a config but no local weights is `del`'d. `gemma-4-12B` is cached with
`config.json` (`layers=48, hidden=3840, max_position_embeddings=262144`) but no
`safetensors`, so `params is None` and it is removed; `data/processed/training-ladder.json`
contains only E2B, E4B, 31B, 27B. The `a8f115c` commit message says "12B, 31B and 27B are
estimates from published configurations and are labelled as such in the artifact" — 12B is
absent, not labelled. Keep the row (`measured=False`) using the config-derived layer/hidden
counts, or add 12B to `UNCACHED`.

### m4. `PaidClient.charge` reports a doubled peak estimate under the name `cost_micro_usd`

**File:** `src/writing_agent/paid.py:108-138`, consumed at `:223-225` and
`scripts/creative_writing_v3.py:151,206`.

`charge` multiplies the listed-price token cost by `PEAK_FACTOR` (2) and returns that for
both `cost` and `accounted`, with basis `"peak_ceiling_estimate"`. So
`cost_micro_usd` is not the amount charged; it is a ceiling twice the listed price, and
the ledger sums that. The commit's "$3.79 of the $10 cap" is therefore an upper bound
presented as spend. The basis string discloses the method, so this is partly a naming /
summary-honesty judgement call; if peak pricing is genuine, the field should be named
`peak_estimate_micro_usd` and the summary should say "estimated".

### m5. `task_score` cannot reproduce the upstream curve it documents

**File:** `src/writing_agent/longform.py:16-18`, `task_score:343-357`.

The module says "Set `legacy_curve=True` to reproduce the upstream numbers", but
`task_score` always calls `staccato_factor(...)` with the default (`legacy_curve=False`)
and exposes no switch. Only the low-level `staccato_factor` can reproduce upstream.
Minor API gap, not a wrong number.

---

## Claim-by-claim verdicts

1. **Absent `0.0` vs withheld `unavailable`.** *Confirmed for `rollout_reward` and
   `group_advantages`; broken at the session layer.* `_component_unit` returns `0.0` for
   `absent` and `None` for `unavailable`, and `rollout_reward`/`group_advantages` preserve
   that (`absent` is `SCORED` and contributes; `unavailable` yields `UNAVAILABLE`/`pending`).
   But `session_reward` treats any non-`SCORED` stage as pending, so a `Reward("absent")`
   would become an absence, and (M1) it loses the `critical` flag. No current producer of
   `Reward("absent")`.
2. **`critical_failures` is a declared-only intersection and `CRITICAL_CAP` bounds it.**
   *Refuted as a safety property.* The intersection is correct for a list of ids, and the
   cap holds inside `rollout_reward`, but a genuinely broken rollout still gets a high
   reward through `session_reward` (M1, 0.625) and through the frozen-set/dict trap (m1,
   0.9).
3. **`group_advantages` pending path / zero-variance / `frac_zero_std`.**
   *Pending path correct. Zero-variance path is wrong* (C1). `frac_zero_std` is honest for
   the float value it computes, but because of C1 it reports `0.0` for groups that should
   be `1.0`.
4. **`sample_power` floors enforced everywhere.** *Refuted.* Floors are enforced only in
   `sample_distribution`; `prose_profile(repeated_prompt=True)` and `score_prose` bypass
   them (M3). The floors themselves (`{D1:2, D2:20, D4:8, D6:8}`, reliable
   `{25,50,25,25}`) are explicitly declared unvalidated; that is a defensible judgement
   call, not a defect.
5. **`sample_distribution` pooling.** *Refuted.* It pools across models, conditions,
   feature configs and within-attempt artifacts; the `prose_profile` config guard is inert
   on this path (M2).
6. **Staccato divergence.** *Confirmed.* Upstream (`/tmp/eqbench-lf/core/scoring.py:160-168`,
   revision matches `EQ_REVISION`) computes `factor = 1.0 - ((st_val-5)/13.0)*0.6` on
   `5..18` and `0.6` above, giving `0.4` at 18 then `0.6` at 18.0001. `staccato_factor`
   reproduces that with `legacy_curve=True` and gives the continuous `0.6` at 18 by
   default. The divergence is real, correctly implemented, and the default matches the
   upstream comment's stated 0..40% intent. Defensible.
7. **`longform_suite` refusals.** *Partially.* `verify_anchors` and `holdout_audit` do
   raise (`ValueError`) on the failure they detect. But `holdout_audit` reports `held_out`
   vacuously when the claim sets are empty (M4), and anchor matching is a bare casefold
   substring test, so an anchor can be satisfied by an incidental substring rather than a
   real mention. Fix the empty-claim case; consider word-boundary matching.
8. **`excludes_all` raises on empty/missing list and cannot pass vacuously.** *Half
   confirmed.* It raises on empty and missing lists (verified). It can pass vacuously on a
   missing target file (M6).
9. **Tests test behaviour.** *Mostly, but three gaps.* Tests that would still pass if the
   function were wrong:
   - `tests/test_reward.py::GroupAdvantageTests::test_identical_rewards_give_no_signal`
     (and `test_advantages_never_contain_negative_zero`): uses exactly-representable values,
     misses C1.
   - `tests/test_sample_distribution.py::PoolingTests::test_attempts_of_one_scenario_pool_into_one_distribution`
     (and the floor tests) use one artifact per card, so they cannot see M2.
   - `tests/test_check_kinds.py::ExcludesAllTests::test_passes_when_no_forbidden_string_appears`
     never exercises a missing path, so it cannot see M6.
   - `tests/test_longform.py::ScoreParsingTests::test_declared_criteria_filter_invented_metrics`
     always passes `criteria`, so the permissive default behind M5 is untested.
   - `tests/test_longform_suite.py::HoldoutTests::test_a_missing_catalog_claims_nothing`
     asserts the behaviour that makes M4 possible.
   The context-budget KV tests are genuine arithmetic checks and are fine.
10. **Combined literary score in `reward.py` vs `scoring.py`.** *Confirmed no problem.*
    `reward.py` imports only `catalog.fingerprint`; nothing imports `reward.py` outside its
    tests; `scoring.py` still computes no combined score. The two modules overlap in
    concept (`quality/intent/continuity` ≈ Q2/Q1/Q13) but not in code. Keep them separate;
    the risk is future leakage, not current duplication.

---

## Not settled

- Whether `PaidClient`'s doubled `cost_micro_usd` is intentional peak-pricing
  conservatism or double-counting. The token prices and `PEAK_FACTOR` need the provider's
  price sheet to resolve; either way the field name overstates.
- The exact upstream staccato intent (the upstream *comment* says 0..40%, the upstream
  *code* does 0..60% then a jump). The reimplementation is faithful to the code and to
  the comment under `legacy_curve`/default respectively; which is "the anchor" is a
  judgement call, already documented.
