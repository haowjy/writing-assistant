# New-core acceptance map — S5.2

Maps the S5.2 acceptance lanes to the design contract and the public producer,
publication, or restore boundary that rejects each forgery. Gate rejections assert a
`ProjectionError` and the first difference/path; disk damage asserts `CorruptRecordError`.

## c. Authority

| Scenario | Acceptance test | Public mechanism and result |
|---|---|---|
| C1 — model final prose and author text cannot choose completion, transition, or reward | `test_task_graph_accept_authority.AuthorityAcceptanceTests.test_model_and_author_text_cannot_route_or_publish_a_reward`; `test_environment_cannot_publish_a_forged_transition_or_terminal_state` | `RolloutEnvironment.commit` rejects a forged reward step and scripted author reply; `store.publish` rejects an event/state relabelled as a terminal transition. The writer prose still routes to the controller's check step. |
| C2 — evaluator status/evidence forgery | `test_evaluator_status_and_family_are_derived_from_admitted_evidence` | Altered result status and evidence are rejected by `commit`; the same forged typed result is rejected by `store.publish` with the head unchanged. |
| C3 — eligibility forgery | `test_eligibility_is_always_ineligible_and_a_forged_true_claim_fails_the_gate` | Full lifecycle records `TrainingEligibilityV1.status = ineligible`; a hash-correct eligible artifact and outcome sibling is rejected at `state.outcome_ref`. |
| C4 — evaluator-family relabelling | `test_evaluator_status_and_family_are_derived_from_admitted_evidence` (`forgery="family"`) | Replaces the admitted deterministic evidence type with another registered evaluator family; both `commit` and `store.publish` reject it. |
| Evaluator artifact bytes changed after publication | `test_rewritten_evaluator_evidence_is_store_corruption_on_open` | Rewrites a reachable evidence artifact and calls `open_head`; store closure reports `CorruptRecordError` and leaves the authoritative head bytes unchanged. |

Design references: §12 category c; adversarial-map rows C1–C4; §6.3's rule that a
cached view does not replace per-operation closure validation.

## d. Privacy

| Scenario | Acceptance test | Surfaces checked |
|---|---|---|
| D1 — private author/evaluator/requirement data stays out of writer-visible context and sampling requests through asks, replies, feedback, checks, and supersession | `test_task_graph_accept_privacy.PrivacyAcceptanceTests.test_full_feedback_lifecycle_keeps_private_canaries_out_of_writer_surfaces` | Every checkpoint's materialized context and context-revision record; all `SamplerInput` and `ToolInput` values; all `SamplingRunner` prepared requests and serialized message/tool/rendering fields; author requests. The fixture confirms its source canaries are present in private artifacts. The evaluator boundary is separately authorized to receive evaluator inputs. |
| E10 — forged first-operation entry context cannot reach a request | `test_forged_entry_context_is_rejected_before_the_first_port_request` | A hash-correct forged root context is passed to `open`; I3 rejects `state.context_ref` before a port request is built. |

Design references: §§6.3 and 12 category d; adversarial-map row D1; S4's documented
`tests.md` gap for mandatory feedback and check turns. The same private ledger canary is
the requirement text, including its superseding feedback text.

## g. Sampling

| Scenario | Acceptance test | Producer and persisted result |
|---|---|---|
| G1 — inline logprob arrays and negative token IDs | `test_adapter_sampling_evidence_rejections_write_no_lineage_records`; `test_malformed_and_unbound_sampling_forgery_is_projection_error_at_publish` | Fresh backend outputs raise `AdapterContractError` before an event/checkpoint/commit is added. Hash-correct disk payloads are rejected by `store.publish` as `ProjectionError` at `artifact.record_type`. |
| G1 — stale verified request | `test_stale_verified_request_is_rejected_before_an_event_is_recorded`; `test_stale_request_replay_is_a_gate_projection_error` | A prepared request from an earlier context raises `AdapterContractError` on commit; the same stale input under a correctly linked event is rejected by the gate as `ProjectionError`. |
| G2 — under-reported token use; malformed or unbound usage | `test_adapter_sampling_evidence_rejections_write_no_lineage_records`; `test_malformed_and_unbound_sampling_forgery_is_projection_error_at_publish` | Token IDs/count mismatch, invalid usage shape, and absent required completion usage fail before publication. Under-reported and unbound usage are also forged into validly linked input payloads and rejected at `store.publish`. |
| G3 — manifest changes after bind and manifest-claim relabelling | `test_manifest_change_after_bind_and_claim_relabelling_are_rejected` | A backend descriptor swap after session bind raises `AdapterContractError` before effects. A group sample claiming a different sealed adapter manifest fails on both producer commit and gate replay. |

Design references: §§4.1a, 6.3 and 12 category g; adversarial-map rows G1–G3.

## i. Newly rejected scenarios and accepted limits

| Scenario | Acceptance test | Result |
|---|---|---|
| X1 — read deletes a file | `test_task_graph_accept_high4.HighFourAcceptanceTests.test_X1_X2_and_oversize_snapshots_fail_at_the_adapter_without_publication`; `test_X1_X2_and_oversize_hash_correct_inputs_fail_at_the_gate` | `AdapterContractError` on the live adapter path; hash-correct forged tool input is `ProjectionError` at the gate. |
| X2 — write commits other bytes or plants a file | Same X1/X2 tests | Wrong requested bytes and an unrelated file are rejected; head and lineage records do not move. |
| X4 — usage under-reported against token IDs | `test_task_graph_accept_sampling.SamplingAcceptanceTests.test_adapter_sampling_evidence_rejections_write_no_lineage_records`; `test_malformed_and_unbound_sampling_forgery_is_projection_error_at_publish` | Producer rejects before lineage records; persisted hash-correct under-report is `ProjectionError`. |
| Oversize snapshot bricks the lineage | `test_X1_X2_and_oversize_snapshots_fail_at_the_adapter_without_publication`; `test_X1_X2_and_oversize_hash_correct_inputs_fail_at_the_gate` | Oversize producer response and persisted effect are rejected; a normal retry from `open_head` can still publish the queued tool result. |
| X6 — parentless mid-run root | `test_X6_parentless_midrun_root_is_rejected_by_entry_fixed_point` | `open` rejects the root at the derived-entry fixed point; no lineage head moves. |
| X3 — fabricated read observation | `test_X3_fabricated_read_observation_is_an_accepted_trusted_adapter_limit` | **Accepted by design.** Tool effect contract keeps files unchanged but does not recompute the observation from files. A trusted adapter may attest a fabricated observation; test names this boundary explicitly. |
| X5 — orphan collect | No S5.2 test; HIGH-5 is deferred by plan §“After the seam” | **Out of scope.** Group collect/head binding and orphan collection are owned by the later HIGH-5 work, not S5.2. |

Design references: §§9, 12 categories g/i, and 13 (“Re-derive local text-tool
observations instead of recording them”); adversarial-map X1–X6. X3 is the trusted-adapter
limit, and X5 remains HIGH-5.

## View-cache soundness and scope audit (§6.3)

| Contract | Acceptance test |
|---|---|
| Warm cache cannot alias sibling checkpoints sharing a head event but holding different state; re-check after a failed publish | `test_task_graph_accept_cache.CacheAcceptanceTests.test_warm_gate_rejects_same_event_sibling_after_a_failed_publish` |
| Only published views enter the view cache | `test_unpublished_candidate_never_enters_the_view_cache` |
| Cache lookup follows closure reads of the entire checkpoint ancestry | `test_cache_lookup_occurs_only_after_the_store_reloads_checkpoint_ancestry` |
| LRU remains bounded and evicts under at least 65 published lineages | `test_lru_evicts_a_lineage_under_load_of_sixty_five_published_views` |
| One gate safely serves concurrent verification calls | `test_two_threads_can_verify_one_warm_head_through_the_same_gate` |
| Full-slice copy-on-read scope-exit audit; returned nested mutation does not escape | `test_full_slice_passes_scope_exit_audit_and_returned_mutations_do_not_escape` with `CWA_TASK_GRAPH_AUDIT_SCOPE_EXIT=1` |
