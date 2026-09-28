# Transition-seam acceptance map (design §12)

Sections a, b and j come from S5.1; c, d, g, i and the §6.3 cache audit come from S5.2.

Test IDs below use the `unittest` module/class/method name. A bracketed value is the
corresponding `subTest` case. Attacks go through the public seams:
`RolloutEnvironment.commit`, `TaskGraphStore.publish`, or on-disk rewrites followed by
`open_head`/`verify`. Tests may call `derive_input` to build an honest candidate before
mutating it; that does not bypass the attack seam.

## S7.3 restored owner-level guards

| Guard | Direct owner test |
|---|---|
| Initial requirement supersession, evaluator terminal evidence, packet/version consistency, and author-packet typing | `test_task_graph_lost_guards.AdmissionGuardTests` |
| Scripted-author decision, proposal, question and positional-selector limits | `test_task_graph_lost_guards.ScriptedPolicyGuardTests` |
| Context-operation budget, entry author-call limit and queued-tool quiescence | `test_task_graph_lost_guards.BudgetGuardTests` |
| Mixed ask/file batches, intake depth, and deeply nested JSON | `test_task_graph_calls.IntakeAndParserTests.test_mixed_ask_batch_is_all_invalid_and_intake_depth_is_bounded`; `test_deep_json_arguments_do_not_escape_as_recursion_error` |
| A runtime commit contains exactly one event | `test_task_graph_gate.LineageGateTests.test_gate_rejects_a_commit_with_more_than_one_event` |
| Context and commit domain tags; entry system-prompt identity | `test_task_graph.TaskGraphRecordsTest.test_canonical_fixture_and_round_trip`; `test_task_graph_derive_entry.DeriveEntryTests.test_entry_system_prompt_identity_is_pinned` |
| A disk-published lineage without the supported transition-semantics pin is rejected by both head readers | `test_task_graph_accept_b.PersistedForgeTests.test_unpinned_transition_semantics_are_refused_by_open_and_read_head` |

## a. Forged writer, tool, author, check and terminal records

Every `commit` row checks the specified exception class and preserves the previous head.
Every persisted event/state row goes through `store.publish`, checks `ProjectionError`, the
first differing path, and the unchanged head. The tool-result observation note is a finding,
not a successful rejection.

| Old source / family | Forgery represented on the new wire | New test ID |
|---|---|---|
| `test_task_graph_writer.py:673-844`, `test_complete_writer_event_field_ownership_and_phases`: action pre/post phase, pre-status, position, continuation, owned state and history subtests | Derived output fields no longer live in patch `set`/`history_set`; the candidate's derived state is forged instead | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[phase, node_id, visit_id, entry_contract, start_checkpoint, loop_counts, instance_ref, cursor, tool_queue, author_request, check_requests, external_requests, applied_responses, feedback_cursor, action_history, tool_result_history, context_ref, requirements_ref, decisions_ref, disclosures_ref, author_packet_ref, budgets_ref, rng_ref, external_inputs_ref, versions_ref, outcome_ref, provenance_ref, files]`; the old `position.lineage_id` forgery maps to category b's relineaged event/state candidate |
| `test_task_graph_writer.py:673-844`, result ownership/history/post-phase and sampled-stop ownership/post-phase subtests | Result ordinals, call cursor, file tree, budgets, stop state and history are derived from the typed input; mutate the corresponding state/event result | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[cursor, tool_result_history, budgets_ref, files, phase]`; `test_derived_event_forgery_matrix_rejects_at_first_event_path` `[kind, actor, audience]` |
| `test_task_graph_writer.py:305-329`, `test_owned_action_values_reject_before_cas_and_on_recovery` (`action_ids`, `model_calls`) | State history/budget outputs | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[action_history, budgets_ref]` |
| `test_task_graph_writer.py:331-391`, `test_action_and_result_cannot_drop_runtime_log` | The old `external_inputs_ref` patch is replaced by a state-identity forgery | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[external_inputs_ref]` |
| `test_task_graph_writer.py:393-478`, `test_first_stop_log_reason_and_ordinal_mutation_matrix` | Runtime-log reason/ordinal/actor fields are no longer on the wire; test the derived stop/actor and checkpoint state instead | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[budgets_ref, action_history]`; `test_derived_event_forgery_matrix_rejects_at_first_event_path` `[actor]`; `test_author_evaluator_and_terminal_inputs_reject_semantic_mismatches` `[terminal_step]`; relineage is covered by category b |
| `test_task_graph_writer.py:479-605`, `test_action_and_sampled_trace_cross_binding_matrix` | Sampling and trace evidence is carried in the typed writer input; request-copy attacks are removed because those fields no longer exist; binding-specific claims are category g | `test_task_graph_accept_a.WriterAndToolInputForgeryTests.test_writer_turn_inputs_reject_active_identity_and_sampling_binding_forgeries` `[action_id, context_revision_ref]`; group/sampling claim rows are in S5.2 |
| `test_task_graph_writer.py:606-670`, action authority and trace restore probes | Outcome and trace claims are no longer effect patches; forged outcome state or event identity is compared to the writer-turn derive | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[outcome_ref]`; `test_derived_event_forgery_matrix_rejects_at_first_event_path` `[actor]` |
| `test_task_graph_writer.py:1126-1211`, forged result origin/hash/charge and independent result fields | Action/result IDs, execution hashes, charge and `file_delta` are derived, not input fields | `test_task_graph_accept_a.WriterAndToolInputForgeryTests.test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[call_id, fake_delta, false_success_with_delta]`; derived counters/tree/history are covered by `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[budgets_ref, files, tool_result_history]` |
| `test_task_graph_writer.py:1212-1257`, all seven tool-result forgeries before head publication | Same typed tool-input and derived-state boundary | `test_task_graph_accept_a.WriterAndToolInputForgeryTests.test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[call_id, fake_delta, false_success_with_delta]`; `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[cursor, budgets_ref, tool_result_history]` |
| `test_task_graph_writer.py:1258-1285`, context event cannot hide file change | No generic context patch can author a file delta | `test_task_graph_accept_j.DowngradeForgeryTests.test_invalid_queue_patch_rejects_as_derived_state_mismatch` and `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[files]` |
| `test_task_graph_scripted.py:812` (`validate_author_request_effect`, forged actor) | Event actor is derived from the accepted input kind | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_event_forgery_matrix_rejects_at_first_event_path` `[actor]` |
| `test_task_graph_scripted.py:1033, 1042, 1051` (`validate_check_result_effect`: request/check/packet/evidence/status, decisions ref, actor) | Target checkpoint, check contract and packet bindings are fields of the derived private check request, not `EvaluatorResultV1`; status/evidence failures from the evaluator port are `AdapterContractError` on fresh commit, while a stale request binding remains `ProjectionError` | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_author_evaluator_and_terminal_inputs_reject_semantic_mismatches` `[evaluator_result, index=0..2]`; event actor and derived decisions are covered by `test_derived_event_forgery_matrix_rejects_at_first_event_path` `[actor]` and `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[decisions_ref]` |
| `test_task_graph_scripted.py:1120, 1129` (`validate_terminal_effect`: reward/status and actor) | Reward/status fields are now a derived `OutcomeV1`; transition choice is a directive-bound environment input | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[outcome_ref]`; `test_author_evaluator_and_terminal_inputs_reject_semantic_mismatches` `[terminal_step]`; event actor is `[actor]` |
| `test_task_graph_writer.py` `EnvironmentBatch.append_record` patch probes (`owned action`, action/result log, authority/trace, result origin/hash/charge, independent/all-seven tool-result families) | Patches are removed; each semantic family is submitted as a typed input or a forged published result | `test_writer_turn_inputs_reject_active_identity_and_sampling_binding_forgeries`; `test_tool_observation_inputs_reject_queue_and_effect_forgeries`; `test_derived_state_forgery_matrix_rejects_at_first_state_path`; `test_derived_event_forgery_matrix_rejects_at_first_event_path` (IDs above) |

### The 26-case `judo-forgery-matrix.py`

| Old matrix row | New result / test ID |
|---|---|
| `record.action_id`, `record.result_id` | Not input fields on `ToolObservationV1`; result identity is derived. Forged output is checked by `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[tool_result_history]` and the event/state candidate rows. |
| `record.call_id` | `test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[call_id]` |
| `record.observation` | `test_task_graph_accept_privacy.PrivacyAcceptanceTests.test_X3_fabricated_read_observation_is_an_accepted_trusted_adapter_limit` — X3 documents that the trusted tool adapter may attest a fabricated read result while leaving files unchanged. |
| `record.file_delta(fake path)`, `delta.forged(write)` | `test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[fake_delta]` |
| `record.before_execution_hash`, `record.after_execution_hash` | Not wire fields; the producer derives the file tree and event. Test forged state `[files]` and tool effect `[fake_delta]`. |
| `record.budget_charge.tool_calls`, `record.budget_charge.read_tokens` | Not wire fields; test derived counters `[budgets_ref]`. |
| `record.loss_eligibility`, `message.loss_eligible`, `message.content`, `message.origin` | Derived context/message values, not tool input fields; test state context identity `[context_ref]` (category b) and event actor / event identity `[actor]`. |
| `record.extra_field` | Strict input records have no extensible patch payload. A recomputed-hash disk rewrite is rejected as `ProjectionError` at `event.payload_ref.newly_forged_field` by `test_task_graph_accept_b.PersistedForgeTests.test_hash_correct_payload_with_extra_field_is_projection_rejection`. |
| `record.record_type` | An unregistered record discriminator in a recomputed-hash disk rewrite is rejected as `ProjectionError` at `event.payload_ref.record_type` by `test_task_graph_accept_b.PersistedForgeTests.test_hash_correct_payload_with_unknown_record_type_is_projection_rejection`. |
| `changes.next_call=0`, `changes.tool_queue=[]`, `changes.phase=checking` | No generic state patch is accepted; `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[cursor, tool_queue, phase]`. |
| `history.tool_result_count(incremented)`, `history.action_count(incremented)` | Derived history mismatch: `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[tool_result_history, action_history]`. |
| `changes.requirements_ref`, `changes.author_packet_ref`, `changes.provenance_ref` (same-value added-key probes) | The old patch key no longer exists. Actual changed derived refs are tested at `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[requirements_ref, author_packet_ref, provenance_ref]`. |
| `write: observation ok=False but files changed` | `test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[false_success_with_delta]`. |
| `read: ok but deletes draft.txt (HIGH-4)` | `test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[read_delete]`. |

## b. Rebuilt `integrity-forgery.py` commits and tampering

The 12 old commits are rebuilt from typed writer inputs. Each row rewrites canonical event,
checkpoint, commit and head bytes with valid identities, then opens the head using a fresh
gate. The head is unchanged by the rejected open, and each semantic mismatch asserts its
first differing path.

| Old `integrity-forgery.py` row | New wire forgery | New test ID |
|---|---|---|
| `submit_final/writer_action.budgets_ref` | Counter no longer sits in a patch; rewrite the derived budget artifact/ref | `test_task_graph_accept_b.PersistedForgeTests.test_integrity_forgery_twelve_valid_hash_commits_reject_on_cold_open` `[writer_counter]` |
| `author_reply/tool_result.budgets_ref` | Derived budget identity mismatch | Same test `[author_counter]` |
| `compaction/context_changed.budgets_ref` | Derived budget identity mismatch | Same test `[context_operations]` |
| `submit_final/context_changed.budgets_ref` | Derived storage-charge identity mismatch | Same test `[context_storage]` |
| `author_reply/decision_disclosed.decisions_ref` | Replace the derived decisions artifact reference | Same test `[undeclared_decision]` |
| `author_reply/context_changed.context_ref` | Replace the chained context revision with the earlier revision | Same test `[private_context_message]` |
| `submit_ask/context_changed.context_ref` | Same derived-context identity forgery for the ask path | Same test `[private_context_message_from_ask]` |
| `reward/external_response.outcome_ref.training_eligibility` | Eligibility is inside derived `OutcomeV1`, not an effect patch | Same test `[eligibility]` |
| `terminal_outcome/termination_recorded.outcome_ref.task_status` | Derived outcome status/ref mismatch | Same test `[false_status]` |
| `reward/external_response` retyped as `seed_attached` | No generic event reducer; use a supported-but-wrong event kind so the wire remains well-formed | Same test `[unsupported_kind]`; category j also covers no generic input route |
| `reward/external_response` retyped as `request_entered` | No event route for a retyped event | Same test `[relabelled_kind]` |
| `check_next/check_recorded` retyped as `budget_charged` | Event kind must equal the input derive's event | Same test `[check_as_budget]`; `test_task_graph_accept_j.DowngradeForgeryTests.test_retyped_admitted_event_rejects_at_event_kind` |
| One on-disk artifact byte changed without recomputing identity | The valid JSON body no longer matches its content-addressed name | `test_task_graph_accept_b.PersistedForgeTests.test_byte_tamper_after_warm_view_is_corrupt_and_does_not_move_head` (`CorruptRecordError`, authority ref bytes unchanged) |
| Parentless root checkpoint carries supplemental refs | Runtime roots may not add `artifact_refs` | `test_task_graph_accept_b.PersistedForgeTests.test_root_checkpoint_with_supplemental_refs_is_rejected` (`ProjectionError` at `checkpoint.artifact_refs`; no head is created) |
| Forged pre-state phase/status | Parentless root must be a fixed point of entry derivation | `test_task_graph_accept_b.PersistedForgeTests.test_forged_parentless_entry_state_fails_the_root_fixed_point` `[pre_phase, pre_status]` |
| `position.lineage_id` changed together with a forged writer event | A different lineage id is valid only when the input derives that lineage start | `test_task_graph_accept_b.PersistedForgeTests.test_relineaged_writer_event_is_not_a_member_start` (`event.lineage_id`) |
| Hash-correct forged typed state artifact | Codec validation reports the typed artifact's own field path, not the event-payload label | `test_task_graph_records.RecordClosureTests.test_forged_typed_artifact_reports_its_own_codec_path` (`state.outcome_ref.forged_field`) |
| On-disk payload body changed and hash/name recomputed | A forged typed record must be a `ProjectionError`, not store corruption | Separate live assertions: `test_hash_correct_payload_with_extra_field_is_projection_rejection` (`event.payload_ref.newly_forged_field`) and `test_hash_correct_payload_with_unknown_record_type_is_projection_rejection` (`event.payload_ref.record_type`). |

## j. Generic-reducer downgrade

| Old source | New route | New test ID |
|---|---|---|
| H2, `integrity-forgery.py --downgrade-only`: `seed_attached` or retyped generic event on an admitted lineage | `seed_attached` is not a new-core event/input kind; a well-formed but unsupported event kind is rejected by the gate | `test_task_graph_accept_j.DowngradeForgeryTests.test_unsupported_generic_input_kind_has_no_commit_route`; `test_retyped_admitted_event_rejects_at_event_kind` |
| H3, `judo-forgery-matrix.py`: empty/invalid queue patch through generic effect | No generic reducer or queue-patch input exists; forge the derived state at publication | `test_task_graph_accept_j.DowngradeForgeryTests.test_invalid_queue_patch_rejects_as_derived_state_mismatch`; also `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[tool_queue]` |

Maps the S5.2 acceptance lanes to the design contract and the public producer,
publication, or restore boundary that rejects each forgery. Gate rejections assert a
`ProjectionError` and the first difference/path; disk damage asserts `CorruptRecordError`.

## c. Authority

| Scenario | Acceptance test | Public mechanism and result |
|---|---|---|
| C1 — model final prose and author text cannot choose completion, transition, or reward | `test_task_graph_accept_authority.AuthorityAcceptanceTests.test_model_and_author_text_cannot_route_or_publish_a_reward`; `test_environment_cannot_publish_a_forged_transition_or_terminal_state` | `RolloutEnvironment.commit` rejects a forged reward step and scripted author reply; `store.publish` rejects an event/state relabelled as a terminal transition. The writer prose still routes to the controller's check step. |
| C2 — evaluator status/evidence forgery | `test_evaluator_status_and_family_are_derived_from_admitted_evidence` | A fresh evaluator result that fails derive verification raises `AdapterContractError` from `commit`; the same typed result persisted and published is `ProjectionError` at its input path, with the head unchanged. |
| C3 — eligibility forgery | `test_eligibility_is_always_ineligible_and_a_forged_true_claim_fails_the_gate` | Full lifecycle records `TrainingEligibilityV1.status = ineligible`; a hash-correct eligible artifact and outcome sibling is rejected at `state.outcome_ref`. |
| C4 — evaluator-family relabelling | `test_evaluator_status_and_family_are_derived_from_admitted_evidence` (`forgery="family"`) | Replaces the admitted deterministic evidence type with another registered evaluator family; fresh port output is `AdapterContractError` at `commit`, while the same persisted input is `ProjectionError` at its input path. |
| Group context-policy input from caller alternatives | `test_foreign_context_policy_from_alternatives_is_projection_error_at_commit`; `test_published_foreign_context_policy_has_same_projection_path` | A foreign `policy_ref` from the caller's `alternatives` callback is `ProjectionError` at `input.policy_ref` on fresh `commit` and persisted `store.publish`; it is not an adapter failure. |
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
| G1 — inline logprob arrays and negative token IDs | `test_adapter_sampling_evidence_rejections_write_no_lineage_records`; `test_malformed_and_unbound_sampling_forgery_is_projection_error_at_publish` | Fresh backend outputs raise `AdapterContractError` before an event/checkpoint/commit is added. Hash-correct disk payloads are rejected by `store.publish` as `ProjectionError` at `event.payload_ref`. |
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

## Findings / known boundaries

- `S5.1-A-OBS-1` is X3, an accepted trusted-adapter limit. The category-a row points to
  `test_X3_fabricated_read_observation_is_an_accepted_trusted_adapter_limit`; no rejection
  test or pending decision remains.
- `S5.1-B-CLASS-1` is fixed: hash-correct extra fields and unknown record types raise
  `ProjectionError` at their payload paths, while a byte changed without recomputing its hash
  remains `CorruptRecordError`.
