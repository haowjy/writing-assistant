# Transition-seam acceptance port: categories a, b and j

Test IDs below use the `unittest` module/class/method name. A bracketed value is the
corresponding `subTest` case. The acceptance modules use only the three public attack
surfaces: `RolloutEnvironment.commit`, `TaskGraphStore.publish`, or on-disk rewrites followed
by `open_head`/`verify`. They do not call a derive or validator directly.

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
| `test_task_graph_writer.py:479-605`, `test_action_and_sampled_trace_cross_binding_matrix` | Sampling/request/trace evidence is carried in the typed writer input; binding-specific claims are category g | `test_task_graph_accept_a.WriterAndToolInputForgeryTests.test_writer_turn_inputs_reject_active_identity_and_sampling_binding_forgeries` `[request_ref, prepared_request_ref, action_id, context_revision_ref]`; group/sampling claim rows are in S5.2 |
| `test_task_graph_writer.py:606-670`, action authority and trace restore probes | Outcome and trace claims are no longer effect patches; forged outcome state or event identity is compared to the writer-turn derive | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[outcome_ref]`; `test_derived_event_forgery_matrix_rejects_at_first_event_path` `[actor]` |
| `test_task_graph_writer.py:1126-1211`, forged result origin/hash/charge and independent result fields | Action/result IDs, execution hashes, charge and `file_delta` are derived, not input fields | `test_task_graph_accept_a.WriterAndToolInputForgeryTests.test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[call_id, fake_delta, false_success_with_delta]`; derived counters/tree/history are covered by `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[budgets_ref, files, tool_result_history]` |
| `test_task_graph_writer.py:1212-1257`, all seven tool-result forgeries before head publication | Same typed tool-input and derived-state boundary | `test_task_graph_accept_a.WriterAndToolInputForgeryTests.test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[call_id, fake_delta, false_success_with_delta]`; `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[cursor, budgets_ref, tool_result_history]` |
| `test_task_graph_writer.py:1258-1285`, context event cannot hide file change | No generic context patch can author a file delta | `test_task_graph_accept_j.DowngradeForgeryTests.test_invalid_queue_patch_rejects_as_derived_state_mismatch` and `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[files]` |
| `test_task_graph_scripted.py:812` (`validate_author_request_effect`, forged actor) | Event actor is derived from the accepted input kind | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_event_forgery_matrix_rejects_at_first_event_path` `[actor]` |
| `test_task_graph_scripted.py:1033, 1042, 1051` (`validate_check_result_effect`: request/check/packet/evidence/status, decisions ref, actor) | Target checkpoint, check contract and packet bindings are fields of the derived private check request, not `EvaluatorResultV1`; mutate request/evidence/status input bindings instead | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_author_evaluator_and_terminal_inputs_reject_semantic_mismatches` `[evaluator_result, index=0..2]`; event actor and derived decisions are covered by `test_derived_event_forgery_matrix_rejects_at_first_event_path` `[actor]` and `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[decisions_ref]` |
| `test_task_graph_scripted.py:1120, 1129` (`validate_terminal_effect`: reward/status and actor) | Reward/status fields are now a derived `OutcomeV1`; transition choice is a directive-bound environment input | `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[outcome_ref]`; `test_author_evaluator_and_terminal_inputs_reject_semantic_mismatches` `[terminal_step]`; event actor is `[actor]` |
| `test_task_graph_writer.py` `EnvironmentBatch.append_record` patch probes (`owned action`, action/result log, authority/trace, result origin/hash/charge, independent/all-seven tool-result families) | Patches are removed; each semantic family is submitted as a typed input or a forged published result | `test_writer_turn_inputs_reject_active_identity_and_sampling_binding_forgeries`; `test_tool_observation_inputs_reject_queue_and_effect_forgeries`; `test_derived_state_forgery_matrix_rejects_at_first_state_path`; `test_derived_event_forgery_matrix_rejects_at_first_event_path` (IDs above) |

### The 26-case `judo-forgery-matrix.py`

| Old matrix row | New result / test ID |
|---|---|
| `record.action_id`, `record.result_id` | Not input fields on `ToolObservationV1`; result identity is derived. Forged output is checked by `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[tool_result_history]` and the event/state candidate rows. |
| `record.call_id` | `test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[call_id]` |
| `record.observation` | `test_task_graph_accept_a.WriterAndToolInputForgeryTests.test_forged_read_observation_is_rejected` — **expected failure**, finding `S5.1-A-OBS-1`; a changed read result is currently accepted and publishes. |
| `record.file_delta(fake path)`, `delta.forged(write)` | `test_tool_observation_inputs_reject_queue_and_effect_forgeries` `[fake_delta]` |
| `record.before_execution_hash`, `record.after_execution_hash` | Not wire fields; the producer derives the file tree and event. Test forged state `[files]` and tool effect `[fake_delta]`. |
| `record.budget_charge.tool_calls`, `record.budget_charge.read_tokens` | Not wire fields; test derived counters `[budgets_ref]`. |
| `record.loss_eligibility`, `message.loss_eligible`, `message.content`, `message.origin` | Derived context/message values, not tool input fields; test state context identity `[context_ref]` (category b) and event actor / event identity `[actor]`. |
| `record.extra_field`, `record.record_type` | Strict input records have no extensible patch payload. Recomputed-hash disk rewrites are tested by `test_task_graph_accept_b.PersistedForgeTests.test_valid_hash_input_payload_forgery_is_projection_rejection` `[extra_key, unknown_record_type]` — **expected failure**, finding `S5.1-B-CLASS-1` (wrong error class; see category b). |
| `changes.next_call=0`, `changes.tool_queue=[]`, `changes.phase=checking` | No generic state patch is accepted; `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[cursor, tool_queue, phase]`. |
| `history.tool_result_ids(extra)`, `history.action_ids(added key)` | Derived history mismatch: `test_derived_state_forgery_matrix_rejects_at_first_state_path` `[tool_result_history, action_history]`. |
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
| On-disk bytes changed without recomputing identity | Artifact bytes no longer match their stored hash | `test_task_graph_accept_b.PersistedForgeTests.test_byte_tamper_after_warm_view_is_corrupt_and_does_not_move_head` (`CorruptRecordError`, authority ref bytes unchanged) |
| Forged pre-state phase/status | Parentless root must be a fixed point of entry derivation | `test_task_graph_accept_b.PersistedForgeTests.test_forged_parentless_entry_state_fails_the_root_fixed_point` `[pre_phase, pre_status]` |
| `position.lineage_id` changed together with a forged writer event | A different lineage id is valid only when the input derives that lineage start | `test_task_graph_accept_b.PersistedForgeTests.test_relineaged_writer_event_is_not_a_member_start` (`event.lineage_id`) |
| On-disk payload body changed and hash/name recomputed | A forged typed record must be a `ProjectionError`, not store corruption | `test_valid_hash_input_payload_forgery_is_projection_rejection` `[extra_key, unknown_record_type]` — **expected failure**, finding `S5.1-B-CLASS-1` |

`S5.1-B-CLASS-1` minimal reproduction: start from `build_rollout_fixture`, commit its
sampled `WriterTurnV1`, add an unknown key to that stored input body, compute a new
`domain_hash("payload", body)`, rewrite the canonical artifact envelope, then rewrite the
event/checkpoint/commit/head with matching identities and call `RolloutEnvironment.open_head`.
The semantic forgery is rejected, but current closure handling raises `CorruptRecordError`
(`WrongRecordDomainError` for the unregistered record type) rather than the required
`ProjectionError`. No source fix was made because the classification boundary is in the
store, outside this acceptance-test lane.

## j. Generic-reducer downgrade

| Old source | New route | New test ID |
|---|---|---|
| H2, `integrity-forgery.py --downgrade-only`: `seed_attached` or retyped generic event on an admitted lineage | `seed_attached` is not a new-core event/input kind; a well-formed but unsupported event kind is rejected by the gate | `test_task_graph_accept_j.DowngradeForgeryTests.test_unsupported_generic_input_kind_has_no_commit_route`; `test_retyped_admitted_event_rejects_at_event_kind` |
| H3, `judo-forgery-matrix.py`: empty/invalid queue patch through generic effect | No generic reducer or queue-patch input exists; forge the derived state at publication | `test_task_graph_accept_j.DowngradeForgeryTests.test_invalid_queue_patch_rejects_as_derived_state_mismatch`; also `test_task_graph_accept_a.PublishedEventAndStateForgeryTests.test_derived_state_forgery_matrix_rejects_at_first_state_path` `[tool_queue]` |

## Findings / known boundaries

- `S5.1-A-OBS-1`: a `ToolObservationV1` with a forged `read_file` result and unchanged file
  effect currently commits. It is retained as an expected failure, not described as a
  rejected forgery. This is the adapter-attested observation boundary; the current design
  inventory calls fabricated read observations accepted (X3), but the old category-a matrix
  row conflicts with that limit and needs an explicit owner decision.
- `S5.1-B-CLASS-1`: valid-hash typed payload forgeries reject but are classified as store
  corruption. The expected-failure test preserves the required `ProjectionError` contract.
