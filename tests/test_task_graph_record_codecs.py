from __future__ import annotations

import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

from tests.task_graph_record_fixtures import (
    EXPECTED_NON_EDGE_HASHES,
    EXPECTED_REFS,
    _hash_specs,
    _is_sha256,
    _walk,
    edge_path_matches,
    record_examples,
    shared_payload_examples,
)
from writing_agent.task_graph import (
    canonical_bytes,
)
from writing_agent.task_graph_errors import (
    MaterializationError,
)
from writing_agent.task_graph_record_contracts import (
    CompactionError,
    ContextPolicyV1,
    ExecutionVersionsV1,
    GroupError,
    GroupSpecV1,
)
from writing_agent.task_graph_records import (
    RECORD_EDGES,
    RECORD_TYPES,
    AdmissionPolicyV1,
    ContextContentV1,
    ContextRevisionV1,
    EnvironmentStepV1,
    ExternalInputsV1,
    MemberStartV1,
    OutcomeV1,
    SampledMessageV1,
    ToolObservationV1,
    WriterTurnV1,
    materialize_context_nodes,
    record_reference_edges,
)

H = "a" * 64
P = "b" * 64
Q = "c" * 64
R = "d" * 64


class RecordCodecTests(unittest.TestCase):
    def test_every_registered_refs_mapping_matches_the_wire_contract(self):
        self.assertEqual(
            {name: tuple(sorted(refs.items())) for name, refs in RECORD_EDGES.items()},
            EXPECTED_REFS,
        )

    def test_pre_s1_group_spec_wire_and_identity_are_unchanged(self):
        literal = (Path(__file__).parent / "fixtures" / "pre_s1_group_spec.json").read_bytes()
        record = GroupSpecV1.from_json(literal)
        self.assertEqual(canonical_bytes(record.to_wire()), literal)
        self.assertEqual(
            record.identity(),
            "3e80cde8f3eba7ce6f376d5d5d1e858abdf3f21d88c2cc3e9df26a35ffa46c9e",
        )
        self.assertEqual(record.schema, 1)
        self.assertEqual({member.schema for member in record.members}, {1})

    def test_every_codec_has_a_canonical_exact_round_trip(self):
        examples = record_examples()
        class_names = {
            item.RECORD_TYPE or item.EDGE_TYPE or type(item).__name__ for item in examples
        }
        self.assertEqual(
            class_names,
            set(RECORD_EDGES)
            - set(shared_payload_examples())
            - {"context_node", "context_revision"},
        )
        record_types = [item.RECORD_TYPE for item in examples if item.RECORD_TYPE is not None]
        self.assertEqual(len(record_types), len(set(record_types)))
        self.assertEqual(
            {item.RECORD_TYPE for item in examples if item.RECORD_TYPE is not None}
            | set(shared_payload_examples()),
            set(RECORD_TYPES),
        )
        self.assertNotIn("ExecutionVersionsV1", RECORD_TYPES)
        for record in examples:
            with self.subTest(record=type(record).__name__):
                wire = record.to_wire()
                decoded = type(record).from_json(canonical_bytes(wire))
                self.assertEqual(canonical_bytes(decoded.to_wire()), canonical_bytes(wire))
                self.assertEqual(set(decoded.to_wire()), set(wire))

    def test_every_codec_rejects_floats_and_unknown_or_missing_keys(self):
        for record in record_examples():
            codec = type(record)
            wire = record.to_wire()
            with self.subTest(record=codec.__name__, bad="float"):
                bad = dict(wire)
                key = next(key for key in bad if key != "record_type")
                bad[key] = 1.25
                with self.assertRaises((TypeError, ValueError)):
                    codec.from_dict(bad)
            with self.subTest(record=codec.__name__, bad="unknown"):
                bad = {**wire, "unknown": "extra"}
                with self.assertRaises(ValueError):
                    codec.from_dict(bad)
            with self.subTest(record=codec.__name__, bad="missing"):
                bad = dict(wire)
                bad.pop(next(key for key in bad if key != "record_type"))
                with self.assertRaises(ValueError):
                    codec.from_dict(bad)

    def test_typed_bool_and_integer_fields_reject_python_coercions(self):
        for record in record_examples():
            wire = record.to_wire()
            codec = type(record)
            if isinstance(record, SampledMessageV1):
                wire["tool_calls_was_list"] = 1
                with self.subTest(record=codec.__name__, field="tool_calls_was_list"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, ToolObservationV1):
                wire["dispatch"]["observation"]["ok"] = 1
                with self.subTest(record=codec.__name__, field="observation.ok"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, WriterTurnV1):
                wire["adapter_trace"]["seed"] = True
                with self.subTest(record=codec.__name__, field="seed"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, MemberStartV1):
                wire["ordinal"] = True
                with self.subTest(record=codec.__name__, field="ordinal"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, (ExternalInputsV1, AdmissionPolicyV1, OutcomeV1)):
                wire["schema"] = True
                with self.subTest(record=codec.__name__, field="schema"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)

    def test_writer_usage_token_counts_are_nonnegative_integers(self):
        record = next(item for item in record_examples() if isinstance(item, WriterTurnV1))
        for value in (True, -1, 1.5):
            wire = record.to_wire()
            wire["usage"]["prompt_tokens"] = value
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                WriterTurnV1.from_dict(wire)

    def test_refs_lint_covers_named_and_hash_shaped_values(self):
        for record in record_examples():
            self.assertEqual(
                {field.name for field in fields(record)},
                set(record.FIELD_SPEC.required) | set(record.FIELD_SPEC.optional),
                type(record).__name__,
            )
            self.assertEqual(
                record.REFS,
                RECORD_EDGES.get(record.RECORD_TYPE, record.REFS),
                type(record).__name__,
            )
            for path, edge in _hash_specs(record.FIELD_SPEC):
                if edge is None:
                    self.assertNotIn(path, record.REFS)
                else:
                    self.assertEqual(record.REFS[path], edge)

    def test_non_edge_hashes_are_pinned_and_example_hashes_have_declared_hash_paths(self):
        non_edges = {}
        examples = []
        for record in record_examples():
            spec = record.FIELD_SPEC
            paths = frozenset(path for path, edge in _hash_specs(spec) if edge is None)
            if paths:
                non_edges[record.RECORD_TYPE or type(record).__name__] = paths
            examples.append((type(record).__name__, spec, record.to_wire()))
        for name, body in shared_payload_examples().items():
            codec = RECORD_TYPES[name]
            spec = codec.FIELD_SPEC if isinstance(codec, type) else codec.fields
            non_edges_for_codec = frozenset(
                path for path, edge in _hash_specs(spec) if edge is None
            )
            if non_edges_for_codec:
                non_edges[name] = non_edges_for_codec
            examples.append((name, spec, body))

        self.assertEqual(non_edges, EXPECTED_NON_EDGE_HASHES)
        for name, spec, body in examples:
            declared_paths = tuple(_hash_specs(spec))
            for path, value in _walk(body):
                if not _is_sha256(value):
                    continue
                self.assertTrue(
                    any(edge_path_matches(template, path) for template, _ in declared_paths),
                    f"{name}.{path} contains an undeclared SHA-256 value",
                )

    def test_outcome_wire_fields_remain_pinned(self):
        outcome = next(item for item in record_examples() if isinstance(item, OutcomeV1))
        self.assertEqual(
            set(outcome.to_wire()),
            set(
                """record_type schema task_status execution_status stop_reason reward_status
                training_eligibility candidate_checkpoint requirement_version checks
                transition_edge_id failed_request_ref reward_ref eligibility_ref""".split()
            ),
        )

    def test_group_and_context_policy_binding_rules_reject_drift(self):
        spec = next(item for item in record_examples() if isinstance(item, GroupSpecV1))
        first, second = spec.members
        group_mutations = (
            ("group ID binding", lambda: replace(spec, group_id="e" * 64)),
            ("runner mode participates in group ID", lambda: replace(spec, runner_mode="real")),
            (
                "member ordinals are canonical",
                lambda: replace(spec, members=(second, first)),
            ),
            (
                "writer seed derivation",
                lambda: replace(
                    spec,
                    members=(replace(first, writer_seed=first.writer_seed + 1), second),
                ),
            ),
            (
                "environment seed derivation",
                lambda: replace(
                    spec,
                    members=(replace(first, environment_seed=first.environment_seed + 1), second),
                ),
            ),
            (
                "member ID binds ordinal",
                lambda: replace(spec, members=(replace(first, member_id="grp-other-0"), second)),
            ),
        )
        for label, mutate in group_mutations:
            with self.subTest(rule=label), self.assertRaises(GroupError):
                mutate()
        with patch("writing_agent.task_graph_record_contracts._group_seed", return_value=7):
            collided_seeds = tuple(
                replace(member, writer_seed=7, environment_seed=7) for member in spec.members
            )
            with self.assertRaises(GroupError):
                replace(spec, members=collided_seeds)

        context_mutations = (
            (
                "compact cannot name a seed",
                lambda: ContextPolicyV1(
                    "compact",
                    seed_name="opening",
                    summarizer_version="visible-text-v1",
                    max_summary_chars=0,
                ),
                CompactionError,
            ),
            (
                "only compact retains a tail",
                lambda: ContextPolicyV1("carry", retained_exchanges=1),
                CompactionError,
            ),
            (
                "seed name contains no whitespace",
                lambda: ContextPolicyV1(
                    "seed", seed_name="opening scene", seed_checkpoint_ref="a" * 64
                ),
                ValueError,
            ),
            (
                "only seed names a prefix",
                lambda: ContextPolicyV1("carry", seed_name="opening", seed_checkpoint_ref="a" * 64),
                CompactionError,
            ),
        )
        for label, mutate, error in context_mutations:
            with self.subTest(rule=label), self.assertRaises(error):
                mutate()

    def test_seed_context_policy_requires_its_checkpoint_at_decode(self):
        body = ContextPolicyV1("carry").to_wire()
        body.update(operation="seed", seed_name="opening")
        with self.assertRaises(CompactionError):
            ContextPolicyV1.from_dict(body)

    def test_unknown_adapter_reference_claims_are_rejected(self):
        record = next(item for item in record_examples() if isinstance(item, WriterTurnV1))
        wire = record.to_wire()
        wire["adapter_trace"]["unregistered_policy_ref"] = H
        with self.assertRaises(ValueError):
            WriterTurnV1.from_dict(wire)

    def test_registered_shared_shapes_and_untagged_execution_versions_are_typed(self):
        expected = {
            "GroupScriptedTerminalV1": (("checkpoint", P),),
            "GroupExecutionFailureV1": (("checkpoint", P),),
            "GroupMemberResultV1": (("checkpoint", P),),
            "GroupDecisionV1": (),
            "GroupAdvantageV1": (("artifact", P),),
            "GroupSegmentCreditV1": (
                ("event", P),
                ("artifact", Q),
                ("artifact", R),
                ("context_revision", H),
                ("artifact", R),
            ),
            "RuntimePortDescriptorV1": (),
            "RuntimeManifestV1": (),
            "DeterministicCheckEvidenceV1": (("checkpoint", H), ("private", R)),
            "FixtureFileCountEvidenceV1": (("checkpoint", H), ("private", R)),
            "TranscriptReviewEvidenceV1": (("checkpoint", H), ("private", R)),
            "DecisionLedgerV1": (),
            "DisclosureLedgerV1": (),
            "AuthorRequestV1": (
                ("artifact", H),
                ("artifact|private", P),
                ("private", Q),
                ("private", R),
            ),
            "CheckRequestV1": (
                ("checkpoint", H),
                ("private", P),
                ("private", Q),
                ("private", R),
            ),
            "RewardV1": (
                ("artifact", H),
                ("private", P),
                ("checkpoint", Q),
                ("artifact", R),
                ("artifact", H),
            ),
            "TrainingEligibilityV1": (("artifact", H),),
            "GroupMemberSeedsV1": (("artifact", P),),
            "RequirementLedgerV1": (),
            "ExecutionVersionsV1": (("artifact", H),),
        }
        for record_type, body in shared_payload_examples().items():
            with self.subTest(record_type=record_type):
                self.assertEqual(record_reference_edges(record_type, body), expected[record_type])
                with self.assertRaises((TypeError, ValueError)):
                    record_reference_edges(record_type, {**body, "unknown": "field"})
                missing = dict(body)
                missing.pop(next(iter(missing)))
                with self.assertRaises((TypeError, ValueError)):
                    record_reference_edges(record_type, missing)
        self.assertIn(("private", Q), expected["CheckRequestV1"])
        versions = next(item for item in record_examples() if isinstance(item, ExecutionVersionsV1))
        self.assertEqual(
            record_reference_edges("ExecutionVersionsV1", versions.to_wire()),
            expected["ExecutionVersionsV1"],
        )
        self.assertNotIn("ExecutionVersionsV1", RECORD_TYPES)

    def test_admission_policy_wire_is_sorted_and_unique(self):
        wire = AdmissionPolicyV1(
            schema=1,
            writer_family=None,
            allowed_tools=("read_file", "write_file"),
            controller_versions=("deterministic-v1",),
            check_versions=("fixture-v1",),
        ).to_wire()
        wire["allowed_tools"] = list(reversed(wire["allowed_tools"]))
        with self.assertRaises(ValueError):
            AdmissionPolicyV1.from_dict(wire)

    def test_environment_directive_union_uses_one_exact_shape_per_kind(self):
        directives = (
            {"kind": "request_author", "source": "mandatory_feedback"},
            {"kind": "request_checks"},
            {"kind": "commit_transition", "edge_id": "next"},
            {"kind": "seal_outcome", "task_status": "complete", "stop_reason": None},
            {"kind": "stop_exhausted", "stop_reason": "writer_budget"},
            {"kind": "publish_reward"},
        )
        for directive in directives:
            with self.subTest(kind=directive["kind"]):
                record = EnvironmentStepV1(directive)
                self.assertEqual(EnvironmentStepV1.from_json(record.to_json()), record)
                invalid = record.to_wire()
                invalid["directive"]["extra"] = "not on this directive"
                with self.assertRaises(ValueError):
                    EnvironmentStepV1.from_dict(invalid)
        for directive in (
            {"kind": "seal_outcome", "task_status": "complete", "stop_reason": ""},
            {"kind": "stop_exhausted", "stop_reason": ""},
        ):
            with self.subTest(directive=directive), self.assertRaises(ValueError):
                EnvironmentStepV1(directive)

    def test_tool_observation_rejects_noop_effects_and_nonpositive_limits(self):
        base = {
            "spec": {"max_file_bytes": 10, "max_workspace_bytes": 20},
            "observation": {"ok": True, "valid": True},
            "effect": {"draft.txt": {"before": "same", "after": "same"}},
        }
        with self.assertRaises(ValueError):
            ToolObservationV1("call-1", base)
        base["effect"] = {}
        base["spec"]["max_file_bytes"] = 0
        with self.assertRaises(ValueError):
            ToolObservationV1("call-1", base)

    def test_materializer_classifies_a_missing_root_invariant(self):
        broken_root = object.__new__(ContextContentV1)
        for name, value in {
            "parent_ref": None,
            "messages": (),
            "tools": None,
            "rendering": None,
        }.items():
            object.__setattr__(broken_root, name, value)
        revision = ContextRevisionV1(H, None, ())
        with self.assertRaises(MaterializationError):
            materialize_context_nodes(revision, {H: broken_root})

    def test_open_provider_maps_remain_ordinary_canonical_json(self):
        record = next(item for item in record_examples() if isinstance(item, WriterTurnV1))
        wire = record.to_wire()
        detail = {"$noncanonical": "provider metadata, not sampled intake"}
        wire["usage"]["vendor_detail"] = detail
        wire["adapter_trace"]["provider_detail"] = detail
        self.assertEqual(WriterTurnV1.from_dict(wire).to_wire(), wire)
