"""Explicit, deterministic regeneration for the task-graph wire goldens."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.task_graph_fixtures import make_entry_fixture
from tests.task_graph_rollout_fixtures import build_rollout_fixture
from tests.test_task_graph import H as HASH_FIXTURE_REF
from tests.test_task_graph import TaskGraphRecordsTest
from tests.test_task_graph_records import record_examples, shared_payload_examples
from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    EventV1,
    GraphInstanceV1,
    MessageV1,
    NodeSpecV1,
    canonical_bytes,
    canonical_json,
    domain_hash,
    domain_hash_bytes,
    file_hash,
    tree_hash,
)
from writing_agent.task_graph_contracts import BudgetContractV1
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_record_contracts import SEMANTICS_V1, ExecutionVersionsV1
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    ContextContentV1,
    EnvironmentStepV1,
    OutcomeV1,
    SampledMessageV1,
    WriterTurnV1,
)
from writing_agent.task_graph_records import (
    ContextRevisionV1 as WireContextRevisionV1,
)

FIXTURES = Path(__file__).parent / "fixtures"
BYTE_GOLDEN = b"golden\x00bytes"
POST_GOLDEN_RECORD_TYPES = frozenset(
    {
        "WriterTurnV2",
        "RendererDescriptorV1",
        "TokenizerDescriptorV1",
        "DecodingDescriptorV1",
        "RuntimeManifestV2",
        "TrainingAdmissionV1",
    }
)
ZERO_HASH = "0" * 64
HASH_REF = HASH_FIXTURE_REF


def _entry(record) -> dict[str, object]:
    body = record.to_wire() if hasattr(record, "to_wire") else record.to_dict()
    return {"body": body, "identity": record.identity()}


def _payload_entry(body: dict[str, object]) -> dict[str, object]:
    return {"body": body, "identity": domain_hash("payload", body)}


def _message(role: str, text: str, origin: str) -> MessageV1:
    return MessageV1(role=role, content=(text,), origin=origin)


def _context_records() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rendering = {
        "projection_version": "v1",
        "prefix_id": "golden-root",
        "template_ref": ZERO_HASH,
        "tokenizer_ref": "1" * 64,
        "tool_schema_ref": "2" * 64,
    }
    root = ContextContentV1(
        parent_ref=None,
        messages=(
            _message("system", "Preserve the author's choices.", "system:1"),
            _message("user", "Write a quiet arrival.", "request:1"),
        ),
        tools=({"name": "read_file", "parameters": {"type": "object"}},),
        rendering=rendering,
    )
    child = ContextContentV1(
        parent_ref=root.identity(),
        messages=(_message("assistant", "Rain softened the station lights.", "golden:action:0"),),
        tools=None,
        rendering=None,
    )
    revision = WireContextRevisionV1(
        content_ref=child.identity(),
        event_head="3" * 64,
        provenance_refs=("3" * 64,),
    )
    writer_turn = WriterTurnV1(
        action_id="golden-rollout:action:0",
        context_revision_ref=revision.identity(),
        raw_output_ref=None,
        usage={"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
        adapter_trace={
            "model": "fixture-model",
            "seed": 17,
            "generated_token_ids": [41, 42, 43],
            "context_revision_ref": revision.identity(),
            "context_content_hash": child.identity(),
            "rendering": rendering,
        },
        message=SampledMessageV1(content="A measured opening.", tool_calls_was_list=True, calls=()),
    )
    return (
        {"root": _entry(root), "child": _entry(child)},
        _entry(revision),
        _entry(writer_turn),
    )


def _environment_steps() -> dict[str, dict[str, object]]:
    directives = {
        "request_author": {"kind": "request_author", "source": "writer_request"},
        "request_checks": {"kind": "request_checks"},
        "commit_transition": {"kind": "commit_transition", "edge_id": "complete"},
        "seal_outcome": {
            "kind": "seal_outcome",
            "task_status": "complete",
            "stop_reason": None,
        },
        "stop_exhausted": {"kind": "stop_exhausted", "stop_reason": "writer_budget"},
        "publish_reward": {"kind": "publish_reward"},
    }
    return {
        kind: _entry(EnvironmentStepV1(directive=directive))
        for kind, directive in directives.items()
    }


def _outcome_stages() -> dict[str, dict[str, object]]:
    def outcome(
        *,
        task_status="unknown",
        execution_status="running",
        stop_reason=None,
        reward_status="pending",
        training_eligibility="pending",
        candidate_checkpoint=None,
        requirement_version=None,
        checks=(),
        transition_edge_id=None,
        reward_ref=None,
        eligibility_ref=None,
    ):
        return OutcomeV1(
            schema=1,
            task_status=task_status,
            execution_status=execution_status,
            stop_reason=stop_reason,
            reward_status=reward_status,
            training_eligibility=training_eligibility,
            candidate_checkpoint=candidate_checkpoint,
            requirement_version=requirement_version,
            checks=checks,
            transition_edge_id=transition_edge_id,
            failed_request_ref=None,
            reward_ref=reward_ref,
            eligibility_ref=eligibility_ref,
        )

    requested_checks = ({"request_ref": "4" * 64, "result_ref": None},)
    completed_checks = ({"request_ref": "4" * 64, "result_ref": "5" * 64},)
    return {
        "entry": _entry(outcome()),
        "checks_requested": _entry(
            outcome(
                candidate_checkpoint="6" * 64, requirement_version="7" * 64, checks=requested_checks
            )
        ),
        "checks_completed": _entry(
            outcome(
                candidate_checkpoint="6" * 64, requirement_version="7" * 64, checks=completed_checks
            )
        ),
        "transitioned": _entry(
            outcome(
                task_status="complete",
                candidate_checkpoint="6" * 64,
                requirement_version="7" * 64,
                checks=completed_checks,
                transition_edge_id="complete",
            )
        ),
        "sealed": _entry(
            outcome(
                task_status="complete",
                execution_status="valid",
                stop_reason="complete",
                candidate_checkpoint="6" * 64,
                requirement_version="7" * 64,
                checks=completed_checks,
                transition_edge_id="complete",
            )
        ),
        "rewarded": _entry(
            outcome(
                task_status="complete",
                execution_status="valid",
                stop_reason="complete",
                reward_status="available",
                training_eligibility="eligible",
                candidate_checkpoint="6" * 64,
                requirement_version="7" * 64,
                checks=completed_checks,
                transition_edge_id="complete",
                reward_ref="8" * 64,
                eligibility_ref="9" * 64,
            )
        ),
    }


def build_records_golden() -> dict[str, object]:
    records: dict[str, object] = {}
    for record in record_examples():
        record_type = record.RECORD_TYPE
        if record_type in POST_GOLDEN_RECORD_TYPES:
            continue
        if record_type is not None:
            records[record_type] = _entry(record)
        elif isinstance(record, ExecutionVersionsV1):
            records["ExecutionVersionsV1"] = _payload_entry(record.to_wire())
    for record_type, body in shared_payload_examples().items():
        records[record_type] = _payload_entry(body)

    contents, revision, writer_turn = _context_records()
    records["ContextContentV1"] = contents
    records["ContextRevisionV1"] = revision
    records["WriterTurnV1"] = writer_turn
    records["EnvironmentStepV1"] = {"directives": _environment_steps()}
    records["OutcomeV1"] = {"lifecycle": _outcome_stages()}
    records["BudgetContractV1"] = {
        "without_max_generated_tokens": _entry(
            BudgetContractV1(
                max_steps=8,
                max_tool_calls=4,
                max_read_tokens=512,
                max_total_bytes=8192,
                max_author_calls=2,
                max_graph_hops=6,
                max_visits=3,
            )
        ),
        "with_max_generated_tokens": _entry(
            BudgetContractV1(
                max_steps=8,
                max_tool_calls=4,
                max_read_tokens=512,
                max_total_bytes=8192,
                max_author_calls=2,
                max_graph_hops=6,
                max_visits=3,
                max_generated_tokens=1024,
            )
        ),
    }
    state = make_entry_fixture().state
    history = {
        **state.history,
        "head": "a" * 64,
        "seq": 4,
        "action_count": 2,
        "tool_result_count": 2,
    }
    state_after_two_actions = replace(state, history=history)
    records["EnvironmentStateV1"] = _entry(state_after_two_actions)

    if set(records) | POST_GOLDEN_RECORD_TYPES != set(RECORD_TYPES) | {
        "BudgetContractV1",
        "ContextContentV1",
        "ContextRevisionV1",
        "ExecutionVersionsV1",
        "EnvironmentStateV1",
    }:
        raise AssertionError("task-graph golden examples do not cover the current wire records")
    return {"transition_semantics": SEMANTICS_V1, "records": records}


def build_records_v2_golden() -> dict[str, object]:
    """Pin the additive V2 wire records independently from the V1 goldens."""
    records = {
        record.RECORD_TYPE: _entry(record)
        for record in record_examples()
        if record.RECORD_TYPE in POST_GOLDEN_RECORD_TYPES
    }
    if set(records) != POST_GOLDEN_RECORD_TYPES:
        raise AssertionError("V2 golden examples do not cover all additive records")
    return {"transition_semantics": SEMANTICS_V1, "records": records}


def _rollout_samples() -> tuple[SampleResult, ...]:
    def call(name: str, arguments: dict[str, object], call_id: str) -> dict[str, object]:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    return (
        SampleResult(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call(
                        "write_file",
                        {"path": "draft.txt", "content": "A blue door opens."},
                        "write-1",
                    )
                ],
            }
        ),
        SampleResult(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("read_file", {"path": "draft.txt"}, "read-1")],
            }
        ),
        SampleResult(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call(
                        "ask_author",
                        {
                            "question": "Which color should the door be?",
                            "decision_ids": ["door"],
                            "proposals": [],
                            "option_refs": [],
                        },
                        "ask-1",
                    )
                ],
            }
        ),
        SampleResult(
            {"role": "assistant", "content": "The revision is ready for review.", "tool_calls": []}
        ),
    )


def build_rollout_golden(root: Path) -> dict[str, object]:
    fixture = build_rollout_fixture(root, mode="slice", sample_results=_rollout_samples())
    event_ids: list[dict[str, str]] = []
    checkpoint_ids = [fixture.runtime.checkpoint_id]
    commit_ids: list[str] = []

    def observe(result) -> None:
        event = fixture.store.load_event(result.event_id)
        event_ids.append({"kind": event.kind, "id": result.event_id})
        checkpoint_ids.append(result.runtime.checkpoint_id)
        commit_ids.append(result.commit_id)

    fixture.env.commit_observer = observe
    result = fixture.driver().run(fixture.runtime, max_steps=40)
    return {
        "transition_semantics": SEMANTICS_V1,
        "route": [
            "entry",
            "write_file",
            "read_file",
            "ask_author",
            "reply",
            "final",
            "checks",
            "seal",
            "reward",
        ],
        "events": event_ids,
        "checkpoint_ids": checkpoint_ids,
        "commit_ids": commit_ids,
        "final_directive": result.directive.kind,
        "final_state_identity": result.runtime.state.identity(),
    }


def build_hash_golden() -> dict[str, object]:
    test_records = TaskGraphRecordsTest()
    message = MessageV1(content=("hello",), origin="a")
    rendering = {
        "projection_version": "v1",
        "prefix_id": "root",
        "template_ref": HASH_REF,
        "tokenizer_ref": HASH_REF,
        "tool_schema_ref": HASH_REF,
    }
    context_content = ContextContentV1(None, (message,), (), rendering)
    context = WireContextRevisionV1(context_content.identity(), None, ())
    commit = CommitV1(parent_commit=None, events=(HASH_REF,), checkpoint=HASH_REF)
    instance = GraphInstanceV1(
        template_ref=HASH_REF,
        entry_node="n",
        nodes=(NodeSpecV1(id="n", entry_contract=HASH_REF),),
    )
    state = test_records.state()
    checkpoint = CheckpointV1(state=state)
    event = EventV1(
        lineage_id="l",
        kind="writer_action",
        audience=("writer",),
        payload_ref=HASH_REF,
        versions_ref=HASH_REF,
        provenance_ref=HASH_REF,
    )
    bytes_envelope = {
        "schema": 1,
        "domain": "payload:bytes",
        "encoding": "base64",
        "body": "Z29sZGVuAGJ5dGVz",
    }
    return {
        "transition_semantics": SEMANTICS_V1,
        "canonical": canonical_json({"b": "é", "a": [1, True, None]}),
        "file": file_hash("x"),
        "tree": tree_hash({"a.txt": "hi"}),
        "message": message.identity(),
        "context_content": context_content.identity(),
        "context": context.identity(),
        "state": state.identity(),
        "checkpoint": checkpoint.identity(),
        "commit": commit.identity(),
        "event": event.identity(),
        "instance": instance.identity(),
        "bytes": {
            "value_hex": BYTE_GOLDEN.hex(),
            "identity": domain_hash_bytes("payload", BYTE_GOLDEN),
            "envelope": bytes_envelope,
            "canonical_envelope": canonical_bytes(bytes_envelope).decode("utf-8"),
        },
    }


def _write_json(path: Path, body: dict[str, object]) -> None:
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def regenerate_goldens() -> None:
    """Rewrite the fixtures explicitly; never called by tests."""
    _write_json(FIXTURES / "task_graph_hashes.json", build_hash_golden())
    _write_json(FIXTURES / "task_graph_records_golden.json", build_records_golden())
    _write_json(FIXTURES / "task_graph_records_v2_golden.json", build_records_v2_golden())
    with TemporaryDirectory() as root:
        rollout = build_rollout_golden(Path(root) / "rollout")
    _write_json(FIXTURES / "task_graph_rollout_golden.json", rollout)


if __name__ == "__main__":
    regenerate_goldens()
