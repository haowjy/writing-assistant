"""The admitted Phase 8 probe tasks and their shared training composition.

This is the single owner of the P1 recipe and deterministic task construction.  It
is intentionally importable without Torch or Transformers; those dependencies are
loaded only when a run or model tokenizer is requested.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from writing_agent.grpo import GRPOSettings
from writing_agent.grpo_task_graph import TaskGraphTaskV1, train_task_graph
from writing_agent.legacy_graph import compile_legacy_scenario
from writing_agent.task_graph import (
    CheckpointV1,
    MaterializedContextV1,
    domain_hash,
    load_canonical_json,
)
from writing_agent.task_graph_admission import (
    AdmittedGraphV1,
    MappingArtifactResolver,
    admit_graph,
)
from writing_agent.task_graph_contracts import (
    AuthorPacketV1,
    CheckContractV1,
    DecisionBindingsV1,
    EvaluatorPacketV1,
    InteractionContractV1,
    InteractionPolicyV1,
    RewardContractV1,
    ScriptedAuthorV1,
)
from writing_agent.task_graph_derive_entry import EntryParamsV1, EntryV1, derive_entry
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_group_contract import derive_group_seed
from writing_agent.task_graph_records import (
    AdmissionPolicyV1,
    ContextContentV1,
    ContextRevisionV1,
    materialize_context_nodes,
)
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_transition import DerivedArtifact

TOKENIZER_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
AUTHOR_PACKET_CANARY = "P8R3C_PRIVATE_AUTHOR_PREF_CANARY_4172"
EVALUATOR_PACKET_CANARY = "P8R3C_PRIVATE_EVALUATOR_SPEC_CANARY_8365"
TOKENIZER_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / TOKENIZER_REVISION
)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "phase8" / "probe-tasks"


def settings() -> GRPOSettings:
    """Return the identity-bound probe recipe."""
    return GRPOSettings(
        model_id="google/gemma-4-E2B-it",
        revision=TOKENIZER_REVISION,
        runtime_profile="task-graph-v1",
        loss_type="dapo",
        enable_thinking=False,
        group_size=4,
        microbatch_size=1,
        max_steps=3,
        context_tokens=4096,
        max_tokens=512,
        max_generated_tokens=1536,
        seed=123,
        gradient_checkpointing=True,
        gradient_checkpointing_use_reentrant=False,
    )


def tiny_gemma(vocab_size: int):
    """Create the small CPU Gemma used by the committed sampler composition check."""
    import torch
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

    torch.manual_seed(123)
    text_config = Gemma4TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=1,
        max_position_embeddings=4096,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=16,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=vocab_size,
        num_kv_shared_layers=2,
        attention_k_eq_v=True,
        enable_moe_block=False,
        final_logit_softcapping=30.0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        eos_token_id=None,
        pad_token_id=0,
    )
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(
            text_config=text_config, tie_word_embeddings=False, attn_implementation="eager"
        )
    ).to(dtype=torch.float32, device="cpu")
    model.eval()
    return model


def _scenario() -> dict[str, Any]:
    """Minimal common entry used to derive admitted probe tasks."""
    return {
        "id": "transition-entry",
        "family": "F2",
        "role": "development",
        "condition": "workspace",
        "source_groups": ["synthetic"],
        "provenance": "synthetic",
        "visible": {
            "brief": "Revise the text files.",
            "initial_files": {
                "draft.txt": "alpha\n",
                "notes/source.md": "snow\nmoon\n",
            },
            "followups": [],
            "tools": ["list_dir", "read_file", "search", "write_file", "patch_file"],
            "budgets": {
                "max_steps": 5,
                "max_tool_calls": 10,
                "max_read_tokens": 100,
                "max_total_bytes": 4096,
            },
            "prose": [],
        },
        "labels": {
            "rubric_version": 1,
            "checks": [],
            "rubrics": {},
            "knowledge": [],
            "source_cutoff": "supplied files only",
        },
    }


class MemoryArtifactReader:
    """Hash-indexed in-memory artifact view shared by entry derivations."""

    def __init__(self, public: dict[str, Any], private: dict[str, Any] | None = None) -> None:
        self.public = dict(public)
        self.private = dict(private or {})
        self.byte_values: dict[str, bytes] = {}
        self.checkpoints: dict[str, CheckpointV1] = {}
        self.context_revisions: dict[str, Any] = {}
        self.context_nodes: dict[str, Any] = {}
        self.events: dict[str, Any] = {}

    def add(self, value: Any, *, private: bool = False) -> str:
        identity = domain_hash("payload", value)
        (self.private if private else self.public)[identity] = value
        return identity

    def artifact(self, ref: str, *, domain: str = "payload", private: bool = False) -> Any:
        if domain == "payload":
            value = (self.private if private else self.public)[ref]
        elif domain == "context_revision":
            value = self.context_revisions[ref]
        elif domain == "context_node":
            value = self.context_nodes[ref]
        elif domain == "event":
            value = self.events[ref]
        else:
            raise KeyError((domain, ref))
        return _copy(value)

    def context(self, ref: str) -> MaterializedContextV1:
        revision = ContextRevisionV1.from_dict(self.artifact(ref, domain="context_revision"))
        nodes = {
            identity: ContextContentV1.from_dict(_copy(value))
            for identity, value in self.context_nodes.items()
        }
        return materialize_context_nodes(revision, nodes)

    def bytes_artifact(self, ref: str) -> bytes:
        return self.byte_values[ref]

    def checkpoint(self, ref: str) -> CheckpointV1:
        return self.checkpoints[ref]


def _copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy(item) for item in value]
    return value


def _artifact_body(value: Any) -> Any:
    if isinstance(value, bytes):
        return load_canonical_json(value)
    if hasattr(value, "to_wire"):
        return value.to_wire()
    return _copy(value)


@dataclass(frozen=True)
class EntryFixture:
    graph: AdmittedGraphV1
    node_id: str
    params: EntryParamsV1
    reader: MemoryArtifactReader
    state: Any
    artifacts: tuple[DerivedArtifact, ...]


def make_entry_fixture(*, rendering_overrides=None, public_records=()) -> EntryFixture:
    """Build the deterministic common entry and its derived artifacts."""
    bundle = compile_legacy_scenario(_scenario())
    graph = bundle.admission()
    reader = MemoryArtifactReader(
        {identity: _copy(body) for identity, body in bundle.public_artifacts.items()},
        {identity: _copy(body) for identity, body in bundle.private_artifacts.items()},
    )
    for record in public_records:
        reader.public[record.identity()] = record.to_wire()
    rendering = {
        "projection_version": "v1",
        "prefix_id": "root",
        "template_ref": reader.add({"pin": "template"}),
        "tokenizer_ref": reader.add({"pin": "tokenizer"}),
        "tool_schema_ref": reader.add({"pin": "tools"}),
    }
    rendering.update(rendering_overrides or {})
    admission_policy = AdmissionPolicyV1.from_admission_policy(graph.policy)
    admission_policy_ref = reader.add(admission_policy.to_wire())
    versions_ref = reader.add(
        {
            "schema": 1,
            "transition_semantics": "task-graph-derive-v1",
            "admission_policy_ref": admission_policy_ref,
            "tool_spec": {"max_file_bytes": 128_000, "max_workspace_bytes": 4096},
        }
    )
    params = EntryParamsV1(
        lineage_id="rollout-fixture",
        visit_id="visit-fixture",
        rendering=rendering,
        versions_ref=versions_ref,
        provenance_ref=reader.add({"fixture": "provenance"}),
        rng_ref=reader.add({"fixture": "rng", "seed": 7}),
    )
    node_id = graph.instance.entry_node
    derived = derive_entry(graph, node_id, params, reader)
    for artifact in derived.artifacts:
        if artifact.kind in {"artifact", "private"}:
            target = reader.private if artifact.kind == "private" else reader.public
            target[artifact.ref] = _artifact_body(artifact.value)
        elif artifact.kind == "context_revision":
            reader.context_revisions[artifact.ref] = _artifact_body(artifact.value)
        elif artifact.kind == "context_node":
            reader.context_nodes[artifact.ref] = _artifact_body(artifact.value)
    return EntryFixture(graph, node_id, params, reader, derived.state, derived.artifacts)


def load_probe_task(path: Path) -> dict[str, Any]:
    """Load and validate one schema-1 declarative probe task."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("probe task config must be a schema 1 object")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError("probe task config needs an id")
    return value


def load_probe_tasks() -> tuple[dict[str, Any], ...]:
    """Load the fixed public task set in stable filename order."""
    return tuple(load_probe_task(path) for path in sorted(CONFIG_DIR.glob("t*.json")))


def build_admitted_entry(config: dict[str, Any]) -> EntryFixture:
    """Build an admitted one-writer probe graph from its public task config."""
    base = make_entry_fixture()
    old_node = base.graph.node(base.node_id)
    reader = base.reader
    keep_public = {
        base.graph.instance.template_ref,
        base.params.versions_ref,
        base.params.provenance_ref,
        base.params.rng_ref,
        *(
            value
            for key, value in base.params.rendering.items()
            if key.endswith("_ref") and isinstance(value, str)
        ),
        *(edge.guard_ref for edge in old_node.edges),
    }
    versions = reader.public[base.params.versions_ref]
    keep_public.add(versions["admission_policy_ref"])
    reader.public = {key: value for key, value in reader.public.items() if key in keep_public}
    reader.private.clear()
    reader.context_revisions.clear()
    reader.context_nodes.clear()

    public = config["public"]
    limits = config["probe_settings"]
    checks_config = config["checks"]
    decision = public["decision"]
    feedback = public["feedback"]
    check_ids = (checks_config["required_id"], *(item["id"] for item in checks_config["optional"]))

    request_ref = reader.add({"kind": "probe-public-request", "schema": 1, "text": public["brief"]})
    files_ref = reader.add(
        {
            "kind": "legacy-visible-files",
            "schema": 1,
            "scenario_id": config["id"],
            "files": public["initial_files"],
        }
    )
    required_id = checks_config["required_id"]
    check_specs = [
        {"id": required_id, "kind": "nonempty", "path": "scene.txt"},
        *checks_config["optional"],
    ]
    checks = []
    for index, item in enumerate(check_specs, start=1):
        required = item["id"] == required_id
        spec = {
            "id": item["id"],
            "metric": f"Q{index}",
            "kind": item["kind"],
            "method": "deterministic",
            "required": required,
            **{key: value for key, value in item.items() if key not in {"id", "kind"}},
        }
        check = CheckContractV1(
            id=item["id"],
            evaluator_version="deterministic-v1",
            applicability="node_exit_candidate",
            required=required,
            spec=spec,
        )
        reader.private[check.identity()] = check.to_dict()
        checks.append(check)

    # Keep this private check outside the admitted evaluator packet: it is a leak
    # detector, not an evaluator operand or a value that may affect task behavior.
    canary_check = CheckContractV1(
        id="privacy_canary_only",
        evaluator_version="deterministic-v1",
        applicability="node_exit_candidate",
        required=False,
        spec={
            "id": "privacy_canary_only",
            "metric": "Q13",
            "kind": "nonempty",
            "method": "deterministic",
            "required": False,
            "path": "scene.txt",
            "private_fixture_canary": EVALUATOR_PACKET_CANARY,
        },
    )
    reader.private[canary_check.identity()] = canary_check.to_dict()

    reward = RewardContractV1(components=checks_config["components"], incomplete_score=0)
    packet = EvaluatorPacketV1(reward_contract_ref=reward.identity(), check_ids=check_ids)
    reader.private[reward.identity()] = reward.to_dict()
    reader.private[packet.identity()] = packet.to_dict()
    author_packet = AuthorPacketV1(
        preferences={
            **public["author_packet"]["preferences"],
            "unused_private_preference": AUTHOR_PACKET_CANARY,
        },
        requirements={},
    )
    answer = author_packet.preferences[decision["binding"]]
    script = ScriptedAuthorV1(
        answers={
            decision["id"]: {
                "mode": "fixed_answer",
                "utterance": f"Use the {answer} choice.",
                "value": answer,
                "selector": None,
                "prerequisite_check_ids": [],
            }
        },
        feedback=(
            {
                "id": feedback["id"],
                "utterance": feedback["utterance"],
                "prerequisite_check_ids": [],
                "requirement_update_ref": None,
            },
        ),
    )
    policy = InteractionPolicyV1(
        public_decisions=({"id": decision["id"], "label": decision["label"]},),
        mandatory_feedback=(feedback["id"],),
    )
    bindings = DecisionBindingsV1(bindings={decision["id"]: decision["binding"]})
    for record in (author_packet, script, bindings):
        reader.private[record.identity()] = record.to_dict()
    reader.public[policy.identity()] = policy.to_dict()

    old_contract = old_node.contract
    entry_contract = replace(
        old_contract.entry_contract,
        request_ref=request_ref,
        files_ref=files_ref,
        tool_allowlist=(*old_contract.entry_contract.tool_allowlist, "ask_author"),
    )
    interaction = InteractionContractV1(
        mode="scripted_author",
        script_ref=script.identity(),
        author_packet_ref=author_packet.identity(),
        interaction_policy_ref=policy.identity(),
        decision_bindings_ref=bindings.identity(),
        mandatory_feedback=(feedback["id"],),
    )
    budgets = replace(
        old_contract.budget_contract,
        max_steps=limits["max_writer_turns"],
        max_tool_calls=limits["max_tool_calls"],
        max_author_calls=limits["max_author_calls"],
        max_generated_tokens=limits["max_generated_tokens"],
        max_context_tokens=limits["max_context_tokens"],
    )
    completion = replace(
        old_contract.completion_contract,
        required_check_ids=(required_id,),
        required_script_turns=0,
        evaluation_packet_ref=packet.identity(),
    )
    contract = replace(
        old_contract,
        entry=entry_contract,
        interaction=interaction,
        budgets=budgets,
        completion=completion,
        mandatory_checks=(checks[0].identity(),),
        optional_checks=tuple(check.identity() for check in checks[1:]),
    )
    reader.public[contract.identity()] = contract.to_dict()
    spec = replace(old_node.spec, entry_contract=contract.identity())
    instance = replace(
        base.graph.instance,
        nodes=(spec,),
        source_refs=(files_ref,),
        request_refs=(request_ref,),
    )
    graph = admit_graph(
        instance,
        MappingArtifactResolver(reader.public, reader.private),
        policy=base.graph.policy,
    )
    derived = derive_entry(graph, base.node_id, base.params, reader)
    return EntryFixture(graph, base.node_id, base.params, reader, derived.state, derived.artifacts)


def bind_native_tokenizer(entry: EntryFixture, descriptor) -> EntryFixture:
    """Pin the tokenizer descriptor to an admitted task entry."""
    entry.reader.public[descriptor.identity()] = descriptor.to_wire()
    params = replace(
        entry.params,
        rendering={**entry.params.rendering, "tokenizer_ref": descriptor.identity()},
    )
    derived = derive_entry(entry.graph, entry.node_id, params, entry.reader)
    return replace(entry, params=params, state=derived.state, artifacts=derived.artifacts)


def persist_entry(store: TaskGraphStore, entry: EntryFixture) -> str:
    for body in entry.reader.public.values():
        store.put_artifact(body)
    for body in entry.reader.private.values():
        store.put_artifact(body, private=True)
    store.persist(entry.graph.instance)
    for artifact in entry.artifacts:
        store.persist_artifact(artifact)
    return store.save_checkpoint(entry.state)


def task_entries(
    store: TaskGraphStore, gate: LineageGate, tokenizer_descriptor
) -> tuple[TaskGraphTaskV1, ...]:
    entries = []
    for config in load_probe_tasks():
        fixture = bind_native_tokenizer(build_admitted_entry(config), tokenizer_descriptor)
        checkpoint = persist_entry(store, fixture)
        environment = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        entries.append(TaskGraphTaskV1(config["id"], environment, checkpoint))
    return tuple(entries)


def _quote(value: str) -> str:
    return f'<|"|>{value}<|"|>'


def _tool_call(name: str, arguments: dict[str, Any]) -> str:
    fields = []
    for key in sorted(arguments):
        value = arguments[key]
        if isinstance(value, str):
            rendered = _quote(value)
        elif isinstance(value, list):
            rendered = "[" + ",".join(_quote(item) for item in value) + "]"
        else:
            raise TypeError("fixture tool arguments use only strings and string arrays")
        fields.append(f"{key}:{rendered}")
    return f"<|tool_call>call:{name}{{{','.join(fields)}}}<tool_call|>"


def _fixture_actions(config: dict[str, Any], lineage: str) -> tuple[str, ...]:
    public = config["public"]
    example = config["scripted_lineages"][lineage]
    first = _tool_call(
        "write_file", {"path": "scene.txt", "content": public["initial_files"]["scene.txt"]}
    )
    decision = public["decision"]
    ask = _tool_call(
        "ask_author",
        {
            "question": decision["question"],
            "decision_ids": [decision["id"]],
            "proposals": [],
            "option_refs": [],
        },
    )
    writes = [_tool_call("write_file", {"path": "scene.txt", "content": example["scene"]})]
    if example["notes"] is not None:
        writes.append(
            _tool_call("write_file", {"path": "field-notes.txt", "content": example["notes"]})
        )
    return (
        first + "<|tool_response>",
        ask + "<|tool_response>",
        "".join(writes) + "<|tool_response>",
        "The scene is ready for review.<turn|>",
        "I applied the feedback revision.<turn|>",
    )


def fixture_plans(configs, recipe: GRPOSettings | None = None, *, all_tie: bool = False):
    """Return deterministic writer-seed-indexed messages used by CPU sampling."""
    recipe = settings() if recipe is None else recipe
    variants = (
        ("nonempty_only",) * 4,
        ("nonempty_only", "decision_phrase", "phrase_and_detail", "all_optional"),
        ("all_optional",) * 4,
    )
    result = {}
    for step in range(recipe.max_steps):
        config = configs[step % len(configs)]
        choices = (
            ("nonempty_only",) * recipe.group_size if all_tie else variants[step % len(variants)]
        )
        group_seed = derive_group_seed(recipe.seed, "task-graph-step", step)
        for ordinal, lineage in enumerate(choices):
            writer_seed = derive_group_seed(group_seed, "writer", ordinal)
            result[writer_seed] = _fixture_actions(config, lineage)
    return result


def make_task_graph_run(
    root: Path,
    *,
    model_factory,
    sample_backend_factory,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
    trainer_callback_factory=None,
    runtime_identity=None,
    tokenizer=None,
    tokenizer_root: Path = TOKENIZER_PATH,
    recipe: GRPOSettings | None = None,
):
    """Build, admit and train the shared P1 task group with injected model seams."""
    import torch
    from transformers import AutoTokenizer

    from writing_agent.native_gemma import NATIVE_STOP_TOKEN_IDS, make_native_manifest_descriptors
    from writing_agent.native_protocol import NATIVE_STOP_TOKENS

    torch.set_num_threads(2)
    tokenizer_root = tokenizer_root.expanduser().resolve()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_root), local_files_only=True, trust_remote_code=False
        )
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.parent.chmod(0o700)
    configs = load_probe_tasks()
    seed_fixture = build_admitted_entry(configs[0])
    seed_derived: EntryV1 = derive_entry(
        seed_fixture.graph, seed_fixture.node_id, seed_fixture.params, seed_fixture.reader
    )
    recipe = settings() if recipe is None else recipe
    tokenizer_hashes = {
        name: hashlib.sha256((tokenizer_root / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    }
    descriptors = make_native_manifest_descriptors(
        tokenizer,
        model_id=recipe.model_id,
        revision=recipe.revision,
        tokenizer_files_sha256=tokenizer_hashes,
        template_ref=seed_derived.view.context.rendering["template_ref"],
        tool_schema_ref=seed_derived.view.context.rendering["tool_schema_ref"],
        max_tokens_per_decision=recipe.max_tokens,
    )
    gate = LineageGate()
    store = TaskGraphStore(root, verifier=gate)
    entries = task_entries(store, gate, descriptors[1])
    if (
        tuple(tokenizer.convert_tokens_to_ids(token) for token in NATIVE_STOP_TOKENS)
        != NATIVE_STOP_TOKEN_IDS
    ):
        raise ValueError("cached Gemma tokenizer does not match native stop tokens")
    if runtime_identity is None:
        runtime_identity = {
            "model": "gemma4-e2b-tiny-random-fp32-cpu-v1",
            "tokenizer_revision": TOKENIZER_REVISION,
            "fixture_tasks": [config["id"] for config in configs],
        }
    return train_task_graph(
        entries,
        root,
        settings=recipe,
        model=None,
        model_factory=model_factory,
        tokenizer=tokenizer,
        manifest_descriptors=descriptors,
        runtime_identity=runtime_identity,
        resume_from_checkpoint=resume,
        stop_after_steps=stop_after_steps,
        sample_backend_factory=sample_backend_factory,
        trainer_callback_factory=trainer_callback_factory,
        tokenizer_root=tokenizer_root,
    )


__all__ = [
    "AUTHOR_PACKET_CANARY",
    "CONFIG_DIR",
    "EVALUATOR_PACKET_CANARY",
    "TOKENIZER_PATH",
    "TOKENIZER_REVISION",
    "EntryFixture",
    "MemoryArtifactReader",
    "bind_native_tokenizer",
    "build_admitted_entry",
    "fixture_plans",
    "load_probe_task",
    "load_probe_tasks",
    "make_entry_fixture",
    "make_task_graph_run",
    "persist_entry",
    "settings",
    "task_entries",
    "tiny_gemma",
]
