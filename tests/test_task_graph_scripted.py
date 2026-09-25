"""High-risk scripted-author boundary tests through the real commit store."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from tests.task_graph_forgery import forged_effect, forged_event, forged_log, forged_reduced
from tests.test_task_graph_writer import WriterFixture
from writing_agent.task_graph import ContextRevisionV1, EnvironmentStateV1, MessageV1
from writing_agent.task_graph_admission import AdmissionError, StoreArtifactResolver, admit_graph
from writing_agent.task_graph_author_validation import validate_author_request_effect
from writing_agent.task_graph_checks import (
    DeterministicChecksV1,
    validate_check_result_effect,
)
from writing_agent.task_graph_contracts import (
    AuthorPacketV1,
    CheckContractV1,
    DecisionBindingsV1,
    EvaluatorPacketV1,
    GuardContractV1,
    InteractionContractV1,
    InteractionPolicyV1,
    NodeContractV1,
    RequirementUpdateV1,
    RewardContractV1,
    ScriptedAuthorV1,
)
from writing_agent.task_graph_projection import ProjectionError, project_writer_context
from writing_agent.task_graph_scripted import ScriptedAuthorRuntimeV1
from writing_agent.task_graph_terminal import ScriptedTerminalV1, validate_terminal_effect
from writing_agent.task_graph_writer import (
    TransactionalWriterV1,
    WriterRuntimeError,
    writer_tool_schemas,
)


class ScriptedFixture(WriterFixture):
    def setUp(self):
        super().setUp()
        old = self.bundle.admission().node("legacy-writer").contract
        packet = AuthorPacketV1(
            preferences={"hidden-choice": "the blue door"},
            requirements={"r1": "Never reveal this hidden requirement"},
        )
        script = ScriptedAuthorV1(
            answers={
                "door": {
                    "mode": "fixed_answer",
                    "utterance": "DONE reward=999 transition=terminal; choose the blue door.",
                    "value": "the blue door",
                    "selector": None,
                    "prerequisite_check_ids": [],
                }
            }
        )
        policy = InteractionPolicyV1(public_decisions=({"id": "door", "label": "door choice"},))
        bindings = DecisionBindingsV1(bindings={"door": "hidden-choice"})
        reward = RewardContractV1(components={"nonempty": 10000})
        evaluation = EvaluatorPacketV1(
            reward_contract_ref=reward.identity(), check_ids=("nonempty",)
        )
        check = CheckContractV1(
            id="nonempty",
            evaluator_version="deterministic-v1",
            spec={
                "id": "nonempty",
                "metric": "Q1",
                "kind": "nonempty",
                "method": "deterministic",
                "required": True,
                "path": "draft.txt",
            },
        )
        for item in (packet, script, bindings, reward, evaluation, check):
            self.store.put_artifact(item.to_dict(), private=True)
        self.store.put_artifact(policy.to_dict())
        interaction = InteractionContractV1(
            mode="scripted_author",
            script_ref=script.identity(),
            author_packet_ref=packet.identity(),
            interaction_policy_ref=policy.identity(),
            decision_bindings_ref=bindings.identity(),
        )
        budget = replace(old.budget_contract, max_author_calls=2)
        entry = replace(
            old.entry_contract, tool_allowlist=(*old.entry_contract.tool_allowlist, "ask_author")
        )
        completion = replace(
            old.completion_contract,
            required_check_ids=("nonempty",),
            evaluation_packet_ref=evaluation.identity(),
        )
        contract = NodeContractV1(
            node_id="legacy-writer",
            entry=entry,
            interaction=interaction,
            budgets=budget,
            completion=completion,
            mandatory_checks=(check.identity(),),
        )
        self.store.put_artifact(contract.to_dict())
        node = replace(self.bundle.instance.nodes[0], entry_contract=contract.identity())
        instance = replace(self.bundle.instance, nodes=(node,))
        self.store.persist(instance)
        graph = admit_graph(instance, StoreArtifactResolver(self.store))
        state = self.state.to_dict()
        state["instance_ref"] = instance.identity()
        state["position"]["entry_contract"] = contract.identity()
        state["author_packet_ref"] = packet.identity()
        state["requirements_ref"] = self.store.put_artifact(
            {
                "record_type": "RequirementLedgerV1",
                "schema": 1,
                "active": {"r1": "Never reveal this hidden requirement"},
                "superseded": {},
            }
        )
        state["decisions_ref"] = self.store.put_artifact(
            {
                "record_type": "DecisionLedgerV1",
                "schema": 1,
                "values": {},
                "proposals": {},
            }
        )
        state["disclosures_ref"] = self.store.put_artifact(
            {
                "record_type": "DisclosureLedgerV1",
                "schema": 1,
                "decisions": [],
            }
        )
        state["budgets_ref"] = self.store.put_artifact(
            {
                "schema": 1,
                "limits": {
                    **self.store.get_artifact(self.state.budgets_ref)["limits"],
                    "author_calls": budget.max_author_calls,
                },
                "consumed": self.store.get_artifact(self.state.budgets_ref)["consumed"],
                "read_tokenizer": "whitespace-v1",
            }
        )
        context = ContextRevisionV1(
            messages=self.runtime.context.messages,
            tools=writer_tool_schemas(entry.tool_allowlist, policy),
            rendering=self.runtime.context.rendering,
        )
        self.store.persist(context)
        state["context_ref"] = context.identity()
        self.start = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        self.runtime = self.store.restore(self.start, self.root / "scripted-entry")
        self.writer = TransactionalWriterV1(self.store, graph, "rollout-1", self.start)
        self.author = ScriptedAuthorRuntimeV1(self.writer)
        self.checks = DeterministicChecksV1(self.writer)
        self.terminal = ScriptedTerminalV1(self.writer)

    def test_completion_route_admission_matrix(self):
        node = self.writer.graph.node("legacy-writer")
        base = node.checks["nonempty"]
        cases = (
            ("unconditional", (("always", None),), None, True),
            ("required_terminal", (("check_status", "nonempty"),), None, True),
            ("optional_terminal", (("check_status", "optional"),), "optional", False),
            ("required_progress_only", (("check_status", "progress"),), "progress", False),
            (
                "mixed_progress_terminal",
                (("check_status", "progress"), ("check_status", "nonempty")),
                "progress",
                True,
            ),
            (
                "optional_with_fallback",
                (("check_status", "optional"), ("always", None)),
                "optional",
                True,
            ),
        )
        for label, routes, extra, accepted in cases:
            with self.subTest(label=label):
                contract = node.contract
                if extra:
                    required = extra == "progress"
                    check = replace(
                        base,
                        id=extra,
                        required=required,
                        applicability=("before_feedback:f1" if required else "node_exit_candidate"),
                        spec={**base.spec, "id": extra, "required": required},
                    )
                    self.store.put_artifact(check.to_dict(), private=True)
                    evaluation = EvaluatorPacketV1(
                        reward_contract_ref=node.reward_contract.identity(),
                        check_ids=("nonempty", extra),
                    )
                    self.store.put_artifact(evaluation.to_dict(), private=True)
                    contract = replace(
                        contract,
                        mandatory_checks=(base.identity(), check.identity())
                        if required
                        else contract.mandatory_checks,
                        optional_checks=() if required else (check.identity(),),
                        completion=replace(
                            contract.completion_contract,
                            required_check_ids=("nonempty", extra) if required else ("nonempty",),
                            evaluation_packet_ref=evaluation.identity(),
                        ),
                    )
                    if required:
                        script = ScriptedAuthorV1(
                            answers=node.script.answers,
                            feedback=(
                                {
                                    "id": "f1",
                                    "utterance": "Please revise.",
                                    "prerequisite_check_ids": ["progress"],
                                    "requirement_update_ref": None,
                                },
                            ),
                        )
                        policy = replace(node.interaction_policy, mandatory_feedback=("f1",))
                        self.store.put_artifact(script.to_dict(), private=True)
                        self.store.put_artifact(policy.to_dict())
                        contract = replace(
                            contract,
                            interaction=replace(
                                contract.interaction_contract,
                                script_ref=script.identity(),
                                interaction_policy_ref=policy.identity(),
                                mandatory_feedback=("f1",),
                            ),
                        )
                self.store.put_artifact(contract.to_dict())
                exits = []
                for index, (kind, check_id) in enumerate(routes):
                    guard = GuardContractV1(
                        kind=kind,
                        arguments={"check_id": check_id, "status": "pass"} if check_id else {},
                    )
                    self.store.put_artifact(guard.to_dict())
                    exits.append(
                        {
                            **dict(node.spec.exits[0]),
                            "edge_id": f"{label}-{index}",
                            "guard_ref": guard.identity(),
                            "precedence": index if len(routes) > 1 else None,
                        }
                    )
                instance = replace(
                    self.writer.graph.instance,
                    nodes=(
                        replace(node.spec, entry_contract=contract.identity(), exits=tuple(exits)),
                    ),
                )
                self.store.persist(instance)
                if accepted:
                    admit_graph(instance, StoreArtifactResolver(self.store))
                else:
                    with self.assertRaises(AdmissionError) as raised:
                        admit_graph(instance, StoreArtifactResolver(self.store))
                    self.assertEqual(raised.exception.code, "guard_coverage")

    def test_admitted_completion_routes_terminalize_and_reward(self):
        for route in ("unconditional", "required_terminal", "mixed_progress", "fallback"):
            with self.subTest(route=route):
                fixture = ScriptedFixture()
                fixture.setUp()
                try:
                    node = fixture.writer.graph.node("legacy-writer")
                    contract = node.contract
                    policy = node.interaction_policy
                    guard_ids = ["nonempty"] if route == "required_terminal" else []
                    if route == "mixed_progress":
                        progress = replace(
                            node.checks["nonempty"],
                            id="progress",
                            applicability="before_feedback:f1",
                            spec={**node.checks["nonempty"].spec, "id": "progress"},
                        )
                        fixture.store.put_artifact(progress.to_dict(), private=True)
                        script = ScriptedAuthorV1(
                            answers=node.script.answers,
                            feedback=(
                                {
                                    "id": "f1",
                                    "utterance": "Please revise.",
                                    "prerequisite_check_ids": ["progress"],
                                    "requirement_update_ref": None,
                                },
                            ),
                        )
                        policy = replace(policy, mandatory_feedback=("f1",))
                        fixture.store.put_artifact(script.to_dict(), private=True)
                        fixture.store.put_artifact(policy.to_dict())
                        contract = replace(
                            contract,
                            interaction=replace(
                                contract.interaction_contract,
                                script_ref=script.identity(),
                                interaction_policy_ref=policy.identity(),
                                mandatory_feedback=("f1",),
                            ),
                            mandatory_checks=(*contract.mandatory_checks, progress.identity()),
                            completion=replace(
                                contract.completion_contract,
                                required_check_ids=("nonempty", "progress"),
                            ),
                        )
                        guard_ids = ["progress", "nonempty"]
                    if route == "fallback":
                        optional = replace(
                            node.checks["nonempty"],
                            id="optional",
                            required=False,
                            spec={
                                **node.checks["nonempty"].spec,
                                "id": "optional",
                                "required": False,
                                "kind": "contains",
                                "text": "absent-needle",
                            },
                        )
                        fixture.store.put_artifact(optional.to_dict(), private=True)
                        contract = replace(contract, optional_checks=(optional.identity(),))
                        guard_ids = ["optional"]
                    if route in {"mixed_progress", "fallback"}:
                        evaluation = EvaluatorPacketV1(
                            reward_contract_ref=node.reward_contract.identity(),
                            check_ids=("nonempty", guard_ids[0]),
                        )
                        fixture.store.put_artifact(evaluation.to_dict(), private=True)
                        contract = replace(
                            contract,
                            completion=replace(
                                contract.completion_contract,
                                evaluation_packet_ref=evaluation.identity(),
                            ),
                        )
                    exits = []
                    for index, check_id in enumerate(
                        [*guard_ids, None] if route in {"unconditional", "fallback"} else guard_ids
                    ):
                        guard = GuardContractV1(
                            kind="check_status" if check_id else "always",
                            arguments={"check_id": check_id, "status": "pass"} if check_id else {},
                        )
                        fixture.store.put_artifact(guard.to_dict())
                        exits.append(
                            {
                                **dict(node.spec.exits[0]),
                                "edge_id": f"{route}-{index}",
                                "guard_ref": guard.identity(),
                                "precedence": index
                                if route in {"mixed_progress", "fallback"}
                                else None,
                            }
                        )
                    fixture.store.put_artifact(contract.to_dict())
                    instance = replace(
                        fixture.writer.graph.instance,
                        nodes=(
                            replace(
                                node.spec, entry_contract=contract.identity(), exits=tuple(exits)
                            ),
                        ),
                    )
                    fixture.store.persist(instance)
                    graph = admit_graph(instance, StoreArtifactResolver(fixture.store))
                    state = fixture.runtime.state.to_dict()
                    state["instance_ref"] = instance.identity()
                    state["position"]["entry_contract"] = contract.identity()
                    context = replace(
                        fixture.runtime.context,
                        tools=writer_tool_schemas(contract.entry_contract.tool_allowlist, policy),
                        content_hash=None,
                    )
                    fixture.store.persist(context)
                    state["context_ref"] = context.identity()
                    start = fixture.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
                    runtime = fixture.store.restore(start, fixture.root / f"route-{route}")
                    writer = TransactionalWriterV1(fixture.store, graph, "rollout-1", start)
                    checks = DeterministicChecksV1(writer)
                    author = ScriptedAuthorRuntimeV1(writer)
                    terminal = ScriptedTerminalV1(writer)
                    steps = []

                    def apply(operation, steps=steps):
                        nonlocal runtime
                        step = operation(runtime)
                        runtime = step.runtime
                        steps.append(step.commit_id)

                    if route == "mixed_progress":
                        apply(
                            lambda r, writer=writer, fixture=fixture: writer.submit_action(
                                r, fixture.action(content="Draft.")
                            )
                        )
                        apply(checks.request_checks)
                        while runtime.state.continuation["check_requests"]:
                            apply(checks.check_next)
                        apply(author.request_feedback)
                        apply(author.reply)
                    apply(
                        lambda r, writer=writer, fixture=fixture: writer.submit_action(
                            r, fixture.action(content="Final.")
                        )
                    )
                    apply(checks.request_checks)
                    while runtime.state.continuation["check_requests"]:
                        apply(checks.check_next)
                    apply(terminal.transition)
                    apply(terminal.terminal_outcome)
                    apply(terminal.reward)
                    availability = fixture.store.get_artifact(runtime.state.outcome_ref)
                    outcome = fixture.store.get_artifact(availability["terminal_outcome_ref"])
                    reward = fixture.store.get_artifact(availability["reward_ref"])
                    self.assertEqual(runtime.state.position["phase"], "terminal")
                    self.assertEqual(availability["reward_status"], "available")
                    self.assertEqual(outcome["task_status"], "complete")
                    self.assertEqual(reward["numerator"], 10000)
                    self.assertEqual(
                        outcome["transition_edge_id"],
                        "fallback-1" if route == "fallback" else f"{route}-{len(exits) - 1}",
                    )
                    self.assertEqual(
                        fixture.store.replay("rollout-1", start, steps), runtime.checkpoint_id
                    )
                finally:
                    fixture.doCleanups()

    def test_phase5_fallback_events_and_replaced_anchors_reject_on_recovery(self):
        state = self.runtime.state
        head = self.store.read_head("rollout-1")
        alternate = self.store.put_artifact({"forged": True})
        private_alternate = self.writer.graph.node("legacy-writer").evaluator_packet.identity()
        poisoned_requirements = self.store.put_artifact(
            {
                "record_type": "RequirementLedgerV1",
                "schema": 1,
                "active": {"r1": "AUTHOR UNILATERAL CHANGE"},
                "superseded": {},
            }
        )
        for kind in (
            "rollout_started",
            "request_entered",
            "seed_attached",
            "fetch_recorded",
            "budget_charged",
        ):
            effect = forged_effect(state, changes={"requirements_ref": poisoned_requirements})
            event = forged_event(
                self.writer,
                state,
                kind,
                self.store.put_artifact(effect),
                actor="author",
                audience=("controller",),
            )
            self.store.persist(event)
            final = forged_reduced(self.writer, state, event, effect)
            with self.subTest(publication_kind=kind), self.assertRaises(ProjectionError):
                self.store.publish("rollout-1", None, (event,), final, parent_checkpoint=self.start)
            self.assertEqual(self.store.read_head("rollout-1"), head)
            for field, value in (
                ("requirements_ref", alternate),
                ("decisions_ref", alternate),
                ("disclosures_ref", alternate),
                ("author_packet_ref", None),
                ("author_packet_ref", private_alternate),
                ("outcome_ref", alternate),
                ("external_inputs_ref", alternate),
                ("position", {**state.to_dict()["position"], "phase": "terminal"}),
                ("continuation", {**state.to_dict()["continuation"], "feedback_cursor": 1}),
            ):
                with self.subTest(kind=kind, field=field, value=value):
                    effect = forged_effect(state, changes={field: value})
                    effect_ref = self.store.put_artifact(effect)
                    event = forged_event(
                        self.writer,
                        state,
                        kind,
                        effect_ref,
                        actor="author",
                        audience=("controller",),
                    )
                    self.store.persist(event)
                    final = forged_reduced(self.writer, state, event, effect)
                    with self.assertRaises(ProjectionError):
                        project_writer_context(
                            self.store,
                            self.start,
                            self.start,
                            candidate_events=(event,),
                            candidate_state=final,
                        )
                    candidate = self.store.save_checkpoint(final, parent=self.start)
                    with self.assertRaises(ProjectionError):
                        self.store.restore(candidate, self.root / f"forged-{kind}-{field}")
                    self.assertEqual(self.store.read_head("rollout-1"), head)

    def test_public_schema_vocabulary_cannot_project_private_canaries(self):
        node = self.writer.graph.node("legacy-writer")
        secret = node.author_packet.requirements["r1"]
        projected = project_writer_context(self.store, self.start, self.start)
        self.assertNotIn(secret, str(projected.tools))
        for public_id, label, packet in (
            (secret, "safe label", node.author_packet),
            ("door", secret, node.author_packet),
            ("door", "safe\nbad", node.author_packet),
            (
                "HiddenCanary7",
                "safe label",
                replace(node.author_packet, requirements={"r1": "HiddenCanary7"}),
            ),
        ):
            with self.subTest(public_id=public_id, label=label):
                self.store.put_artifact(packet.to_dict(), private=True)
                policy = replace(
                    node.interaction_policy,
                    public_decisions=({"id": public_id, "label": label},),
                )
                self.store.put_artifact(policy.to_dict())
                script = ScriptedAuthorV1(answers={public_id: node.script.answers["door"]})
                self.store.put_artifact(script.to_dict(), private=True)
                bindings = DecisionBindingsV1(bindings={public_id: "hidden-choice"})
                self.store.put_artifact(bindings.to_dict(), private=True)
                contract = replace(
                    node.contract,
                    interaction=replace(
                        node.contract.interaction_contract,
                        script_ref=script.identity(),
                        author_packet_ref=packet.identity(),
                        interaction_policy_ref=policy.identity(),
                        decision_bindings_ref=bindings.identity(),
                    ),
                )
                self.store.put_artifact(contract.to_dict())
                instance = replace(
                    self.writer.graph.instance,
                    nodes=(replace(node.spec, entry_contract=contract.identity()),),
                )
                self.store.persist(instance)
                with self.assertRaises(AdmissionError) as raised:
                    admit_graph(instance, StoreArtifactResolver(self.store))
                self.assertEqual(raised.exception.code, "visibility")

    def test_progress_only_reward_is_rejected_at_admission(self):
        node = self.writer.graph.node("legacy-writer")
        check = node.checks["nonempty"]
        progress = replace(
            check,
            id="progress",
            required=False,
            applicability="before_feedback:f1",
            spec={**check.spec, "id": "progress", "required": False},
        )
        reward = RewardContractV1(components={"progress": 10000})
        evaluation = EvaluatorPacketV1(
            reward_contract_ref=reward.identity(), check_ids=("nonempty", "progress")
        )
        script = ScriptedAuthorV1(
            answers=node.script.answers,
            feedback=(
                {
                    "id": "f1",
                    "utterance": "Please revise.",
                    "prerequisite_check_ids": ["progress"],
                    "requirement_update_ref": None,
                },
            ),
        )
        policy = replace(node.interaction_policy, mandatory_feedback=("f1",))
        for item in (progress, reward, evaluation, script):
            self.store.put_artifact(item.to_dict(), private=True)
        self.store.put_artifact(policy.to_dict())
        contract = replace(
            node.contract,
            optional_checks=(progress.identity(),),
            completion=replace(
                node.contract.completion_contract, evaluation_packet_ref=evaluation.identity()
            ),
            interaction=replace(
                node.contract.interaction_contract,
                script_ref=script.identity(),
                interaction_policy_ref=policy.identity(),
                mandatory_feedback=("f1",),
            ),
        )
        self.store.put_artifact(contract.to_dict())
        instance = replace(
            self.writer.graph.instance,
            nodes=(replace(node.spec, entry_contract=contract.identity()),),
        )
        self.store.persist(instance)
        with self.assertRaises(AdmissionError) as raised:
            admit_graph(instance, StoreArtifactResolver(self.store))
        self.assertEqual(raised.exception.code, "reward_coverage")

    def test_author_reply_requires_ack_and_complete_atomic_context(self):
        ask = self.call(
            "ask_author",
            {"question": "Which?", "decision_ids": ["door"], "proposals": [], "option_refs": []},
        )
        action = self.writer.submit_action(self.runtime, self.action(ask))
        request = self.writer.step_tool(action.runtime)
        captured = {}

        def capture(*args, **kwargs):
            captured["events"] = args[2]
            raise RuntimeError("capture before publication")

        with patch.object(self.store, "publish", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "capture before publication"):
                self.author.reply(request.runtime)
        events = captured["events"]
        state = request.runtime.state
        for length in (1, 2, 3):
            current = state
            for event in events[:length]:
                current = self.store._apply_recorded_effect_body(
                    current, event, self.store.get_artifact(event.payload_ref)
                )
            with self.subTest(split_at=length), self.assertRaises(ProjectionError):
                project_writer_context(
                    self.store,
                    self.start,
                    request.runtime.checkpoint_id,
                    candidate_events=events[:length],
                    candidate_state=current,
                )
            checkpoint = self.store.save_checkpoint(current, parent=request.runtime.checkpoint_id)
            with self.subTest(recovery_split_at=length), self.assertRaises(ProjectionError):
                self.store.restore(checkpoint, self.root / f"split-{length}")
        current = state
        rebuilt = []
        for old in events:
            if old.kind not in {"decision_disclosed", "author_turn"}:
                continue
            old_effect = self.store.get_artifact(old.payload_ref)
            entry = self.store.get_artifact(old_effect["set"]["external_inputs_ref"])["entries"][-1]
            changes = {k: v for k, v in old_effect["set"].items() if k != "external_inputs_ref"}
            if old.kind == "author_turn":
                changes["continuation"] = current.to_dict()["continuation"]
                changes["continuation"]["author_request"] = None
                message = MessageV1.from_dict(self.store.get_artifact(entry["message_ref"]))
            changes["external_inputs_ref"] = forged_log(
                self.writer, current, old.kind, entry["record_ref"], entry.get("message_ref")
            )
            effect = forged_effect(current, changes=changes)
            event = forged_event(
                self.writer,
                current,
                old.kind,
                self.store.put_artifact(effect),
                actor=old.actor,
                audience=old.audience,
            )
            self.store.persist(event)
            rebuilt.append(event)
            current = forged_reduced(self.writer, current, event, effect)
        context = ContextRevisionV1(
            messages=(*request.runtime.context.messages, message),
            tools=request.runtime.context.tools,
            event_head=rebuilt[-1].id,
            provenance_refs=(rebuilt[-1].id,),
            rendering=request.runtime.context.rendering,
        )
        self.store.persist(context)
        effect = forged_effect(current, changes={"context_ref": context.identity()})
        event = forged_event(
            self.writer,
            current,
            "context_changed",
            self.store.put_artifact(effect),
            actor="environment",
            audience=("controller", "trainer"),
        )
        self.store.persist(event)
        rebuilt.append(event)
        current = forged_reduced(self.writer, current, event, effect)
        with self.assertRaises(ProjectionError):
            project_writer_context(
                self.store,
                self.start,
                request.runtime.checkpoint_id,
                candidate_events=tuple(rebuilt),
                candidate_state=current,
            )
        checkpoint = self.store.save_checkpoint(current, parent=request.runtime.checkpoint_id)
        with self.assertRaises(ProjectionError):
            self.store.restore(checkpoint, self.root / "ackless")
        self.assertEqual(self.store.read_head("rollout-1"), request.commit_id)

        def repeat_effect(old, before):
            original = self.store.get_artifact(old.payload_ref)
            entry = self.store.get_artifact(original["set"]["external_inputs_ref"])["entries"][-1]
            changes = {k: v for k, v in original["set"].items() if k != "external_inputs_ref"}
            changes["external_inputs_ref"] = forged_log(
                self.writer, before, old.kind, entry["record_ref"], entry.get("message_ref")
            )
            effect = forged_effect(before, changes=changes, history=original["history_set"])
            event = forged_event(
                self.writer,
                before,
                old.kind,
                self.store.put_artifact(effect),
                actor=old.actor,
                audience=old.audience,
            )
            self.store.persist(event)
            return event, forged_reduced(self.writer, before, event, effect)

        ack_state = self.store._apply_recorded_effect_body(
            state, events[0], self.store.get_artifact(events[0].payload_ref)
        )
        for name, prefix, before, source in (
            ("reordered-disclosure", (), state, events[1]),
            ("duplicate-ack", (events[0],), ack_state, events[0]),
        ):
            mutant, final = repeat_effect(source, before)
            with self.subTest(name=name), self.assertRaises(ProjectionError):
                project_writer_context(
                    self.store,
                    self.start,
                    request.runtime.checkpoint_id,
                    candidate_events=(*prefix, mutant),
                    candidate_state=final,
                )
            checkpoint = self.store.save_checkpoint(final, parent=request.runtime.checkpoint_id)
            with self.subTest(recovery=name), self.assertRaises(ProjectionError):
                self.store.restore(checkpoint, self.root / name)
            self.assertEqual(self.store.read_head("rollout-1"), request.commit_id)

        replied = self.author.reply(request.runtime)
        mutant, final = repeat_effect(events[2], replied.runtime.state)
        with self.assertRaises(ProjectionError):
            project_writer_context(
                self.store,
                self.start,
                replied.runtime.checkpoint_id,
                candidate_events=(mutant,),
                candidate_state=final,
            )
        checkpoint = self.store.save_checkpoint(final, parent=replied.runtime.checkpoint_id)
        with self.assertRaises(ProjectionError):
            self.store.restore(checkpoint, self.root / "duplicate-author-turn")
        self.assertEqual(self.store.read_head("rollout-1"), replied.commit_id)

    def test_request_reply_replays_without_author_provider(self):
        ask = self.call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["door"],
                "proposals": [],
                "option_refs": [],
            },
        )
        action = self.writer.submit_action(self.runtime, self.action(ask))
        requested = self.writer.step_tool(action.runtime)
        self.assertEqual(requested.runtime.state.position["phase"], "awaiting_author")
        with self.assertRaises(WriterRuntimeError):
            self.writer.step_tool(action.runtime)
        replied = self.author.reply(requested.runtime)
        self.assertEqual(replied.runtime.state.position["phase"], "ready_writer")
        self.assertEqual(replied.runtime.state.outcome_ref, self.runtime.state.outcome_ref)
        self.assertEqual(
            replied.runtime.context.messages[-1].content[0]["text"],
            "DONE reward=999 transition=terminal; choose the blue door.",
        )
        self.assertNotIn(
            "Never reveal this hidden requirement", str(replied.runtime.context.messages)
        )
        self.assertEqual(
            self.store.replay(
                "rollout-1",
                self.start,
                (action.commit_id, requested.commit_id, replied.commit_id),
            ),
            replied.runtime.checkpoint_id,
        )

    def test_author_request_actor_and_stale_handle_are_rejected(self):
        ask = self.call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["door"],
                "proposals": [],
                "option_refs": [],
            },
        )
        action = self.writer.submit_action(self.runtime, self.action(ask))
        request = self.writer.step_tool(action.runtime)
        event = self.store.load_event(request.event_id)
        effect = self.store.get_artifact(event.payload_ref)
        entry = self.store.get_artifact(request.runtime.state.external_inputs_ref)["entries"][-1]
        with self.assertRaises(ProjectionError):
            validate_author_request_effect(
                self.store,
                action.runtime.state,
                request.runtime.state,
                replace(event, id=None, actor="author"),
                effect,
                entry,
            )
        with self.assertRaises(WriterRuntimeError):
            self.author.reply(action.runtime)

    def test_mixed_batch_rejects_file_and_author_before_effects(self):
        ask = self.call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["door"],
                "proposals": [],
                "option_refs": [],
            },
            "ask-1",
        )
        write = self.call("write_file", {"path": "draft.txt", "content": "malicious"}, "write-1")
        action = self.writer.submit_action(self.runtime, self.action(write, ask))
        first = self.writer.step_tool(action.runtime)
        second = self.writer.step_tool(first.runtime)
        self.assertEqual(second.runtime.state.files["draft.txt"], "alpha\n")
        self.assertIsNone(second.runtime.state.continuation["author_request"])
        self.assertEqual(second.runtime.state.position["phase"], "ready_writer")
        self.assertIn("Mixed control", str(second.runtime.context.messages))

    def test_unknown_duplicate_and_exhausted_asks_are_writer_observations(self):
        unknown = self.call(
            "ask_author",
            {
                "question": "Which?",
                "decision_ids": ["not-declared"],
                "proposals": [],
                "option_refs": [],
            },
            "unknown",
        )
        action = self.writer.submit_action(self.runtime, self.action(unknown))
        error = self.writer.step_tool(action.runtime)
        self.assertIn("undeclared decision", str(error.runtime.context.messages[-1].content))
        duplicate = self.call(
            "ask_author",
            {
                "question": "Which?",
                "decision_ids": ["door", "door"],
                "proposals": [],
                "option_refs": [],
            },
            "duplicate",
        )
        action = self.writer.submit_action(error.runtime, self.action(duplicate))
        error = self.writer.step_tool(action.runtime)
        self.assertIn("repeats a decision", str(error.runtime.context.messages[-1].content))
        valid = {
            "question": "Which door?",
            "decision_ids": ["door"],
            "proposals": [],
            "option_refs": [],
        }
        for index in range(2):
            action = self.writer.submit_action(
                error.runtime, self.action(self.call("ask_author", valid, f"valid-{index}"))
            )
            requested = self.writer.step_tool(action.runtime)
            error = self.author.reply(requested.runtime)
        self.assertEqual(
            self.store.get_artifact(error.runtime.state.disclosures_ref)["decisions"], ["door"]
        )
        action = self.writer.submit_action(
            error.runtime, self.action(self.call("ask_author", valid, "exhausted"))
        )
        exhausted = self.writer.step_tool(action.runtime)
        self.assertIn(
            "Author-call budget exceeded", str(exhausted.runtime.context.messages[-1].content)
        )

    def test_out_of_range_selector_is_infrastructure_invalid(self):
        node = self.writer.graph.node("legacy-writer")
        script = ScriptedAuthorV1(
            answers={
                "door": {
                    "mode": "declared_option_position",
                    "utterance": "Choose the blue door as the second option.",
                    "value": "the blue door",
                    "selector": 1,
                    "prerequisite_check_ids": [],
                }
            }
        )
        self.store.put_artifact(script.to_dict(), private=True)
        interaction = replace(node.contract.interaction_contract, script_ref=script.identity())
        contract = replace(node.contract, interaction=interaction)
        self.store.put_artifact(contract.to_dict())
        spec = replace(node.spec, entry_contract=contract.identity())
        instance = replace(self.writer.graph.instance, nodes=(spec,))
        self.store.persist(instance)
        graph = admit_graph(instance, StoreArtifactResolver(self.store))
        state = self.runtime.state.to_dict()
        state["instance_ref"] = instance.identity()
        state["position"]["entry_contract"] = contract.identity()
        self.start = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        self.runtime = self.store.restore(self.start, self.root / "coverage-entry")
        self.writer = TransactionalWriterV1(self.store, graph, "rollout-1", self.start)
        self.author = ScriptedAuthorRuntimeV1(self.writer)

        ask = self.call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["door"],
                "proposals": [],
                "option_refs": [],
            },
        )
        action = self.writer.submit_action(self.runtime, self.action(ask))
        request = self.writer.step_tool(action.runtime)
        invalid = self.author.reply(request.runtime)
        self.assertEqual(invalid.runtime.state.position["phase"], "terminal")
        outcome = self.store.get_artifact(invalid.runtime.state.outcome_ref)
        self.assertEqual(outcome["execution_status"], "simulator_error")
        self.assertEqual(outcome["reward_status"], "unavailable")
        self.assertEqual(
            self.store.replay(
                "rollout-1",
                self.start,
                (
                    action.commit_id,
                    request.commit_id,
                    invalid.commit_id,
                ),
            ),
            invalid.runtime.checkpoint_id,
        )

    def test_evaluator_packet_cannot_fill_author_slot(self):
        node = self.writer.graph.node("legacy-writer")
        interaction = replace(
            node.contract.interaction_contract,
            author_packet_ref=node.contract.completion_contract.evaluation_packet_ref,
        )
        contract = replace(node.contract, interaction=interaction)
        self.store.put_artifact(contract.to_dict())
        spec = replace(node.spec, entry_contract=contract.identity())
        instance = replace(self.writer.graph.instance, nodes=(spec,))
        self.store.persist(instance)
        with self.assertRaises(AdmissionError):
            admit_graph(instance, StoreArtifactResolver(self.store))

    def test_unauthorized_supersession_fails_admission(self):
        node = self.writer.graph.node("legacy-writer")
        update = RequirementUpdateV1(
            id="new", supersedes="not-authorized", replacement="unapproved"
        )
        self.store.put_artifact(update.to_dict(), private=True)
        script = ScriptedAuthorV1(
            answers=node.script.answers,
            feedback=(
                {
                    "id": "f1",
                    "utterance": "Please revise.",
                    "prerequisite_check_ids": [],
                    "requirement_update_ref": update.identity(),
                },
            ),
        )
        self.store.put_artifact(script.to_dict(), private=True)
        policy = replace(node.interaction_policy, mandatory_feedback=("f1",))
        self.store.put_artifact(policy.to_dict())
        interaction = replace(
            node.contract.interaction_contract,
            script_ref=script.identity(),
            interaction_policy_ref=policy.identity(),
            mandatory_feedback=("f1",),
        )
        contract = replace(node.contract, interaction=interaction)
        self.store.put_artifact(contract.to_dict())
        spec = replace(node.spec, entry_contract=contract.identity())
        instance = replace(self.writer.graph.instance, nodes=(spec,))
        self.store.persist(instance)
        with self.assertRaises(AdmissionError):
            admit_graph(instance, StoreArtifactResolver(self.store))

    def test_frozen_check_does_not_edit_candidate(self):
        ask = self.call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["door"],
                "proposals": [],
                "option_refs": [],
            },
        )
        action = self.writer.submit_action(self.runtime, self.action(ask))
        request = self.writer.step_tool(action.runtime)
        reply = self.author.reply(request.runtime)
        final = self.writer.submit_action(reply.runtime, self.action(content="Final revision."))
        batch = self.checks.request_checks(final.runtime)
        result = self.checks.check_next(batch.runtime)
        self.assertEqual(result.runtime.state.files, final.runtime.state.files)
        self.assertEqual(result.runtime.state.position["phase"], "awaiting_checks")
        event = self.store.load_event(result.event_id)
        effect = self.store.get_artifact(event.payload_ref)
        log = self.store.get_artifact(result.runtime.state.external_inputs_ref)
        entry = log["entries"][-1]
        record = self.store.get_artifact(entry["record_ref"])
        wrong_ref = self.store.put_artifact({"wrong": "binding"})
        for field in (
            "target_checkpoint",
            "check_contract_hash",
            "evaluator_packet_ref",
            "evidence_ref",
            "status",
        ):
            forged = {**record, field: "unavailable" if field == "status" else wrong_ref}
            forged_entry = {**entry, "record_ref": self.store.put_artifact(forged)}
            with self.subTest(field=field), self.assertRaises(ProjectionError):
                validate_check_result_effect(
                    self.store,
                    batch.runtime.state,
                    result.runtime.state,
                    event,
                    effect,
                    forged_entry,
                )
        with self.assertRaises(ProjectionError):
            validate_check_result_effect(
                self.store,
                batch.runtime.state,
                result.runtime.state,
                event,
                {**effect, "set": {**effect["set"], "decisions_ref": wrong_ref}},
                entry,
            )
        with self.assertRaises(ProjectionError):
            validate_check_result_effect(
                self.store,
                batch.runtime.state,
                result.runtime.state,
                replace(event, id=None, actor="author"),
                effect,
                entry,
            )

    def test_terminal_reward_replays(self):
        ask = self.call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["door"],
                "proposals": [],
                "option_refs": [],
            },
        )
        action = self.writer.submit_action(self.runtime, self.action(ask))
        request = self.writer.step_tool(action.runtime)
        reply = self.author.reply(request.runtime)
        final = self.writer.submit_action(reply.runtime, self.action(content="Final revision."))
        batch = self.checks.request_checks(final.runtime)
        result = self.checks.check_next(batch.runtime)
        with self.assertRaises(WriterRuntimeError):
            self.terminal.reward(result.runtime)
        transition = self.terminal.transition(result.runtime)
        outcome = self.terminal.terminal_outcome(transition.runtime)
        reward = self.terminal.reward(outcome.runtime)
        self.assertEqual(reward.runtime.state.position["phase"], "terminal")
        self.assertEqual(
            self.store.get_artifact(reward.runtime.state.outcome_ref)["reward_status"],
            "available",
        )
        availability = self.store.get_artifact(reward.runtime.state.outcome_ref)
        reward_record = self.store.get_artifact(availability["reward_ref"])
        self.assertEqual(
            (reward_record["numerator"], reward_record["normalization"]), (10000, 10000)
        )
        self.assertEqual(
            self.store.replay(
                "rollout-1",
                self.start,
                (
                    action.commit_id,
                    request.commit_id,
                    reply.commit_id,
                    final.commit_id,
                    batch.commit_id,
                    result.commit_id,
                    transition.commit_id,
                    outcome.commit_id,
                    reward.commit_id,
                ),
            ),
            reward.runtime.checkpoint_id,
        )

        event = self.store.load_event(reward.event_id)
        effect = self.store.get_artifact(event.payload_ref)
        entry = self.store.get_artifact(reward.runtime.state.external_inputs_ref)["entries"][-1]
        forged_reward = {**reward_record, "numerator": reward_record["numerator"] - 1}
        forged_reward_ref = self.store.put_artifact(forged_reward)
        forged_availability = {**availability, "reward_ref": forged_reward_ref}
        forged_availability_ref = self.store.put_artifact(forged_availability)
        forged_state = reward.runtime.state.to_dict()
        forged_state["outcome_ref"] = forged_availability_ref
        with self.assertRaises(ProjectionError):
            validate_terminal_effect(
                self.store,
                outcome.runtime.state,
                EnvironmentStateV1.from_dict(forged_state),
                event,
                {**effect, "set": {**effect["set"], "outcome_ref": forged_availability_ref}},
                entry,
            )
        with self.assertRaises(ProjectionError):
            validate_terminal_effect(
                self.store,
                outcome.runtime.state,
                reward.runtime.state,
                replace(event, id=None, actor="author"),
                effect,
                entry,
            )

    def test_failed_required_check_terminalizes_incomplete_with_declared_reward(self):
        action = self.writer.submit_action(
            self.runtime,
            self.action(self.call("write_file", {"path": "draft.txt", "content": ""}, "empty")),
        )
        written = self.writer.step_tool(action.runtime)
        final = self.writer.submit_action(written.runtime, self.action(content="Final revision."))
        batch = self.checks.request_checks(final.runtime)
        result = self.checks.check_next(batch.runtime)
        outcome = self.terminal.terminal_outcome(result.runtime)
        reward = self.terminal.reward(outcome.runtime)
        terminal_record = self.store.get_artifact(outcome.runtime.state.outcome_ref)
        self.assertEqual(terminal_record["task_status"], "incomplete")
        self.assertEqual(terminal_record["stop_reason"], "required_check_failed")
        availability = self.store.get_artifact(reward.runtime.state.outcome_ref)
        reward_record = self.store.get_artifact(availability["reward_ref"])
        self.assertEqual(reward_record["numerator"], 0)

    def test_feedback_supersedes_only_authorized_requirement(self):
        node = self.writer.graph.node("legacy-writer")
        old_interaction = node.contract.interaction_contract
        old_policy = node.interaction_policy
        old_script = node.script
        update = RequirementUpdateV1(
            id="r2", supersedes="r1", replacement="The blue door is now binding"
        )
        self.store.put_artifact(update.to_dict(), private=True)
        script = ScriptedAuthorV1(
            answers=old_script.answers,
            feedback=(
                {
                    "id": "f1",
                    "utterance": "Please revise for the blue door.",
                    "prerequisite_check_ids": [],
                    "requirement_update_ref": update.identity(),
                },
            ),
        )
        self.store.put_artifact(script.to_dict(), private=True)
        policy = replace(old_policy, mandatory_feedback=("f1",))
        self.store.put_artifact(policy.to_dict())
        interaction = replace(
            old_interaction,
            script_ref=script.identity(),
            interaction_policy_ref=policy.identity(),
            mandatory_feedback=("f1",),
        )
        contract = replace(
            node.contract,
            interaction=interaction,
            budgets=replace(node.contract.budget_contract, max_steps=7),
        )
        self.store.put_artifact(contract.to_dict())
        spec = replace(node.spec, entry_contract=contract.identity())
        instance = replace(self.writer.graph.instance, nodes=(spec,))
        self.store.persist(instance)
        graph = admit_graph(instance, StoreArtifactResolver(self.store))
        state = self.runtime.state.to_dict()
        state["instance_ref"] = instance.identity()
        state["position"]["entry_contract"] = contract.identity()
        state["requirements_ref"] = self.store.put_artifact(
            {
                "record_type": "RequirementLedgerV1",
                "schema": 1,
                "active": {"r1": "Never reveal this hidden requirement"},
                "superseded": {},
            }
        )
        budget = self.store.get_artifact(state["budgets_ref"])
        budget["limits"]["writer_turns"] = 7
        state["budgets_ref"] = self.store.put_artifact(budget)
        self.start = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        self.runtime = self.store.restore(self.start, self.root / "feedback-entry")
        self.writer = TransactionalWriterV1(self.store, graph, "rollout-1", self.start)
        self.author = ScriptedAuthorRuntimeV1(self.writer)
        self.checks = DeterministicChecksV1(self.writer)
        self.terminal = ScriptedTerminalV1(self.writer)

        ask = self.call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["door"],
                "proposals": [],
                "option_refs": [],
            },
        )
        write = self.writer.submit_action(
            self.runtime,
            self.action(
                self.call(
                    "write_file", {"path": "draft.txt", "content": "first draft\n"}, "write-1"
                )
            ),
        )
        written = self.writer.step_tool(write.runtime)
        read = self.writer.submit_action(
            written.runtime, self.action(self.call("read_file", {"path": "draft.txt"}, "read-1"))
        )
        observed = self.writer.step_tool(read.runtime)
        self.assertIn("first draft", str(observed.runtime.context.messages[-1].content))
        action = self.writer.submit_action(observed.runtime, self.action(ask))
        request = self.writer.step_tool(action.runtime)
        answer = self.author.reply(request.runtime)
        draft = self.writer.submit_action(answer.runtime, self.action(content="Draft."))
        feedback_request = self.author.request_feedback(draft.runtime)
        feedback = self.author.reply(feedback_request.runtime)
        ledger = self.store.get_artifact(feedback.runtime.state.requirements_ref)
        self.assertEqual(ledger["active"], {"r2": "The blue door is now binding"})
        self.assertNotIn(
            "Never reveal this hidden requirement", str(feedback.runtime.context.messages)
        )
        revision = self.writer.submit_action(
            feedback.runtime,
            self.action(
                self.call(
                    "write_file",
                    {"path": "draft.txt", "content": "blue door revision\n"},
                    "revision-1",
                )
            ),
        )
        revised = self.writer.step_tool(revision.runtime)
        final = self.writer.submit_action(
            revised.runtime, self.action(content="Revised draft complete.")
        )
        batch = self.checks.request_checks(final.runtime)
        result = self.checks.check_next(batch.runtime)
        transition = self.terminal.transition(result.runtime)
        outcome = self.terminal.terminal_outcome(transition.runtime)
        reward = self.terminal.reward(outcome.runtime)
        self.assertEqual(
            self.store.replay(
                "rollout-1",
                self.start,
                (
                    write.commit_id,
                    written.commit_id,
                    read.commit_id,
                    observed.commit_id,
                    action.commit_id,
                    request.commit_id,
                    answer.commit_id,
                    draft.commit_id,
                    feedback_request.commit_id,
                    feedback.commit_id,
                    revision.commit_id,
                    revised.commit_id,
                    final.commit_id,
                    batch.commit_id,
                    result.commit_id,
                    transition.commit_id,
                    outcome.commit_id,
                    reward.commit_id,
                ),
            ),
            reward.runtime.checkpoint_id,
        )

    def test_feedback_budget_stop_is_incomplete_not_simulator_failure(self):
        node = self.writer.graph.node("legacy-writer")
        script = ScriptedAuthorV1(
            answers=node.script.answers,
            feedback=(
                {
                    "id": "f1",
                    "utterance": "Please revise.",
                    "prerequisite_check_ids": [],
                    "requirement_update_ref": None,
                },
            ),
        )
        self.store.put_artifact(script.to_dict(), private=True)
        policy = replace(node.interaction_policy, mandatory_feedback=("f1",))
        self.store.put_artifact(policy.to_dict())
        interaction = replace(
            node.contract.interaction_contract,
            script_ref=script.identity(),
            interaction_policy_ref=policy.identity(),
            mandatory_feedback=("f1",),
        )
        contract = replace(node.contract, interaction=interaction)
        self.store.put_artifact(contract.to_dict())
        spec = replace(node.spec, entry_contract=contract.identity())
        instance = replace(self.writer.graph.instance, nodes=(spec,))
        self.store.persist(instance)
        graph = admit_graph(instance, StoreArtifactResolver(self.store))
        state = self.runtime.state.to_dict()
        state["instance_ref"] = instance.identity()
        state["position"]["entry_contract"] = contract.identity()
        self.start = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        self.runtime = self.store.restore(self.start, self.root / "budget-entry")
        self.writer = TransactionalWriterV1(self.store, graph, "rollout-1", self.start)
        self.author = ScriptedAuthorRuntimeV1(self.writer)
        self.terminal = ScriptedTerminalV1(self.writer)

        current = self.runtime
        commits = []
        for index in range(2):
            ask = self.call(
                "ask_author",
                {
                    "question": "Which door?",
                    "decision_ids": ["door"],
                    "proposals": [],
                    "option_refs": [],
                },
                f"ask-{index}",
            )
            action = self.writer.submit_action(current, self.action(ask))
            request = self.writer.step_tool(action.runtime)
            reply = self.author.reply(request.runtime)
            current = reply.runtime
            commits.extend((action.commit_id, request.commit_id, reply.commit_id))
        final = self.writer.submit_action(current, self.action(content="Done."))
        stopped = self.terminal.stop_incomplete(final.runtime)
        reward = self.terminal.reward(stopped.runtime)
        outcome = self.store.get_artifact(stopped.runtime.state.outcome_ref)
        self.assertEqual(outcome["stop_reason"], "author_budget")
        availability = self.store.get_artifact(reward.runtime.state.outcome_ref)
        reward_record = self.store.get_artifact(availability["reward_ref"])
        self.assertEqual(reward_record["components"]["nonempty"]["status"], "not_run")
        self.assertEqual(reward_record["numerator"], 0)
        self.assertEqual(
            self.store.replay(
                "rollout-1",
                self.start,
                (
                    *commits,
                    final.commit_id,
                    stopped.commit_id,
                    reward.commit_id,
                ),
            ),
            reward.runtime.checkpoint_id,
        )


class AskAuthorBudgetMatrix(unittest.TestCase):
    def test_tool_author_and_syntax_precedence_in_production_and_recovery(self):
        for tool_remaining in (True, False):
            for author_remaining in (True, False):
                for valid in (True, False):
                    with self.subTest(
                        tool_remaining=tool_remaining,
                        author_remaining=author_remaining,
                        valid=valid,
                    ):
                        fixture = ScriptedFixture()
                        fixture.setUp()
                        try:
                            state = fixture.runtime.state.to_dict()
                            budget = fixture.store.get_artifact(state["budgets_ref"])
                            if not tool_remaining:
                                budget["consumed"]["tool_calls"] = budget["limits"]["tool_calls"]
                            if not author_remaining:
                                budget["consumed"]["author_calls"] = budget["limits"][
                                    "author_calls"
                                ]
                            state["budgets_ref"] = fixture.store.put_artifact(budget)
                            start = fixture.store.save_checkpoint(
                                EnvironmentStateV1.from_dict(state)
                            )
                            runtime = fixture.store.restore(start, fixture.root / "matrix-entry")
                            writer = TransactionalWriterV1(
                                fixture.store, fixture.writer.graph, "rollout-1", start
                            )
                            author = ScriptedAuthorRuntimeV1(writer)
                            ask = fixture.call(
                                "ask_author",
                                {
                                    "question": "Which?",
                                    "decision_ids": ["door" if valid else "unknown"],
                                    "proposals": [],
                                    "option_refs": [],
                                },
                            )
                            action = writer.submit_action(runtime, fixture.action(ask))
                            result = writer.step_tool(action.runtime)
                            commits = [action.commit_id, result.commit_id]
                            if tool_remaining and author_remaining and valid:
                                self.assertEqual(
                                    result.runtime.state.position["phase"], "awaiting_author"
                                )
                                result = author.reply(result.runtime)
                                commits.append(result.commit_id)
                                self.assertEqual(result.runtime.state.continuation["next_call"], 1)
                            else:
                                observation = result.runtime.context.messages[-1].content[0][
                                    "content"
                                ]
                                expected = (
                                    "Tool-call budget exceeded"
                                    if not tool_remaining
                                    else "undeclared decision"
                                    if not valid
                                    else "Author-call budget exceeded"
                                )
                                self.assertIn(expected, observation["error"])
                                self.assertEqual(result.runtime.state.continuation["next_call"], 1)
                            consumed = fixture.store.get_artifact(result.runtime.state.budgets_ref)[
                                "consumed"
                            ]
                            self.assertEqual(consumed["attempted_tool_calls"], 1)
                            self.assertEqual(
                                consumed["tool_calls"],
                                budget["consumed"].get("tool_calls", 0) + int(tool_remaining),
                            )
                            self.assertEqual(
                                fixture.store.replay("rollout-1", start, commits),
                                result.runtime.checkpoint_id,
                            )
                        finally:
                            fixture.doCleanups()


class WriterBudgetLifecycleMatrix(unittest.TestCase):
    def test_every_exhaustion_has_typed_reward_and_offline_replay(self):
        for scenario in ("first", "last_tool", "post_author", "generated", "total"):
            with self.subTest(scenario=scenario):
                fixture = ScriptedFixture()
                fixture.setUp()
                try:
                    state = fixture.runtime.state.to_dict()
                    budget = fixture.store.get_artifact(state["budgets_ref"])
                    if scenario == "first":
                        budget["consumed"]["writer_turns"] = budget["limits"]["writer_turns"]
                    elif scenario in {"post_author", "generated", "total"}:
                        budget["limits"][
                            "total_tokens" if scenario == "total" else "generated_tokens"
                        ] = 1
                    state["budgets_ref"] = fixture.store.put_artifact(budget)
                    start = fixture.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
                    runtime = fixture.store.restore(start, fixture.root / "exhaustion-entry")
                    writer = TransactionalWriterV1(
                        fixture.store, fixture.writer.graph, "rollout-1", start
                    )
                    author = ScriptedAuthorRuntimeV1(writer)
                    terminal = ScriptedTerminalV1(writer)
                    commits = []
                    if scenario == "first":
                        stopped = writer.stop_exhausted(runtime)
                    elif scenario == "last_tool":
                        for index in range(budget["limits"]["writer_turns"]):
                            action = writer.submit_action(
                                runtime,
                                fixture.action(fixture.call("list_dir", {}, f"list-{index}")),
                            )
                            result = writer.step_tool(action.runtime)
                            commits.extend((action.commit_id, result.commit_id))
                            runtime = result.runtime
                        stopped = writer.stop_exhausted(runtime)
                    elif scenario == "post_author":
                        ask = fixture.call(
                            "ask_author",
                            {
                                "question": "Which?",
                                "decision_ids": ["door"],
                                "proposals": [],
                                "option_refs": [],
                            },
                        )
                        action = writer.submit_action(
                            runtime, fixture.action(ask), usage={"completion_tokens": 1}
                        )
                        request = writer.step_tool(action.runtime)
                        reply = author.reply(request.runtime)
                        commits.extend((action.commit_id, request.commit_id, reply.commit_id))
                        runtime = reply.runtime
                        stopped = writer.stop_exhausted(runtime)
                    else:
                        stopped = writer.submit_action(
                            runtime,
                            fixture.action(content="over budget"),
                            usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                        )
                    rewarded = terminal.reward(stopped.runtime)
                    outcome = fixture.store.get_artifact(stopped.runtime.state.outcome_ref)
                    availability = fixture.store.get_artifact(rewarded.runtime.state.outcome_ref)
                    reward = fixture.store.get_artifact(availability["reward_ref"])
                    self.assertEqual(outcome["record_type"], "TerminalOutcomeV1")
                    self.assertEqual(outcome["candidate_checkpoint"], runtime.checkpoint_id)
                    self.assertEqual(outcome["task_status"], "incomplete")
                    self.assertEqual(
                        reward["numerator"],
                        writer.graph.node("legacy-writer").reward_contract.incomplete_score,
                    )
                    self.assertEqual(availability["reward_status"], "available")
                    self.assertEqual(
                        fixture.store.replay(
                            "rollout-1", start, (*commits, stopped.commit_id, rewarded.commit_id)
                        ),
                        rewarded.runtime.checkpoint_id,
                    )
                finally:
                    fixture.doCleanups()
