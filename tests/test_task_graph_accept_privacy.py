"""Privacy acceptance over every view, port input, and sampling request."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_rollout_fixtures import (
    AUTHOR_PACKET_CANARY,
    CANARIES,
    EVALUATOR_PACKET_CANARY,
    LEDGER_CANARY,
    build_rollout_fixture,
    make_gatherers,
    run_slice,
)
from writing_agent.task_graph import MessageV1
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import ContextContentV1, ContextRevisionV1


class PrivacyAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_full_feedback_lifecycle_keeps_private_canaries_out_of_writer_surfaces(self) -> None:
        fixture = build_rollout_fixture(self.root / "feedback", mode="feedback")
        ports = []
        requests = []
        old_step_input = fixture.env.step_input

        def capture_step_input(runtime):
            result = old_step_input(runtime)
            port = result[2]
            if port is not None:
                ports.append(port)
            return result

        fixture.env.step_input = capture_step_input

        backend = fixture.gatherers.sampler.backend

        class CaptureSampler:
            def sample(self, prepared):
                requests.append(prepared)
                return backend.sample(prepared)

        tools = fixture.gatherers.tools.provider

        class CaptureTools:
            def execute(self, spec, snapshot, action):
                ports.append((spec, snapshot, action))
                return tools.execute(spec, snapshot, action)

        author = fixture.gatherers.author.source

        class CaptureAuthor:
            def reply(self, port):
                ports.append(port)
                return author.reply(port)

        evaluator = fixture.gatherers.evaluator.evaluator

        class CaptureEvaluator:
            family = evaluator.family

            def evaluate(self, request):
                return evaluator.evaluate(request)

        fixture.gatherers = make_gatherers(
            fixture,
            sampler=CaptureSampler(),
            tools=CaptureTools(),
            author=CaptureAuthor(),
            evaluator=CaptureEvaluator(),
        )
        final = run_slice(fixture)

        # The fixture plants one canary in each private source class; the ledger
        # canary is itself private requirement text, including after supersession.
        private_sources = repr(fixture.entry.reader.private)
        self.assertIn(AUTHOR_PACKET_CANARY, private_sources)
        self.assertIn(EVALUATOR_PACKET_CANARY, private_sources)
        self.assertIn(LEDGER_CANARY, private_sources)
        self.assertEqual(final.state.position["phase"], "terminal")
        self.assertIn("mandatory_feedback", repr(ports))
        self.assertGreaterEqual(len(requests), 5)
        self.assertEqual(len(fixture.sampler_inputs), len(requests))

        context_surfaces = []
        for checkpoint_id in fixture.checkpoint_ids:
            checkpoint = fixture.store.load_checkpoint(checkpoint_id)
            revision = fixture.store.load_context_revision(checkpoint.state.context_ref)
            context = fixture.store.materialize_context(checkpoint.state.context_ref)
            context_surfaces.extend((revision.to_wire(), context))
        for sampler_input in fixture.sampler_inputs:
            self.assertIn(type(sampler_input).__name__, {"SamplerInput"})
        for port in ports:
            if type(port).__name__ == "ToolInput":
                continue
            if type(port).__name__ == "SamplerInput":
                continue
            if type(port).__name__ == "CheckInput":
                continue  # CheckInput is the authorized evaluator boundary.
            if isinstance(port, tuple):
                continue  # ToolRunner's narrower provider arguments.
            self.assertEqual(type(port).__name__, "AuthorInput")

        writer_requests = repr(
            [
                {
                    "request": request.request(),
                    "messages_json": request.messages_json,
                    "tools_json": request.tools_json,
                    "rendering_json": request.rendering_json,
                }
                for request in requests
            ]
        )
        writer_ports = repr(fixture.sampler_inputs) + repr(
            [port for port in ports if type(port).__name__ in {"SamplerInput", "ToolInput"}]
        )
        contexts = repr(context_surfaces)
        for canary in CANARIES.values():
            self.assertNotIn(canary, contexts)
            self.assertNotIn(canary, writer_ports)
            self.assertNotIn(canary, writer_requests)

    def test_forged_entry_context_is_rejected_before_the_first_port_request(self) -> None:
        fixture = build_rollout_fixture(self.root / "first-operation", mode="feedback")
        original = fixture.store.materialize_context(fixture.runtime.state.context_ref)
        content = ContextContentV1(
            parent_ref=None,
            messages=(
                *original.messages,
                MessageV1(
                    role="user",
                    content=(AUTHOR_PACKET_CANARY,),
                    origin="forged-entry:1",
                ),
            ),
            tools=original.tools,
            rendering=original.rendering,
        )
        revision = ContextRevisionV1(
            content_ref=content.identity(),
            event_head=None,
            provenance_refs=(),
        )
        fixture.store.persist(content)
        fixture.store.persist(revision)
        forged_state = replace(fixture.runtime.state, context_ref=revision.identity())
        forged_root = fixture.store.save_checkpoint(forged_state)
        built_requests = []

        def request_after_open() -> None:
            opened = fixture.env.open(forged_root)
            built_requests.append(fixture.env.step_input(opened)[2])

        with self.assertRaises(ProjectionError) as caught:
            request_after_open()
        self.assertIn("state.context_ref", str(caught.exception))
        self.assertEqual(built_requests, [])
        self.assertEqual(fixture.sampler_inputs, [])


if __name__ == "__main__":
    unittest.main()
