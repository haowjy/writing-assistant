"""The derive lanes export one collision-free writer-input dispatch-key vocabulary."""

from __future__ import annotations

import unittest

from writing_agent.task_graph_derive_author import DERIVES as AUTHOR_DERIVES
from writing_agent.task_graph_derive_context import DERIVES as CONTEXT_DERIVES
from writing_agent.task_graph_derive_outcome import DERIVES as OUTCOME_DERIVES
from writing_agent.task_graph_derive_writer import DERIVES as WRITER_DERIVES


class DeriveRegistryTests(unittest.TestCase):
    def test_lane_keys_are_unique_and_cover_wire_v1_derives(self) -> None:
        lanes = (WRITER_DERIVES, AUTHOR_DERIVES, OUTCOME_DERIVES, CONTEXT_DERIVES)
        keys = [key for lane in lanes for key in lane]
        combined = {key: derive for lane in lanes for key, derive in lane.items()}
        self.assertEqual(len(keys), len(combined))
        self.assertEqual(
            set(combined),
            {
                "WriterTurnV1",
                "WriterTurnV2",
                "ToolObservationV1",
                ("EnvironmentStepV1", "request_author"),
                "AuthorReplyV1",
                ("EnvironmentStepV1", "request_checks"),
                ("EnvironmentStepV1", "commit_transition"),
                ("EnvironmentStepV1", "seal_outcome"),
                ("EnvironmentStepV1", "stop_exhausted"),
                ("EnvironmentStepV1", "publish_reward"),
                "EvaluatorResultV1",
                "ContextOperationInputV1",
                "MemberStartV1",
            },
        )
        self.assertEqual(
            AUTHOR_DERIVES[("EnvironmentStepV1", "request_author")].__module__,
            "writing_agent.task_graph_derive_author",
        )
        for directive in (
            "request_checks",
            "commit_transition",
            "seal_outcome",
            "stop_exhausted",
            "publish_reward",
        ):
            self.assertEqual(
                OUTCOME_DERIVES[("EnvironmentStepV1", directive)].__module__,
                "writing_agent.task_graph_derive_outcome",
            )


if __name__ == "__main__":
    unittest.main()
