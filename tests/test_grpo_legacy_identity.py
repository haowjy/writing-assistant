"""Legacy plans retain the identity published by the DAPO baseline."""

import unittest

from tests.test_grpo import REVISION, task
from writing_agent.catalog import fingerprint
from writing_agent.grpo import GRPOSettings, inspect_grpo


class LegacySettingsIdentityTests(unittest.TestCase):
    def test_probe_settings_identity_matches_9cb9944(self):
        plan = inspect_grpo(
            [task()],
            "unused",
            settings=GRPOSettings(revision=REVISION),
            reward_spec={"id": "test", "config": {}, "mode": "mechanical-only-smoke"},
            admission={"mode": "engineered-fixture", "label": "test"},
        )

        # Computed with inspect_grpo at 9cb9944 in a detached /tmp worktree.
        self.assertEqual(
            fingerprint(plan["settings"]),
            "f5c748667101c6359fdf1481cc0482c603ff126a8250f12fc4d45207208bb71b",
        )


if __name__ == "__main__":
    unittest.main()
