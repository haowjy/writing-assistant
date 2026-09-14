"""Native event and permission boundaries for the OpenCode comparison pilot."""

import unittest
from pathlib import Path

from scripts.pilot_grok import configuration, parse_events


class GrokPilotTests(unittest.TestCase):
    def test_permissions_only_allow_workspace_files(self):
        permission = configuration(Path("/tmp/case"), True)["permission"]
        self.assertEqual(permission["*"], "deny")
        self.assertEqual(permission["external_directory"], "deny")
        self.assertEqual(permission["read"]["*"], "deny")
        self.assertEqual(permission["read"]["notes/*"], "allow")
        self.assertEqual(permission["read"]["tmp/case/notes/*"], "allow")
        self.assertEqual(permission["edit"]["drafts/*"], "allow")
        self.assertEqual(configuration(Path("/tmp/case"), False)["permission"], {"*": "deny"})

    def test_native_reasoning_tools_and_errors_are_distinct(self):
        events = [
            {"type": "reasoning", "sessionID": "session", "part": {"text": "thinking"}},
            {
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "callID": "c1",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "/tmp/case/kb/index.md"},
                        "output": "The keeper has the key.",
                    },
                },
            },
            {"type": "text", "part": {"text": "prose"}},
            {
                "type": "step_finish",
                "part": {"reason": "stop", "tokens": {"input": 5, "output": 3, "reasoning": 2}},
            },
            {"type": "error", "error": {"message": "transport failed"}},
        ]
        messages, trace, usage, final, session, reason, errors = parse_events(events)
        self.assertEqual(messages[0]["thinking"], "thinking")
        self.assertEqual(final, "prose")
        self.assertEqual(trace[0]["native_tool"], "read")
        self.assertEqual(trace[0]["call"]["function"]["name"], "read_file")
        self.assertEqual(trace[0]["observation"]["result"], "The keeper has the key.")
        self.assertEqual(usage, {"input": 5, "output": 3, "reasoning": 2})
        self.assertEqual((session, reason), ("session", "stop"))
        self.assertEqual(errors, [{"message": "transport failed"}])
