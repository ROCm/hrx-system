# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experimental.loom_serve import agent_trace


def session_entries():
    # These are the entry and usage shapes persisted by pi's message_end path.
    message = {
        "role": "assistant",
        "timestamp": 1000,
        "api": "openai-completions",
        "provider": "loom",
        "model": "qwen3.8-27b",
        "usage": {
            "input": 10,
            "cacheRead": 20,
            "cacheWrite": 2,
            "output": 3,
            "totalTokens": 35,
        },
        "stopReason": "toolUse",
        "content": [
            {"type": "toolCall", "name": "read", "arguments": {"path": "PRIVATE"}}
        ],
    }
    continuation = copy.deepcopy(message)
    continuation.update(
        timestamp=1600, stopReason="stop", content=[{"type": "text", "text": "PRIVATE"}]
    )
    return [
        {"type": "session", "version": 3, "id": "session", "cwd": "PRIVATE"},
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "message": {"role": "user", "content": "PRIVATE"},
        },
        {
            "type": "message",
            "id": "b",
            "parentId": "a",
            "timestamp": "1970-01-01T00:00:01.500Z",
            "message": message,
        },
        {
            "type": "message",
            "id": "c",
            "parentId": "b",
            "message": {"role": "toolResult", "content": "PRIVATE"},
        },
        {
            "type": "message",
            "id": "d",
            "parentId": "c",
            "timestamp": "1970-01-01T00:00:02.000Z",
            "message": continuation,
        },
    ]


class AgentTraceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "private-session.jsonl"

    def write_session(self, entries):
        self.path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    def test_export_round_trip_counts_clock_and_privacy(self):
        self.write_session(session_entries())
        trace = agent_trace.export_sessions([self.path])
        self.assertEqual(trace.origin_unix_us, 1000000)
        self.assertEqual(trace.sessions[0].arrival_us, 0)
        first, second = trace.sessions[0].requests
        self.assertEqual(
            (first.prefill_tokens, first.retained_tokens, first.output_tokens),
            (12, 20, 3),
        )
        self.assertEqual((first.delay_us, first.observed_duration_us), (0, 500000))
        self.assertEqual(
            (second.delay_us, second.observed_duration_us), (100000, 400000)
        )
        encoded = json.dumps(trace.to_dict())
        self.assertNotIn("PRIVATE", encoded)
        self.assertNotIn(self.path.name, encoded)
        output = Path(self.directory.name) / "trace.json"
        output.write_text(encoded)
        self.assertEqual(agent_trace.load_trace(output), trace)

    def test_multiple_clients_preserve_relative_first_arrivals(self):
        self.write_session(session_entries())
        other = Path(self.directory.name) / "other.jsonl"
        entries = session_entries()
        entries[2]["message"]["timestamp"] += 50
        other.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        trace = agent_trace.export_sessions([self.path, other])
        self.assertEqual([session.arrival_us for session in trace.sessions], [0, 50000])

    def test_rejects_unsupported_or_incomplete_histories(self):
        changes = [
            (0, "version", 2),
            (0, "parentSession", "another-file"),
            (4, "parentId", "a"),
            (4, "id", "b"),
            (3, "type", "compaction"),
        ]
        for index, key, value in changes:
            with self.subTest(key=key, value=value):
                entries = session_entries()
                entries[index][key] = value
                self.write_session(entries)
                with self.assertRaises(ValueError):
                    agent_trace.export_sessions([self.path])
        for length in (1, 2, 3, 4):
            with self.subTest(length=length):
                self.write_session(session_entries()[:length])
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    agent_trace.export_sessions([self.path])

    def test_rejects_bad_clock_usage_provider_and_failure(self):
        for key, value in (
            ("timestamp", 1400),
            ("stopReason", "error"),
            ("api", "other"),
            ("model", "other"),
        ):
            with self.subTest(key=key):
                entries = session_entries()
                entries[4]["message"][key] = value
                self.write_session(entries)
                with self.assertRaises(ValueError):
                    agent_trace.export_sessions([self.path])
        for key, value in (
            ("input", -1),
            ("input", True),
            ("output", 0),
            ("totalTokens", 999),
        ):
            with self.subTest(key=key, value=value):
                entries = session_entries()
                entries[2]["message"]["usage"][key] = value
                self.write_session(entries)
                with self.assertRaises(ValueError):
                    agent_trace.export_sessions([self.path])

    def test_rejects_duplicate_input(self):
        self.write_session(session_entries())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            agent_trace.export_sessions([self.path, self.path])


if __name__ == "__main__":
    unittest.main()
