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

from experimental.loom_serve.tools import agent_trace


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


def call_entries(context_window=8192):
    entries = [
        {
            "format": agent_trace.CALL_FORMAT,
            "provider": "loom",
            "model": "qwen3.8-27b",
            "configuration": {
                "pi_version": "0.82.1",
                "context_window_tokens": context_window,
                "max_output_tokens": 1024,
                "thinking_level": "off",
                "compaction_enabled": True,
                "reserve_tokens": 2048,
                "keep_recent_tokens": 2048,
            },
        }
    ]
    for index, purpose in enumerate(("agent", "compaction", "compaction", "agent")):
        retained = 100 if index == 0 else 0
        entries.extend(
            [
                {
                    "type": "request_start",
                    "index": index,
                    "start_us": 1000 + index * 1000,
                    "purpose": purpose,
                    "api": "openai-completions",
                },
                {
                    "type": "request_end",
                    "index": index,
                    "end_us": 1500 + index * 1000,
                    "stop_reason": "stop",
                    "usage": {
                        "input": 10,
                        "output": 3,
                        "cacheRead": retained,
                        "cacheWrite": 0,
                        "totalTokens": 13 + retained,
                    },
                },
            ]
        )
    entries.append({"type": "recording_end", "requests": 4})
    return entries


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

    def test_calls_preserve_split_compaction_and_cache_reset(self):
        self.write_session(call_entries())
        trace = agent_trace.export_sessions([self.path])
        session = trace.sessions[0]
        self.assertEqual(session.configuration.context_window_tokens, 8192)
        self.assertEqual(
            [request.purpose for request in session.requests],
            ["agent", "compaction", "compaction", "agent"],
        )
        self.assertEqual(
            [request.delay_us for request in session.requests], [0, 500, 500, 500]
        )
        self.assertEqual(
            [request.retained_tokens for request in session.requests], [100, 0, 0, 0]
        )
        output = Path(self.directory.name) / "trace.json"
        output.write_text(json.dumps(trace.to_dict()))
        self.assertEqual(agent_trace.load_trace(output), trace)

    def test_clients_can_share_weights_without_sharing_context_budgets(self):
        self.write_session(call_entries(8192))
        other = Path(self.directory.name) / "other.jsonl"
        other.write_text(
            "".join(json.dumps(entry) + "\n" for entry in call_entries(16384))
        )
        trace = agent_trace.export_sessions([self.path, other])
        self.assertEqual(
            [session.configuration.context_window_tokens for session in trace.sessions],
            [8192, 16384],
        )

    def test_rejects_broken_call_lifecycles(self):
        changes = [
            (1, "index", 1),
            (2, "index", 1),
            (1, "purpose", "other"),
            (1, "api", "other"),
            (2, "end_us", 999),
            (3, "start_us", 1499),
            (2, "type", "request_start"),
            (2, "type", "request_error"),
            (2, "stop_reason", "error"),
            (2, "stop_reason", "aborted"),
            (9, "requests", 3),
        ]
        for index, key, value in changes:
            with self.subTest(key=key, value=value):
                entries = call_entries()
                entries[index][key] = value
                self.write_session(entries)
                with self.assertRaises(ValueError):
                    agent_trace.export_sessions([self.path])
        for length in (1, 2, 3, 9):
            with self.subTest(length=length):
                self.write_session(call_entries()[:length])
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    agent_trace.export_sessions([self.path])
        self.write_session(call_entries() + [{"type": "recording_end", "requests": 4}])
        with self.assertRaisesRegex(ValueError, "follows"):
            agent_trace.export_sessions([self.path])

    def test_rejects_invalid_client_configuration(self):
        for key, value in (
            ("context_window_tokens", True),
            ("max_output_tokens", -1),
            ("compaction_enabled", 1),
            ("pi_version", ""),
        ):
            with self.subTest(key=key):
                entries = call_entries()
                entries[0]["configuration"][key] = value
                self.write_session(entries)
                with self.assertRaises(ValueError):
                    agent_trace.export_sessions([self.path])

    def test_composition_keeps_sources_and_changes_only_release_schedule(self):
        self.write_session(call_entries())
        original = agent_trace.export_sessions([self.path])
        original_data = original.to_dict()
        composed, description = agent_trace.compose_trace(
            original,
            {
                "format": agent_trace.COMPOSITION_FORMAT,
                "instances": [
                    {
                        "source_session": 0,
                        "start_us": 9000,
                        "request_count": 3,
                        "pauses": [{"before_request": 2, "duration_us": 2000}],
                    },
                    {"source_session": 0, "start_us": 100},
                ],
            },
        )
        self.assertEqual(original.to_dict(), original_data)
        self.assertEqual(composed.origin_unix_us, 0)
        self.assertEqual(
            [session.arrival_us for session in composed.sessions], [9000, 100]
        )
        self.assertEqual(
            [request.delay_us for request in composed.sessions[0].requests],
            [0, 500, 2500],
        )
        for session in composed.sessions:
            self.assertEqual(session.source_sha256, original.sessions[0].source_sha256)
            self.assertEqual(session.configuration, original.sessions[0].configuration)
            for index, request in enumerate(session.requests):
                source = original.sessions[0].requests[index]
                self.assertEqual(
                    (
                        request.prefill_tokens,
                        request.retained_tokens,
                        request.output_tokens,
                        request.purpose,
                    ),
                    (
                        source.prefill_tokens,
                        source.retained_tokens,
                        source.output_tokens,
                        source.purpose,
                    ),
                )
        self.assertEqual(description["instances"][1]["request_count"], 4)
        self.assertEqual(description["instances"][1]["pauses"], [])

    def test_composition_rejects_invalid_sources_and_ineffective_controls(self):
        self.write_session(call_entries())
        trace = agent_trace.export_sessions([self.path])
        for overrides in (
            {"source_session": 1},
            {"source_session": True},
            {"start_us": -1},
            {"request_count": 0},
            {"request_count": 5},
            {"context_window_tokens": 16384},
            {"pauses": [{"before_request": 0, "duration_us": 10}]},
            {"pauses": [{"before_request": 4, "duration_us": 10}]},
            {"pauses": [{"before_request": 1, "duration_us": -1}]},
            {"pauses": [{"before_request": 1, "duration_us": 10}] * 2},
            {"pauses": {}},
        ):
            with self.subTest(overrides=overrides):
                instance = {"source_session": 0, "start_us": 0, **overrides}
                with self.assertRaises(ValueError):
                    agent_trace.compose_trace(
                        trace,
                        {
                            "format": agent_trace.COMPOSITION_FORMAT,
                            "instances": [instance],
                        },
                    )


if __name__ == "__main__":
    unittest.main()
