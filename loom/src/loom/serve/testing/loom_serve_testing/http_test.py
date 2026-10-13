# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from loom_serve_testing.http import Events, collect_events, running_server


class HttpTest(unittest.TestCase):
    def test_child_failure_preserves_diagnostics_after_temporary_logs_are_removed(self):
        command = [
            sys.executable,
            "-c",
            "import sys; "
            'print(\'{"event":"ready","address":"127.0.0.1:1"}\', '
            "file=sys.stderr, flush=True); "
            "print('native failure evidence', file=sys.stderr, flush=True); "
            "sys.exit(7)",
        ]
        with self.assertRaises(RuntimeError) as failure:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "server.log"
                with running_server(command, path) as (process, address):
                    self.assertEqual(address, "127.0.0.1:1")
                    self.assertEqual(process.wait(), 7)
        self.assertFalse(path.exists())
        self.assertIn("server retirement failed: 7", str(failure.exception))
        self.assertIn("native failure evidence", str(failure.exception))

    def test_names_are_scoped_and_native_session_ids_are_preserved(self):
        events = Events()
        for model, serial in (("scout", 7), ("researcher", 11)):
            events.record(
                dict(event="session_begin", name="shared", session=serial, model=model)
            )
            events.record(dict(event="admit", session=serial, model=model, request=1))
        self.assertEqual(
            events.wait("admit", "shared", model="researcher")["session"], 11
        )
        self.assertEqual(events.wait("admit", "shared", model="scout")["session"], 7)
        self.assertEqual(len(events.values), 4)
        events.record(None)
        with self.assertRaisesRegex(RuntimeError, "observer closed"):
            events.wait("request_stats", "shared", model="scout")

    def test_gaps_and_restarts_invalidate_history_even_after_a_match(self):
        for loss in (
            dict(event="gap", **{"from": 1, "to": 8}),
            dict(event="stream_reset"),
        ):
            events = Events()
            events.record(dict(event="admit", request=1))
            events.record(loss)
            with self.assertRaisesRegex(RuntimeError, "incomplete telemetry history"):
                events.wait("admit", request=1)
            with self.assertRaisesRegex(RuntimeError, "incomplete telemetry history"):
                _ = events.values

    def test_stop_drains_one_fixed_head_not_a_continuously_advancing_producer(self):
        stopped = threading.Event()
        stopped.set()
        polls = []

        class Client:
            def poll(self, model):
                polls.append(model)
                next_position = len(polls) * 64
                return SimpleNamespace(
                    stream={"scope": "model", "identity": "1:1"},
                    records=[
                        {"event": "stream", "head": next_position + 64},
                        {"event": "sample", "sequence": next_position - 1},
                        {"event": "cursor", "next": next_position},
                    ],
                )

        records = []
        collect_events(Client(), ["chat"], stopped, records.append)
        self.assertEqual(polls, ["chat", "chat"])
        self.assertEqual(records[-2]["sequence"], 127)
        self.assertIsNone(records[-1])
        self.assertEqual(records[0]["model"], "chat")
        self.assertEqual(len(records), 3)

    def test_reset_during_head_capture_cannot_wait_for_the_old_ordinal(self):
        polls = []

        class Client:
            def poll(self, model):
                polls.append(model)
                return SimpleNamespace(
                    stream={"scope": "model", "identity": f"1:{len(polls)}"},
                    records=[
                        {"event": "stream", "head": 1000},
                        {"event": "cursor", "next": 64},
                    ],
                )

        events = Events()
        with self.assertRaisesRegex(RuntimeError, "stream restarted"):
            collect_events(Client(), ["chat"], threading.Event(), events.record)
        self.assertEqual(len(polls), 2)
        self.assertTrue(events.closed)

    def test_observation_failure_closes_waiters_and_propagates(self):
        class Client:
            def poll(self, model):
                raise OSError("reader disconnected")

        events = Events()
        with self.assertRaisesRegex(OSError, "reader disconnected"):
            collect_events(Client(), ["chat"], threading.Event(), events.record)
        with self.assertRaisesRegex(RuntimeError, "observer closed"):
            events.wait("request_stats")


if __name__ == "__main__":
    unittest.main()
