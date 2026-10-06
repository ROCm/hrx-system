# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import json
import tempfile
import unittest
from pathlib import Path

from experimental.loom_serve.tools import dashboard, observe


class DashboardTest(unittest.TestCase):
    def records(self):
        stream = io.StringIO()
        log = observe.EventLog(stream)
        log.emit("observer", {"event": "run", "host": {"node": "test-host"}})
        log.emit(
            "system",
            {
                "event": "system_inventory",
                "channels": [
                    {
                        "label": "amdgpu/edge/input",
                        "unit": "degC",
                        "path": "/sensor/temp",
                    },
                    {
                        "label": "accel0/power/runtime_status",
                        "unit": "text",
                        "path": "/sensor/state",
                    },
                ],
                "errors": [],
            },
        )
        log.emit(
            "system",
            {
                "event": "system",
                "values": [62, "suspended"],
                "unavailable": {},
                "errors": {},
            },
        )
        log.emit(
            "server.stderr",
            {
                "event": "epoch",
                "epoch": 1,
                "spans": 8,
                "token_capacity": 128,
                "prefill_tokens": 120,
                "decode_tokens": 8,
                "selected_tokens_including_eos": 8,
                "traversals": 1,
                "model_ms": 500,
            },
        )
        log.emit(
            "server.stderr", {"event": "diagnostic", "text": "unsafe\x1b[2J\rtext 雪"}
        )
        log.emit("observer", {"event": "run_end", "return_code": 0})
        return stream.getvalue().encode()

    def test_recorded_view_preserves_units_counts_and_terminal_safety(self):
        view = dashboard.RunView()
        reader = dashboard.LogReader(io.BytesIO(self.records()))
        self.assertEqual(reader.poll(view), 6)
        result = "\n".join(dashboard.render(view))
        self.assertIn("FINISHED exit=0", result)
        self.assertIn(
            "120 prompt inputs + 8 decode inputs -> 8 selected outputs", result
        )
        self.assertIn("62.0 degC", result)
        self.assertIn("suspended", result)
        self.assertIn(r"unsafe\x1b[2J\rtext \u96ea", result)
        self.assertNotIn("\x1b", result)
        self.assertEqual(view.fill_history[0], 1)
        self.assertIn(" | ", "\n".join(dashboard.render(view, width=140)))
        self.assertIn(
            "/sensor/temp", "\n".join(dashboard.render(view, all_sensors=True))
        )

    def test_real_file_tail_holds_incomplete_utf8_and_resumes(self):
        records = b"".join(
            json.dumps(json.loads(line), ensure_ascii=False).encode() + b"\n"
            for line in self.records().splitlines()
        )
        split = records.index("雪".encode()) + 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_bytes(records[:split])
            with path.open("rb") as stream:
                view, reader = dashboard.RunView(), dashboard.LogReader(stream)
                self.assertEqual(reader.poll(view), 4)
                self.assertEqual(reader.poll(view), 0)
                self.assertTrue(reader.pending)
                with path.open("ab") as writer:
                    writer.write(records[split:])
                self.assertEqual(reader.poll(view), 2)
                self.assertFalse(reader.pending)
                self.assertIn("run_end", view.latest)
                self.assertEqual(reader.poll(view), 0)

    def test_speculative_fill_counts_only_committed_progress(self):
        view = dashboard.RunView()
        record = json.loads(self.records().splitlines()[3])
        record["data"].update(
            token_capacity=32,
            spans=2,
            prefill_tokens=16,
            decode_tokens=2,
            selected_tokens_including_eos=2,
            rows=[
                {"kind": "prefill", "tokens": 16, "consumed_tokens": 16},
                {"kind": "verify", "tokens": 4, "consumed_tokens": 2},
            ],
            mtp={"rows": 1, "proposed_tokens": 3, "accepted_draft_inputs": 1},
        )
        view.apply(record)
        self.assertEqual(view.fill_history[0], 18 / 32)
        result = "\n".join(dashboard.render(view))
        self.assertIn("Committed fill", result)
        self.assertIn(
            "16 prompt inputs + 2 decode inputs -> 2 selected outputs", result
        )

    def test_corrupt_or_missing_record_fails_at_the_log_boundary(self):
        for payload in (b"not json\n", self.records().split(b"\n", 1)[1]):
            with (
                self.subTest(payload=payload[:20]),
                self.assertRaisesRegex(ValueError, "log record 0 at byte 0"),
            ):
                dashboard.LogReader(io.BytesIO(payload)).poll(dashboard.RunView())

    def test_cursor_stops_without_consuming_later_work(self):
        records = [json.loads(line) for line in self.records().splitlines()]
        view = dashboard.RunView()
        # Use timestamps from this same serialization, not another EventLog.
        payload = b"".join(json.dumps(record).encode() + b"\n" for record in records)
        reader = dashboard.LogReader(io.BytesIO(payload))
        self.assertEqual(reader.poll(view, records[3]["elapsed_ns"]), 4)
        self.assertNotIn("run_end", view.latest)
        self.assertEqual(reader.poll(view), 2)

    def test_history_is_bounded_and_freshness_survives_stopped_gpu_progress(self):
        view = dashboard.RunView()
        reader = dashboard.LogReader(io.BytesIO(self.records()))
        reader.poll(view)
        view.latest.pop("run_end")
        heartbeat = {
            "version": 1,
            "sequence": 6,
            "elapsed_ns": 10_000_000_000,
            "unix_time_ns": 100_000_000_000,
            "source": "server.stderr",
            "data": {
                "event": "heartbeat",
                "phase": "execute",
                "interval_output_tokens_per_second": 0,
            },
        }
        for _ in range(100):
            view.apply(heartbeat)
        self.assertEqual(len(view.output_rates), 60)
        result = "\n".join(dashboard.render(view, now_ns=105_000_000_000))
        self.assertIn("record age 5.0s (no run_end)", result)
        self.assertIn("Phase execute  heartbeat age 5.0s", result)


if __name__ == "__main__":
    unittest.main()
