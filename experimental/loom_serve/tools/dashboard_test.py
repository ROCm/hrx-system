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


def counters(requests=40, inputs=4096, outputs=320, batches=50):
    return {
        "requests": requests,
        "completed": requests,
        "rejected": 0,
        "cancelled": 0,
        "failed": 0,
        "input_tokens": inputs,
        "output_tokens": outputs,
        "batches": batches,
    }


def record(data, model="chat", identity="100:1", elapsed_ns=10_000_000_000):
    return {
        "version": 1,
        "sequence": 0,
        "unix_time_ns": 100_000_000_000,
        "elapsed_ns": elapsed_ns,
        "source": "server",
        "stream": {"scope": "model", "model": model, "identity": identity},
        "data": data,
    }


class DashboardTest(unittest.TestCase):
    def records(self):
        stream = io.StringIO()
        log = observe.EventLog(stream)
        log.emit(
            "observer",
            {
                "event": "observation_begin",
                "url": "http://remote:8080",
                "observer_host": "local-sensors",
            },
        )
        log.emit(
            "local_system",
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
            "local_system",
            {
                "event": "system",
                "values": [62, "suspended"],
                "unavailable": {},
                "errors": {},
            },
        )
        metadata = record({})["stream"]
        for data in [
            {"event": "stream", "partial": True},
            {"event": "gap", "from": 0, "to": 128},
            {
                "event": "epoch",
                "epoch": 50,
                "spans": 8,
                "token_capacity": 128,
                "prefill_tokens": 120,
                "decode_tokens": 8,
                "selected_tokens_including_eos": 8,
                "traversals": 1,
                "model_ns": 500_000_000,
            },
            {"event": "counters", "time_ns": 1000, "values": counters()},
            {
                "event": "text_work",
                "prefill_tokens": 4000,
                "decode_tokens": 96,
                "completed_epochs": 50,
                "traversals": 60,
                "model_ns": 10_000_000_000,
            },
        ]:
            log.emit("server", data, stream=metadata)
        log.emit("observer", {"event": "diagnostic", "text": "unsafe\x1b[2J\rtext 雪"})
        log.emit("observer", {"event": "observation_end"})
        return stream.getvalue().encode()

    def test_recorded_view_uses_owner_totals_not_retained_epochs(self):
        view = dashboard.RunView()
        reader = dashboard.LogReader(io.BytesIO(self.records()))
        self.assertEqual(reader.poll(view), 10)
        result = "\n".join(dashboard.render(view))
        self.assertIn("OBSERVER STOPPED (server lifetime independent)", result)
        self.assertIn("4,096 input tokens  320 selected outputs", result)
        self.assertIn("4,000 prompt + 96 decode inputs", result)
        self.assertIn("missed 128 records", result)
        self.assertIn("62.0 degC", result)
        self.assertIn("suspended", result)
        self.assertIn(r"unsafe\u001b[2J\rtext \u96ea", result)
        self.assertNotIn("\x1b", result)
        self.assertEqual(view.streams[("model", "chat")].fill_history[0], 1)
        self.assertIn(" | ", "\n".join(dashboard.render(view, width=140)))
        self.assertIn(
            "/sensor/temp", "\n".join(dashboard.render(view, all_sensors=True))
        )
        self.assertIn("LOCAL SYSTEM  |  local-sensors", result)

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
                self.assertEqual(reader.poll(view), 8)
                self.assertEqual(reader.poll(view), 0)
                self.assertTrue(reader.pending)
                with path.open("ab") as writer:
                    writer.write(records[split:])
                self.assertEqual(reader.poll(view), 2)
                self.assertFalse(reader.pending)
                self.assertIn("observation_end", view.latest)
                self.assertEqual(reader.poll(view), 0)

    def test_speculative_fill_is_a_sample_not_a_total(self):
        view = dashboard.RunView()
        value = json.loads(self.records().splitlines()[5])
        value["data"].update(
            token_capacity=32,
            spans=2,
            prefill_tokens=16,
            decode_tokens=2,
            selected_tokens_including_eos=2,
        )
        view.apply(value)
        self.assertEqual(view.streams[("model", "chat")].fill_history[0], 18 / 32)
        result = "\n".join(dashboard.render(view))
        self.assertIn("Sampled committed fill", result)
        self.assertIn("16P + 2D / 32 slots", result)
        self.assertIn("Work counters unavailable", result)

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
        payload = b"".join(json.dumps(value).encode() + b"\n" for value in records)
        reader = dashboard.LogReader(io.BytesIO(payload))
        self.assertEqual(reader.poll(view, records[5]["elapsed_ns"]), 6)
        self.assertNotIn("observation_end", view.latest)
        self.assertIsNone(view.streams[("model", "chat")].counters)
        self.assertEqual(reader.poll(view), 4)

    def test_models_and_incarnations_do_not_mix_or_count_shared_memory_twice(self):
        view = dashboard.RunView()
        view.apply(
            record(
                {
                    "event": "counters",
                    "time_ns": 1_000_000_000,
                    "values": counters(outputs=100),
                }
            )
        )
        view.apply(
            record(
                {
                    "event": "counters",
                    "time_ns": 2_000_000_000,
                    "values": counters(outputs=120),
                }
            )
        )
        view.apply(
            record(
                {
                    "event": "counters",
                    "time_ns": 2_000_000_000,
                    "values": counters(inputs=0, outputs=0),
                },
                model="pixels",
                identity="100:2",
            )
        )
        view.apply(
            record(
                {
                    "event": "snapshot",
                    "time_ns": 1_500_000_000,
                    "counters": counters(outputs=110),
                    "phase": "execute",
                    "queued": 0,
                    "active": 4,
                }
            )
        )
        chat = view.streams[("model", "chat")]
        self.assertEqual(chat.counters["output_tokens"], 120)
        self.assertEqual(list(chat.output_rates), [20])
        view.apply(
            record(
                {
                    "event": "counters",
                    "time_ns": 3_000_000_000,
                    "values": counters(0, 0, 0, 0),
                },
                identity="200:3",
            )
        )
        self.assertEqual(len(view.streams), 2)
        self.assertEqual(view.streams[("model", "chat")].resets, 1)
        self.assertFalse(view.streams[("model", "chat")].output_rates)
        self.assertEqual(view.streams[("model", "pixels")].counters["requests"], 40)
        for model, identity in [("chat", "200:3"), ("pixels", "100:2")]:
            view.apply(
                record(
                    {
                        "event": "shared_device_memory",
                        "workspace_live_bytes": 2**30,
                        "workspace_committed_bytes": 2**31,
                    },
                    model=model,
                    identity=identity,
                )
            )
        result = "\n".join(dashboard.render(view))
        self.assertEqual(result.count("1.00 GiB live / 2.00 GiB backing"), 2)
        self.assertIn("not a model charge", result)

    def test_history_is_bounded_and_freshness_survives_stopped_device_progress(self):
        view = dashboard.RunView()
        for index in range(100):
            view.apply(
                record(
                    {
                        "event": "snapshot",
                        "phase": "execute",
                        "active": 4,
                        "queued": 3,
                        "counters": counters(),
                        "time_ns": index + 1,
                    }
                )
            )
        self.assertEqual(len(view.streams[("model", "chat")].output_rates), 60)
        result = "\n".join(dashboard.render(view, now_ns=105_000_000_000))
        self.assertIn("record age 5.0s", result)
        self.assertIn("Phase execute  active 4  queued 3  snapshot age 5.0s", result)


if __name__ == "__main__":
    unittest.main()
