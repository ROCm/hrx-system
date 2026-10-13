# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import errno
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

from loom_serve_testing.logs import LogBatch, LogClient, ModelNotFound

from experimental.loom_serve.tools import observe, system_stats


class SystemStatsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.proc = self.root / "proc"
        self.sys = self.root / "sys"
        self.write(
            "proc/meminfo",
            "MemTotal: 1024 kB\nMemAvailable: 512 kB\nSwapTotal: 256 kB\nSwapFree: 128 kB\n",
        )
        self.write("proc/stat", "cpu 10 0 10 80 0 0 0 0 0 0\n")
        self.write("proc/loadavg", "1.00 2.00 3.00 1/100 7\n")
        for resource in ("cpu", "memory", "io"):
            self.write(
                f"proc/pressure/{resource}",
                "some avg10=1.50 avg60=1.00 avg300=0.50 total=1234\n",
            )
        fields = ["0"] * 22
        fields[0], fields[11], fields[12], fields[17], fields[21] = (
            "S",
            "100",
            "50",
            "5",
            "16",
        )
        self.write("proc/7/stat", "7 (model (GPU) task) " + " ".join(fields) + "\n")
        self.device = self.sys / "devices/gpu"
        self.hwmon = self.device / "hwmon/hwmon0"
        self.write("sys/devices/gpu/power/runtime_status", "active\n")
        self.write("sys/devices/gpu/hwmon/hwmon0/name", "amdgpu\n")
        self.write("sys/devices/gpu/hwmon/hwmon0/temp1_label", "edge\n")
        self.write("sys/devices/gpu/hwmon/hwmon0/temp1_input", "62000\n")
        self.write("sys/devices/gpu/hwmon/hwmon0/power1_average", "100000000\n")
        self.write("sys/devices/gpu/hwmon/hwmon0/power/runtime_status", "unsupported\n")
        (self.sys / "class/hwmon").mkdir(parents=True)
        (self.sys / "class/hwmon/hwmon0").symlink_to(self.hwmon)
        self.write(
            "sys/devices/system/cpu/cpufreq/policy0/scaling_governor", "performance\n"
        )
        self.write(
            "sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq", "3000000\n"
        )
        self.stats = system_stats.SystemStats(self.proc, self.sys)

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def readings(self, sample):
        return {
            channel["label"]: value
            for channel, value in zip(
                self.stats.inventory()["channels"], sample["values"], strict=True
            )
        }

    def test_units_policy_and_host_counters(self):
        sample = self.stats.sample(7, self.root)
        values = self.readings(sample)
        self.assertEqual(values["amdgpu/edge/input"], 62)
        self.assertEqual(values["amdgpu/power1/average"], 100)
        self.assertEqual(values["cpu/policy0/scaling_cur_freq"], 3000)
        self.assertEqual(values["cpu/policy0/scaling_governor"], "performance")
        self.assertEqual(sample["host"]["memory_bytes"]["MemAvailable"], 512 * 1024)
        self.assertEqual(sample["process"]["threads"], 5)
        self.assertEqual(sample["cpu_pressure"]["some"]["avg10"], 1.5)
        self.assertIsNone(sample["host"]["cpu_busy_percent"])
        self.write("proc/stat", "cpu 20 0 10 90 0 0 0 0 0 0\n")
        self.assertEqual(
            self.stats.sample(7, self.root)["host"]["cpu_busy_percent"], 50
        )
        self.assertFalse(sample["errors"])

    def test_suspended_device_is_not_sampled_and_missing_is_not_zero(self):
        self.write("sys/devices/gpu/power/runtime_status", "suspended\n")
        (self.hwmon / "temp1_input").unlink()
        sample = self.stats.sample(7, self.root)
        self.assertIsNone(self.readings(sample)["amdgpu/edge/input"])
        self.assertEqual(set(sample["unavailable"].values()), {"runtime_suspended"})
        self.write("sys/devices/gpu/power/runtime_status", "active\n")
        sample = self.stats.sample(7, self.root)
        self.assertEqual(len(sample["unavailable"]), 1)
        self.assertIsNone(self.readings(sample)["amdgpu/edge/input"])
        self.assertEqual(self.readings(sample)["amdgpu/power1/average"], 100)

    def test_bad_sensor_is_explicit_and_other_readings_survive(self):
        self.write("sys/devices/gpu/hwmon/hwmon0/temp1_input", "not a number\n")
        sample = self.stats.sample(7, self.root)
        self.assertEqual(len(sample["unavailable"]), 1)
        self.assertIsNone(self.readings(sample)["amdgpu/edge/input"])
        self.assertEqual(self.readings(sample)["amdgpu/power1/average"], 100)


class ObserverTest(unittest.TestCase):
    def test_concurrent_events_are_whole_and_ordered(self):
        stream = io.StringIO()
        log = observe.EventLog(stream)
        threads = [
            threading.Thread(
                target=lambda: [
                    log.emit("test", {"event": "value", "index": i}) for i in range(100)
                ]
            )
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([r["sequence"] for r in records], list(range(300)))
        self.assertEqual(
            [r["elapsed_ns"] for r in records], sorted(r["elapsed_ns"] for r in records)
        )

    def client(self):
        client = mock.Mock(spec=LogClient)
        client.address = urlsplit("http://127.0.0.1:8080")
        client.unix_socket = None
        client.models.return_value = ["chat", "image"]
        client.forget.return_value = None

        def poll(model):
            return LogBatch(
                {
                    "scope": "server" if model is None else "model",
                    **({} if model is None else {"model": model}),
                    "identity": "1:1",
                },
                [{"event": "stream"}, {"event": "cursor", "value": "1:1:0"}],
            )

        client.poll.side_effect = poll
        return client

    def test_one_bounded_round_preserves_routing_without_local_sampling(self):
        client, stopped = self.client(), threading.Event()

        class Sink(io.StringIO):
            def write(self, line):
                record = json.loads(line)
                if (
                    record.get("stream", {}).get("model") == "image"
                    and record["data"]["event"] == "cursor"
                ):
                    stopped.set()
                return super().write(line)

        sink = Sink()
        with mock.patch.object(observe, "SystemStats") as stats:
            observe.record_streams(client, observe.EventLog(sink), stopped)
            stats.assert_not_called()
        self.assertEqual(
            client.poll.call_args_list,
            [mock.call(None), mock.call("chat"), mock.call("image")],
        )
        records = [json.loads(line) for line in sink.getvalue().splitlines()]
        self.assertEqual(records[0]["data"]["event"], "observation_begin")
        self.assertEqual(records[-1]["data"]["event"], "observation_end")
        self.assertEqual(
            [r["stream"].get("model") for r in records if r["source"] == "server"],
            [None, None, "chat", "chat", "image", "image"],
        )

    def test_sink_failure_stops_only_this_reader(self):
        client = self.client()

        class Sink(io.StringIO):
            def write(self, line):
                if json.loads(line)["source"] == "server":
                    raise OSError(errno.ENOSPC, "full observer disk")
                return super().write(line)

        with self.assertRaises(OSError) as raised:
            observe.record_streams(client, observe.EventLog(Sink()), threading.Event())
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(client.poll.call_args_list, [mock.call(None)])

    def test_empty_catalog_add_remove_and_same_name_return(self):
        client, stopped, sink = self.client(), threading.Event(), io.StringIO()
        client.models.side_effect = [[], ["chat"], [], ["chat"]]
        client.forget.return_value = "1:1:0"
        polls = 0
        original = client.poll.side_effect

        def poll(model):
            nonlocal polls
            value = original(model)
            if model is not None:
                polls += 1
                if polls == 2:
                    stopped.set()
            return value

        client.poll.side_effect = poll
        observe.record_streams(client, observe.EventLog(sink), stopped, interval=0)
        records = [json.loads(line) for line in sink.getvalue().splitlines()]
        transitions = [
            r["data"] for r in records if r["data"]["event"].startswith("model_")
        ]
        self.assertEqual(
            transitions,
            [
                {"event": "model_available"},
                {
                    "event": "model_unavailable",
                    "model": "chat",
                    "previous_cursor": "1:1:0",
                },
                {"event": "model_available"},
            ],
        )
        self.assertEqual(client.models.call_count, 4)
        client.forget.assert_called_once_with("chat")

    def test_fixed_selection_waits_for_registration_without_duplicate_absence(self):
        client, stopped, sink = self.client(), threading.Event(), io.StringIO()
        polls = 0
        original = client.poll.side_effect

        def poll(model):
            nonlocal polls
            if model is not None:
                polls += 1
                if polls <= 2:
                    raise ModelNotFound(model)
                stopped.set()
            return original(model)

        client.poll.side_effect = poll
        observe.record_streams(
            client, observe.EventLog(sink), stopped, interval=0, models=["chat"]
        )
        records = [json.loads(line) for line in sink.getvalue().splitlines()]
        self.assertEqual(
            [
                r["data"]["event"]
                for r in records
                if r["data"]["event"].startswith("model_")
            ],
            ["model_unavailable", "model_available"],
        )
        client.models.assert_not_called()

    def test_failed_source_is_not_silently_replaced_by_empty_data(self):
        client, sink = self.client(), io.StringIO()
        client.poll.side_effect = ConnectionResetError("server closed")
        with self.assertRaises(ConnectionResetError):
            observe.record_streams(client, observe.EventLog(sink), threading.Event())
        self.assertNotIn("observation_end", sink.getvalue())

    def test_existing_log_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text("valuable evidence\n")
            with mock.patch.object(
                sys, "argv", ["observe", "--url=http://localhost:8080", f"--log={path}"]
            ):
                with self.assertRaises(FileExistsError):
                    observe.main()
            self.assertEqual(path.read_text(), "valuable evidence\n")

    def test_unix_option_reaches_the_independent_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            socket_path = str(Path(directory) / 'serve "local".sock')
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "observe",
                        "--url=http://localhost",
                        f"--log={path}",
                        f"--unix-socket={socket_path}",
                        "--timeout=2.5",
                    ],
                ),
                mock.patch.object(observe, "record_streams") as record,
            ):
                observe.main()
            client = record.call_args.args[0]
            self.assertEqual(client.unix_socket, socket_path)
            self.assertEqual(client.timeout, 2.5)
            self.assertEqual(client.address.geturl(), "http://localhost")


if __name__ == "__main__":
    unittest.main()
