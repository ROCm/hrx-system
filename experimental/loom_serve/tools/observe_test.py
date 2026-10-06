# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import errno
import io
import json
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

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
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "run.jsonl"

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

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

    def test_real_child_output_and_nonzero_exit_are_preserved(self):
        command = [
            sys.executable,
            "-c",
            'import sys; print("model output"); print(\'{"event":"heartbeat","active_rows":4}\',file=sys.stderr); sys.exit(7)',
        ]
        self.assertEqual(observe.record_run(command, self.path, quiet=True), 7)
        records = self.records()
        self.assertTrue(
            any(r["data"] == {"event": "heartbeat", "active_rows": 4} for r in records)
        )
        self.assertTrue(
            any(
                r["data"] == {"event": "diagnostic", "text": "model output"}
                for r in records
            )
        )
        self.assertEqual(records[-1]["data"], {"event": "run_end", "return_code": 7})
        self.assertTrue(any(r["data"]["event"] == "system_inventory" for r in records))

    def test_existing_log_is_never_overwritten(self):
        self.path.write_text("valuable evidence\n")
        with self.assertRaises(FileExistsError):
            observe.record_run(["does-not-exist"], self.path)
        self.assertEqual(self.path.read_text(), "valuable evidence\n")

    def test_sink_failure_drains_child_cleanup(self):
        original_open = Path.open
        log_path = self.path

        class FailingSink:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def write(self, text):
                if '"event":"sink_trigger"' in text:
                    raise OSError(errno.ENOSPC, "injected full log filesystem")
                return self.stream.write(text)

            def flush(self):
                self.stream.flush()

        def open_sink(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return FailingSink(stream) if path == log_path else stream

        # More than a pipe's capacity must drain after the sink has failed.
        # Only the child's explicit event triggers failure, not a timer.
        child = (
            "import signal,sys\n"
            "def retire(*args):\n"
            " print('x'*131072,file=sys.stderr,flush=True)\n"
            " print('child-retired',file=sys.stderr,flush=True)\n"
            " sys.exit(0)\n"
            "signal.signal(signal.SIGTERM,retire)\n"
            'print(\'{"event":"sink_trigger"}\',file=sys.stderr,flush=True)\n'
            "signal.pause()\n"
        )
        errors = io.StringIO()
        with (
            mock.patch.object(Path, "open", open_sink),
            contextlib.redirect_stderr(errors),
        ):
            with self.assertRaises(ExceptionGroup) as raised:
                observe.record_run([sys.executable, "-c", child], self.path, quiet=True)
        self.assertTrue(
            any(
                isinstance(error, OSError) and error.errno == errno.ENOSPC
                for error in raised.exception.exceptions
            )
        )
        self.assertIn("child-retired", errors.getvalue())

    def test_broken_console_still_drains_child_cleanup(self):
        class BrokenConsole:
            def write(self, text):
                raise BrokenPipeError(errno.EPIPE, "closed console")

        child = (
            "import signal,sys\n"
            "def retire(*args):\n"
            " print('x'*131072,flush=True)\n"
            " print('child-retired',file=sys.stderr,flush=True)\n"
            " sys.exit(0)\n"
            "signal.signal(signal.SIGTERM,retire)\n"
            "print('READY',flush=True)\n"
            "signal.pause()\n"
        )
        errors = io.StringIO()
        with (
            contextlib.redirect_stdout(BrokenConsole()),
            contextlib.redirect_stderr(errors),
        ):
            with self.assertRaises(ExceptionGroup) as raised:
                observe.record_run([sys.executable, "-c", child], self.path)
        self.assertTrue(
            any(
                isinstance(error, BrokenPipeError)
                for error in raised.exception.exceptions
            )
        )
        self.assertIn("child-retired", errors.getvalue())

    def test_signal_forwarding_waits_for_child_cleanup(self):
        child = 'import signal,sys; signal.signal(signal.SIGTERM,lambda *args: sys.exit(0)); print("READY",flush=True); signal.pause()'
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "experimental.loom_serve.tools.observe",
                "--log",
                str(self.path),
                "--",
                sys.executable,
                "-c",
                child,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(process.stdout.readline(), "READY\n")
            process.send_signal(signal.SIGTERM)
            output, errors = process.communicate()
            self.assertEqual(process.returncode, 0, errors)
            self.assertEqual(output, "")
            self.assertEqual(
                self.records()[-1]["data"], {"event": "run_end", "return_code": 0}
            )
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
