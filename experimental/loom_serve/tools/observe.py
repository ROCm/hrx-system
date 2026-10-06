# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launch a model runner and merge its events with read-only host telemetry.

The JSONL file is created exclusively. Each line has one ordered envelope and
an unchanged server event or a system sample. Diagnostics are preserved too;
logs may contain application output and should be treated as private run data.
SIGINT/SIGTERM reach the owned child, whose normal completion is always awaited.
"""

import argparse
import json
import math
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from experimental.loom_serve.tools.system_stats import SystemStats


class EventLog:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()
        self.start = time.monotonic_ns()
        self.sequence = 0

    def emit(self, source, data):
        with self.lock:
            record = {
                "version": 1,
                "sequence": self.sequence,
                "unix_time_ns": time.time_ns(),
                "elapsed_ns": time.monotonic_ns() - self.start,
                "source": source,
                "data": data,
            }
            self.stream.write(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            )
            self.stream.flush()
            self.sequence += 1


def server_event(line):
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("event"), str):
        return data
    return {"event": "diagnostic", "text": line.rstrip("\n")}


def record_run(command, log_path, interval=1.0, quiet=False):
    stats = SystemStats()
    stopped = threading.Event()
    failures = []
    with log_path.open("x", encoding="utf-8") as stream:
        log = EventLog(stream)
        log.emit(
            "observer",
            {
                "event": "run",
                "command": command,
                "cwd": str(Path.cwd()),
                "host": platform.uname()._asdict(),
                "sample_interval_seconds": interval,
            },
        )
        log.emit("system", stats.inventory())
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            bufsize=1,
        )

        def forward(signum=signal.SIGTERM, _frame=None):
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                # Exit may race forwarding; there is no remaining work to stop.
                pass

        def failed(error):
            failures.append(error)
            stopped.set()
            forward()
            try:
                print(f"loom observer failed: {error}", file=sys.stderr, flush=True)
            except OSError as console_error:
                failures.append(console_error)

        def guarded(operation):
            try:
                operation()
            except Exception as error:
                failed(error)

        def emit(source, data):
            if failures:
                return
            # A failed sink ends the run, but pipe readers keep draining while
            # the child retires device work. Closing a reader here could block
            # the child's final heartbeat and prevent orderly shutdown.
            guarded(lambda: log.emit(source, data))

        def drain(pipe, name, console):
            console_available = True
            for line in pipe:
                emit(name, server_event(line))
                if console_available and (not quiet or failures):
                    try:
                        console.write(line)
                        console.flush()
                    except OSError as error:
                        console_available = False
                        failed(error)

        def sample():
            while not stopped.is_set():
                start = time.monotonic()
                emit("system", stats.sample(process.pid, log_path.parent))
                stopped.wait(max(0, start + interval - time.monotonic()))

        threads = [
            threading.Thread(
                target=guarded,
                args=(lambda: drain(process.stdout, "server.stdout", sys.stdout),),
                name="loom-stdout",
            ),
            threading.Thread(
                target=guarded,
                args=(lambda: drain(process.stderr, "server.stderr", sys.stderr),),
                name="loom-stderr",
            ),
            threading.Thread(target=guarded, args=(sample,), name="loom-system"),
        ]
        handlers = {
            number: signal.signal(number, forward)
            for number in (signal.SIGINT, signal.SIGTERM)
        }
        started = []
        try:
            emit("observer", {"event": "process_start", "pid": process.pid})
            for thread in threads:
                thread.start()
                started.append(thread)
            return_code = process.wait()
        finally:
            if process.poll() is None:
                forward()
            process.wait()
            stopped.set()
            for thread in started:
                thread.join()
            process.stdout.close()
            process.stderr.close()
            for number, handler in handlers.items():
                signal.signal(number, handler)
        if failures:
            raise ExceptionGroup("model observer failed", failures)
        log.emit("observer", {"event": "run_end", "return_code": return_code})
    return return_code if return_code >= 0 else 128 - return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Log child output without mirroring it to the terminal.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a model runner command is required after --")
    if not math.isfinite(arguments.interval) or arguments.interval <= 0:
        parser.error("interval must be finite and positive")
    if sys.platform != "linux":
        parser.error("system telemetry currently requires Linux /proc and /sys")
    raise SystemExit(
        record_run(command, arguments.log, arguments.interval, arguments.quiet)
    )


if __name__ == "__main__":
    main()
