# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Attach to a serving endpoint and copy bounded /logz polls to JSONL.

The observer owns no server process, pipe or device work. A blocked sink pauses
only this reader; a failed sink terminates only this observer. Missing starts,
overwritten records and changed stream incarnations remain visible in the log.
Optional --local-pid samples this host, not the host named by a remote URL.
"""

import argparse
import contextlib
import json
import math
import platform
import signal
import sys
import threading
import time
from pathlib import Path

from experimental.loom_serve.http.logs import LogClient
from experimental.loom_serve.tools.system_stats import SystemStats


class EventLog:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()
        self.start = time.monotonic_ns()
        self.sequence = 0

    def emit(self, source, data, *, stream=None):
        with self.lock:
            record = {
                "version": 1,
                "sequence": self.sequence,
                "unix_time_ns": time.time_ns(),
                "elapsed_ns": time.monotonic_ns() - self.start,
                "source": source,
                "data": data,
            }
            if stream is not None:
                record["stream"] = stream
            self.stream.write(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            )
            self.stream.flush()
            self.sequence += 1


def record_streams(
    client,
    log,
    stopped,
    *,
    interval=0.25,
    models=None,
    local_pid=None,
    log_directory=Path("."),
):
    models = client.models() if models is None else models
    log.emit(
        "observer",
        {
            "event": "observation_begin",
            "url": client.address.geturl(),
            "models": models,
            "observer_host": platform.node(),
            "poll_interval_seconds": interval,
            "local_pid": local_pid,
        },
    )
    stats = None
    if local_pid is not None:
        stats = SystemStats()
        log.emit("local_system", stats.inventory())
    while not stopped.is_set():
        start = time.monotonic()
        # One bounded batch per stream per round prevents a hot model from
        # monopolizing this reader. No server-side subscription is created.
        for model in [None, *models]:
            if stopped.is_set():
                break
            batch = client.poll(model)
            for record in batch.records:
                log.emit("server", record, stream=batch.stream)
        if stats is not None and not stopped.is_set():
            log.emit("local_system", stats.sample(local_pid, log_directory))
        stopped.wait(max(0, start + interval - time.monotonic()))
    log.emit("observer", {"event": "observation_end"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Server HTTP(S) origin.")
    parser.add_argument("--log", required=True, help="New JSONL path, or - for stdout.")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Network observation timeout in seconds.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Observe this model; repeat, or omit to discover all.",
    )
    parser.add_argument(
        "--local-pid",
        type=int,
        help="Also sample this host and the given local server PID.",
    )
    arguments = parser.parse_args()
    for name in ("interval", "timeout"):
        value = getattr(arguments, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if arguments.local_pid is not None:
        if arguments.local_pid <= 0 or sys.platform != "linux":
            parser.error("local-pid requires a positive Linux process ID")
    try:
        client = LogClient(arguments.url, arguments.timeout)
    except ValueError as error:
        parser.error(str(error))
    stopped = threading.Event()
    handlers = {
        number: signal.signal(number, lambda *_: stopped.set())
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        path = None if arguments.log == "-" else Path(arguments.log)
        sink = (
            contextlib.nullcontext(sys.stdout)
            if path is None
            else path.open("x", encoding="utf-8")
        )
        with sink as stream:
            record_streams(
                client,
                EventLog(stream),
                stopped,
                interval=arguments.interval,
                models=arguments.model,
                local_pid=arguments.local_pid,
                log_directory=Path.cwd() if path is None else path.parent,
            )
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
