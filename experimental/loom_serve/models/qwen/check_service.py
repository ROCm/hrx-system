# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real-weight server configuration and retained HTTP checks.

Runs sequential residencies against one checkpoint. The caller supplies an
already built server and a qualified GPU execution environment. Readiness and
shutdown are explicit; an outer test runner owns any hang deadline.
"""

import argparse
import contextlib
import json
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.models.qwen import benchmark_service


@contextlib.contextmanager
def running_server(command, log_path, on_event=None):
    with log_path.open("x") as log, ThreadPoolExecutor(max_workers=1) as readers:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        address = None
        drain = None

        def record(line):
            log.write(line)
            log.flush()
            if on_event and line.startswith("{"):
                on_event(json.loads(line))

        def consume():
            try:
                for line in process.stderr:
                    record(line)
            finally:
                if on_event:
                    on_event(None)

        try:
            for line in process.stderr:
                record(line)
                if line.startswith('{"event":"ready"'):
                    address = json.loads(line)["address"]
                    drain = readers.submit(consume)
                    break
            yield process, address
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            returncode = process.wait()
            if drain:
                drain.result()
            process.stderr.close()
            if address is not None and returncode:
                raise RuntimeError(
                    f"server retirement failed: {returncode}; {log_path}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    command = [
        str(arguments.server.resolve()),
        f"--model={arguments.model.resolve()}",
        f"--weights={arguments.weights.resolve()}",
        f"--tokenizer={arguments.tokenizer.resolve()}",
        "--prefill_capacity=32",
        "--context_capacity=2048",
        "--rows=4",
        "--port=0",
        "--heartbeat_ms=1000",
        "--mtp",
    ]
    for capacity in (1, 2, 3):
        log_path = arguments.output / f"rejected-{capacity}.log"
        with running_server(
            command + [f"--epoch={capacity}:8", "--mtp_depth=3"], log_path
        ) as (process, address):
            if address is not None or process.wait() == 0:
                raise RuntimeError(f"infeasible shape reached readiness: {log_path}")
        expected = (
            "mtp_depth=3 requires an epoch shape with at least 4 tokens; "
            f"largest capacity is {capacity}"
        )
        if expected not in log_path.read_text():
            raise RuntimeError(f"startup failed for the wrong reason: {log_path}")
        print(json.dumps({"event": "rejected", "token_capacity": capacity}), flush=True)

    log_path = arguments.output / "warm-narrow.log"
    with running_server(command + ["--epoch=1:8", "--mtp_depth=0"], log_path) as (
        _,
        address,
    ):
        if address is None:
            raise RuntimeError(f"warm target-only shape failed startup: {log_path}")
    print(json.dumps({"event": "warm_narrow_ready"}), flush=True)

    log_path = arguments.output / "minimum-verifier.log"
    with running_server(command + ["--epoch=4:8", "--mtp_depth=3"], log_path) as (
        _,
        address,
    ):
        if address is None:
            raise RuntimeError(f"minimum verifier shape failed startup: {log_path}")
    print(json.dumps({"event": "minimum_verifier_ready"}), flush=True)

    log_path = arguments.output / "retained.log"
    with running_server(
        command + ["--epoch=1:8", "--epoch=32:8", "--mtp_depth=3"], log_path
    ) as (_, address):
        if address is None:
            raise RuntimeError(f"feasible shape table failed startup: {log_path}")
        client_arguments = argparse.Namespace(
            long_lines=8, max_tokens=32, session_prefix="service-check"
        )
        barrier = threading.Barrier(5)
        with ThreadPoolExecutor(max_workers=4) as clients:
            futures = [
                clients.submit(
                    benchmark_service.client,
                    client_arguments,
                    urlsplit(f"http://{address}"),
                    index,
                    barrier,
                )
                for index in range(4)
            ]
            barrier.wait()
            for future in as_completed(futures):
                print(
                    json.dumps({"event": "client_complete", **future.result()}),
                    flush=True,
                )
    print(json.dumps({"event": "pass", "clients": 4, "requests": 8}), flush=True)


if __name__ == "__main__":
    main()
