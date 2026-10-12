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
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.models.qwen import benchmark_service
from experimental.loom_serve.testing.http import running_server


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
    ]
    for capacity in (1, 2, 3):
        log_path = arguments.output / f"rejected-{capacity}.log"
        with running_server(
            command + [f"--epoch={capacity}:8", "--draft_depth=3"], log_path
        ) as (process, address):
            if address is not None or process.wait() == 0:
                raise RuntimeError(f"infeasible shape reached readiness: {log_path}")
        expected = (
            "draft_depth=3 requires an epoch shape with at least 4 tokens; "
            f"largest capacity is {capacity}"
        )
        if expected not in log_path.read_text():
            raise RuntimeError(f"startup failed for the wrong reason: {log_path}")
        print(json.dumps({"event": "rejected", "token_capacity": capacity}), flush=True)

    log_path = arguments.output / "target-only-narrow.log"
    with running_server(command + ["--epoch=1:8", "--draft_depth=0"], log_path) as (
        _,
        address,
    ):
        if address is None:
            raise RuntimeError(f"target-only shape failed startup: {log_path}")
    print(json.dumps({"event": "target_only_narrow_ready"}), flush=True)

    log_path = arguments.output / "minimum-verifier.log"
    with running_server(command + ["--epoch=4:8", "--draft_depth=3"], log_path) as (
        _,
        address,
    ):
        if address is None:
            raise RuntimeError(f"minimum verifier shape failed startup: {log_path}")
    print(json.dumps({"event": "minimum_verifier_ready"}), flush=True)

    log_path = arguments.output / "retained.log"
    with running_server(
        command + ["--epoch=1:8", "--epoch=32:8", "--draft_depth=3"], log_path
    ) as (_, address):
        if address is None:
            raise RuntimeError(f"feasible shape table failed startup: {log_path}")
        client_arguments = argparse.Namespace(
            long_lines=8,
            max_tokens=32,
            session_prefix="service-check",
            model="qwen3.8-27b",
            retention="required",
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
